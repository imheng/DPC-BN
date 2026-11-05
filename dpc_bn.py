"""DPC-BN：已知簇数的七步聚类。样本和簇编号都从 0 开始。"""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler


def step1_distances(samples):
    """步骤 1：按列标准化，再计算样本之间的欧氏距离。"""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or min(samples.shape) < 1 or not np.isfinite(samples).all():
        raise ValueError('输入应为非空有限数值矩阵，每行一个样本。')
    points = StandardScaler().fit_transform(samples)
    return points, cdist(points, points)


def step2_density(points, distances, percentile):
    """步骤 2：用无序点对距离的指定百分位数作为高斯核带宽。"""
    if not 0 < percentile <= 100:
        raise ValueError('百分位数必须在 (0,100] 内。')
    pairs = pdist(points)
    bandwidth = np.percentile(pairs, percentile) if pairs.size else 1.0
    if bandwidth <= 0:
        positive = pairs[pairs > 0]
        bandwidth = positive.min() if positive.size else 1.0
    density = np.empty(len(points))
    for start in range(0, len(points), 256):
        block = slice(start, start + 256)
        # 对角项 exp(0)=1，减去自身贡献。
        density[block] = np.exp(-(distances[block] / bandwidth)**2).sum(axis=1) - 1
    return bandwidth, np.maximum(density, 0)


def step3_peaks(distances, density, neighbors):
    """步骤 3：在每个点的近邻中选出密度排名最高的局部峰。"""
    if not isinstance(neighbors, (int, np.integer)) or neighbors < 1:
        raise ValueError('近邻数必须为正整数。')
    ranked = np.argsort(-density, kind='stable')  # 同密度时按原始索引排序
    rank = np.empty(len(density), dtype=int)
    rank[ranked] = np.arange(len(density))
    nearby = distances.copy()
    np.fill_diagonal(nearby, np.inf)  # 排除自身，同时保留其他零距离样本
    nearest = np.argsort(nearby, axis=1, kind='stable')[:, :min(neighbors, len(density)-1)]
    return np.flatnonzero(np.all(rank[:, None] < rank[nearest], axis=1))


def step4_tree(distances):
    """步骤 4：Prim 最小生成树。两点树路径上最大边长即极小极大路径距离。"""
    n = len(distances)
    tree = [[] for _ in range(n)]
    visited = np.zeros(n, dtype=bool)
    cost = np.full(n, np.inf)
    cost[0] = 0
    parent = np.full(n, -1)
    for _ in range(n):
        node = np.argmin(np.where(visited, np.inf, cost))
        visited[node] = True
        if parent[node] >= 0:
            other = parent[node]
            tree[node].append((other, cost[node]))
            tree[other].append((node, cost[node]))
        improve = (~visited) & (distances[node] < cost)
        cost[improve] = distances[node, improve]
        parent[improve] = node
    return tree


def bottleneck_from(tree, root):
    """只计算一个中心到全体样本的极小极大路径距离，无须存储全体点对极小极大路径距离。"""
    values = np.zeros(len(tree))
    stack = [(root, -1)]
    while stack:
        node, parent = stack.pop()
        for neighbor, weight in tree[node]:
            if neighbor != parent:
                values[neighbor] = max(values[node], weight)
                stack.append((neighbor, node))
    return values


def step5_centers(tree, density, peaks, clusters):
    """步骤 5：先取最高密度点，再最大化到已有中心的最小极小极大路径距离。"""
    first = int(np.argmax(density))
    centers = [first]
    columns = [bottleneck_from(tree, first)]
    separation = columns[0].copy()
    used = np.zeros(len(tree), dtype=bool)
    used[first] = True
    fallback = 0
    while len(centers) < clusters:
        pool = peaks[~used[peaks]]
        if pool.size == 0:
            pool = np.flatnonzero(~used)
            fallback += 1
        # 分离度并列：密度高者优先，仍并列则索引小者优先。
        chosen = min(pool, key=lambda i: (-separation[i], -density[i], i))
        centers.append(chosen)
        used[chosen] = True
        column = bottleneck_from(tree, chosen)
        columns.append(column)
        separation = np.minimum(separation, column)
    return np.asarray(centers), np.column_stack(columns), fallback


def step6_assign(distances, centers, bottlenecks):
    """步骤 6：先比较极小极大路径距离，再比较欧氏距离，最后按中心入选次序。"""
    tied = bottlenecks == bottlenecks.min(axis=1, keepdims=True)
    labels = np.where(tied, distances[:, centers], np.inf).argmin(axis=1)
    labels[centers] = np.arange(len(centers))
    return labels


def step7_result(labels, centers, peaks, density, bandwidth, fallback):
    """步骤 7：输出标签、中心及可检查的中间结果。"""
    return dict(labels=labels, centers=centers, peaks=peaks, density=density,
                dc=bandwidth, fallback_count=fallback, sizes=np.bincount(labels))


def fit(samples, clusters, percentile=2, neighbors=5):
    points, distances = step1_distances(samples)                      # 1. 标准化与距离
    if not isinstance(clusters, (int, np.integer)) or not 1 <= clusters <= len(points):
        raise ValueError('给定簇数必须为 1 至样本数之间的整数。')
    dc, density = step2_density(points, distances, percentile)        # 2. 局部密度
    peaks = step3_peaks(distances, density, neighbors)                # 3. 局部密度峰
    tree = step4_tree(distances)                                      # 4. 构建最小生成树
    centers, columns, fallback = step5_centers(tree, density, peaks, clusters)  # 5. 选择中心
    labels = step6_assign(distances, centers, columns)                # 6. 分配样本
    return step7_result(labels, centers, peaks, density, dc, fallback) # 7. 输出结果


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mat', type=Path, help='MAT 文件，data 矩阵默认每行一个样本')
    parser.add_argument('--clusters', type=int, required=True, help='人为给定的簇数 C')
    parser.add_argument('--transpose', action='store_true', help='若每列一个样本则转置')
    parser.add_argument('--percentile', type=float, default=2)
    parser.add_argument('--neighbors', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True, help='NPZ 输出路径')
    args = parser.parse_args()
    samples = loadmat(args.mat, variable_names=['data'])['data']
    result = fit(samples.T if args.transpose else samples, args.clusters,
                 args.percentile, args.neighbors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        np.savez_compressed(stream, **result)
    print('中心索引:', result['centers'], '簇大小:', result['sizes'])


if __name__ == '__main__':
    main()
