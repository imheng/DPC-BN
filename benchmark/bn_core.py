import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json,csv,time,hashlib,platform
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

def mst(D):
    # Dense Prim, O(n^2); exact zero-weight edges preserved, no scipy zero-as-absent convention.
    n=len(D);used=np.zeros(n,bool);best=np.full(n,np.inf);best[0]=0;parent=np.full(n,-1);adj=[[] for _ in range(n)]
    for _ in range(n):
        i=int(np.argmin(np.where(used,np.inf,best)));used[i]=True
        if parent[i]>=0:
            j=int(parent[i]);w=float(D[i,j]);adj[i].append((j,w));adj[j].append((i,w))
        update=(~used)&(D[i]<best);best[update]=D[i,update];parent[update]=i
    return adj

def column(adj,root):
    result=np.zeros(len(adj));stack=[(root,-1,0.)]
    while stack:
        i,p,v=stack.pop();result[i]=v
        for j,w in adj[i]:
            if j!=p:stack.append((j,i,max(v,w)))
    return result

def fit(D,adj,rho,C,neighbors):
    n=len(D);rank=np.empty(n,int);rank[np.argsort(-rho,kind='stable')]=np.arange(n)
    candidates=np.flatnonzero(np.all(rank[:,None]<rank[neighbors],axis=1));first=int(np.argmax(rho));centers=[first];columns=[column(adj,first)];selected=np.zeros(n,bool);selected[first]=True;fallback=0
    separation=columns[0].copy()
    while len(centers)<C:
        ids=candidates[~selected[candidates]]
        if not len(ids):ids=np.flatnonzero(~selected);fallback+=1
        chosen=min(ids,key=lambda i:(-separation[i],-rho[i],int(i)))
        centers.append(int(chosen));selected[chosen]=True;col=column(adj,chosen);columns.append(col);separation=np.minimum(separation,col)
    scores=np.column_stack(columns);pred=np.where(scores==scores.min(axis=1)[:,None],D[:,centers],np.inf).argmin(axis=1)
    pred[centers]=np.arange(C)
    return pred,np.array(centers),len(candidates),fallback

def tests():
    for X in [np.array([[0.],[0.],[1.],[3.]]),np.random.default_rng(4).normal(size=(17,3))]:
        D=cdist(X,X);F=D.copy()
        for k in range(len(X)):F=np.minimum(F,np.maximum(F[:,k,None],F[None,k,:]))
        T=mst(D);B=np.column_stack([column(T,i) for i in range(len(X))]);assert np.allclose(B,F)
    print('Prim and zero-weight minimax checks passed',flush=True)
