# Synthetic benchmark

This directory contains the fixed-parameter synthetic benchmark for DPC-BN, including the operational Gaussian DPC baseline and a four-way ablation. All methods receive the reference cluster count. Reference labels are otherwise used only for evaluation.

## Data and verification

Obtain the synthetic MATLAB files from [Clustering-Datasets](https://github.com/imheng/Clustering-Datasets). This is a multi-source collection, not exclusively the SIPU benchmark. Raw data are not redistributed here. `dataset_manifest.json` records 77 historical filenames, shapes, label counts, transpose decisions, byte-level SHA-256 hashes, content hashes, and duplicate mappings. The hashes identify the actual experiment inputs; a current repository commit is not assumed to identify the historical experiment version.

From the repository root:

```bash
python -m pip install -r benchmark/requirements.txt
python benchmark/verify_data.py --data /path/to/Clustering-Datasets/synthetic
python benchmark/reproduce.py --data /path/to/Clustering-Datasets/synthetic --output benchmark/output
python benchmark/summarize.py --results benchmark/output/results.csv --output benchmark/output/summary
```

Stop and resolve input differences if verification fails. The runner processes all `.mat` files in the supplied directory, so use the 77-file synthetic directory without added files. Features are read from `data` and reference labels from `label`; an orientation mismatch triggers transpose.

## Protocol

- Standardize columns with population standard deviations.
- Gaussian bandwidth uses pairwise distance percentiles 1, 2, and 3; primary results use 2.
- Local peak neighborhood is fixed at five (capped at n−1).
- Four conditions: `DPC`, `centers_only`, `assignment_only`, `DPC_BN`.
- Exclude exact feature-and-label duplicates and Jain from aggregate summaries, leaving 73 datasets. Jain is retained in raw results as a development example.
- Retain every sample and every distinct label, including possible noise labels.
- Differences within 1e-12 count as ties. Sign tests are exploratory because related benchmark families may not be independent.

The runner writes per-dataset predictions and 924 metric rows (77 × 3 × 4). `bn_core.py` also checks MST minimax distances against an independent Floyd calculation, including coincident observations. The summary script regenerates overview and ablation figures from the rerun, with presentation formatting that may differ from manuscript figures.

## Archived reference results

`reference_results/` contains the original experiment tables. Its `results.csv` uses `BN_centers_DPC_assignment` for the centers-only condition; assignment-only results are in `ablation.csv`. The new standalone runner places all four conditions in one file with the names above. These schemas should not be compared row-for-row without mapping condition names.

Primary reference means are 0.572727 (DPC), 0.523096 (centers only), 0.673161 (assignment only), and 0.744385 (both). DPC-BN has 38 wins, 12 ties, and 23 losses on the 73-dataset primary comparison.

`extension.csv` preserves tabular and Digits reference scores; the synthetic runner does not reproduce those external-data experiments. `top12.csv` contains retrospectively selected favorable cases, not an independent benchmark.

## Limitations

This package does not add single-linkage/MST-cut competitors or a tie-break sensitivity study. It does not claim new experimental results. Complete per-file original-source attribution remains to be assembled. Runtime is quadratic in memory, and large inputs can exceed available RAM. The experiments can use a free-threaded Python environment, but no parallel speedup is claimed.
