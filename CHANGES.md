# TRCC v1 — changes vs v0 (`TRCC-old/`)

## Algorithmic

| # | v0 issue | v1 fix |
|---|---|---|
| 1 | Cytokine signal `α·S_i` was point-local in the assignment score, identical across clusters → cancelled in `argmax` → signal had no effect on clustering. | Replaced centroid+score architecture with density-peak discovery; the signal `S_i` and the peakiness score `γ_i = S_i · δ_i` directly determine cluster anchors and parent-propagation. |
| 2 | Greedy single-pass merge with stale centroids, order-dependent. | Iterative merge to fixed point; centroids and cluster signals are recomputed every iteration. |
| 3 | Convergence by hashing rounded centroid means — brittle. | No outer iteration needed in v1 (the algorithm is feed-forward except for the merge fixed point). |
| 4 | `_clusters_to_labels` was O(N²k) with last-write-wins ties; many points ended up `-1` accidentally. | Labels are an integer array maintained directly through propagation; no recovery search. |
| 5 | Cytokine signal recomputed every outer iteration. | Computed once. |
| 6 | Random k-means-style init, sensitive to seeds. | Density peaks: top-`k` by `γ` — deterministic, anchored to the data. |
| 7 | Per-point Python loop applying Gaussian to neighbor distances. | Single vectorized kNN query + matrix Gaussian. |
| 8 | Per-cluster Python list of points then re-stacked each iteration. | Integer label array; centroids recomputed by `np.bincount`-style aggregation. |
| 9 | Optuna full silhouette O(N²). | `autotune.tune` sub-samples silhouette to ≤4000 points; supports ARI when labels available. |
| 10 | Label recovery in benchmark used float `==` and O(N²) `np.where`. | Algorithms return labels directly. |
| 11 | Two divergent implementations (`TRCC.py` Torch, `TRCC_M1.py` NumPy) with different defaults. | Single canonical NumPy + sklearn implementation. |
| 12 | No `predict()` for unseen data. | `predict()` assigns to nearest density-anchored centroid. |
| 13 | Filtered points silently disappeared. | DBSCAN-style `-1` label for noise. |
| 14 | Hyperparameters unscaled (depend on feature scale). | `n_neighbors`, `sigma`, `merge_threshold` all default to `"auto"` derived from kth-NN distances. |
| 15 | `random_state` hardcoded inside the algorithm. | Exposed as constructor param. |
| 16 | Fixed-radius neighbors fail on varying density. | kNN-based signal — adapts locally. |
| 17 | Merge ignored connectivity / density between clusters. | Mutual-reachability + path-density bottleneck check; clusters merge only when joined by a high-density ridge. |
| 18 | Hard argmax in assignment. | Density-peak parent propagation gives smooth label assignments along density gradient. |
| 19 | Benchmarks only used silhouette; under-rewards non-convex methods. | Benchmark suite reports ARI, NMI, silhouette, runtime, and emits CSV + markdown + 4-panel plots. |

## Engineering

- Sklearn-compatible class (`TRCC().fit(X)` / `.fit_predict(X)` / `.predict(Xnew)`).
- One package: `trcc/{core,benchmark,autotune}.py`.
- 12 unit tests covering shape, determinism, blob recovery, `predict`
  consistency, noise labeling, high-D, explicit `n_clusters`, input
  validation, and a moons regression check.
- Reproducible benchmark harness: `python -m trcc.benchmark --out results --data Data`.
- Original code preserved untouched in `TRCC-old/`.

## Performance

- Single-shot signal computation + vectorized parent search.
- Merge step refits `NearestNeighbors` once and batches all candidate
  path-density evaluations into a single kNN query per iteration.
- 37-dataset benchmark (incl. 9k+ point CSVs) runs in <30 s end-to-end.

## What is unchanged on purpose

- TRCC's identity — the cytokine-signal metaphor — is preserved and now
  load-bearing.
- The user-facing concept (`α`/`β`-style tuning) is replaced by physically
  meaningful auto-defaults plus an explicit Optuna tuner; this is more
  honest than dimensionless knobs.
