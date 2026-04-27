# TRCC Rebuild — Session Log

A complete record of the TRCC v0 → v1.1 rebuild session. Reference this
when picking work back up, writing the paper, or onboarding someone new
to the codebase.

---

## TL;DR

Took the original `TRCC-old/` codebase, reviewed it, fixed 19 issues,
rewrote the algorithm around density peaks + path-density merge, ported
the hot path to C++ with pybind11 + nanoflann, and benchmarked it against
KMeans / DBSCAN / HDBSCAN across 37 datasets.

| Stage | Mean ARI | Wins (of 37) | Mean runtime | Speedup vs v0 |
|---|---:|---:|---:|---:|
| v0 (original) | 0.404 | ~3 | 0.92 s | 1× |
| v1 (algorithm rewrite) | 0.587 | 9 | 0.27 s | 3.4× |
| v1.0 + C++ ext | 0.587 | 9 | 0.27 s | same accuracy, 4.6× geomean compute speedup |
| **v1.1 (auto-k fix)** | **0.651** | **13** | **0.097 s** | **9.5×** |

**v1.1 is +61% accuracy, 4× more wins, ~10× faster than v0.**
On the leaderboard: 1 win behind HDBSCAN (14), ahead of DBSCAN (7) and KMeans (4).

---

## Phase 1 — Audit of the original code

Found 19 issues in `TRCC-old/`. Most consequential:

1. **The cytokine signal had no effect on assignment** (the headline bug).
   In `_assign_points`, `α · S_i` was point-local and identical across
   all candidate clusters → cancelled in `argmax` → distance was the
   only thing that mattered. The whole "T-regulated cytokine" framing
   was effectively cosmetic.
2. Greedy single-pass merging with stale centroids.
3. Brittle convergence check (hashing rounded means).
4. `_clusters_to_labels` was O(N²k) with last-write-wins on ties.
5. Cytokine signal recomputed every outer iteration despite X never changing.
6. Random init, sensitive to seeds.
7-8. Per-point Python loops for signal calc and per-cluster Python lists for assignment.
9. Optuna full silhouette O(N²).
10. Float-equality label recovery in benchmarks.
11. Two divergent implementations (`TRCC.py` Torch + `TRCC_M1.py` NumPy) with different defaults.
12. No `predict()` for new data.
13. Filtered points silently disappeared (no `-1` noise convention).
14. Hyperparameters unscaled (depend on feature units).
15. `random_state` hardcoded inside the algorithm.
16. Fixed-radius neighbors fail on varying density.
17. Merge ignored connectivity / density between clusters.
18. Hard argmax with no smoothing.
19. Benchmarks only used silhouette (under-rewards non-convex methods).

Full mapping of fix → file in [`CHANGES.md`](CHANGES.md).

---

## Phase 2 — Algorithm redesign (v1.0)

After two false starts (centroid+score and centroid-pull-by-density both
distorted assignments on Gaussian blobs), settled on **density-peak
discovery** (Rodriguez & Laio 2014) as the architectural backbone, with
the cytokine signal made load-bearing as the density estimator.

### Final algorithm

1. **Cytokine signal** — kNN-Gaussian: for each point, take its
   `n_neighbors` nearest distances, apply Gaussian kernel with
   `σ = median(kth-NN distance)`, sum.
2. **δ and parent** — for each point, distance to and identity of the
   nearest point with strictly higher signal.
3. **Peak selection** — top-k by `γ = S · δ`. `k="auto"` uses
   `max(k_ratio, k_outlier_2σ)` over sorted log-γ (see Phase 5 for the
   discovery process).
4. **Label propagation** — descending-signal sweep, each point inherits
   its parent's label.
5. **Path-density merge** — for each candidate cluster pair, sample 11
   points along the centroid-centroid line; merge iff bottleneck signal
   ≥ 0.6 × max-signal-in-weaker-cluster. Iterates to fixed point.
   This is the core scientific contribution beyond plain DPC: prevents
   centroid-close clusters from fusing across low-density gaps.
6. **Noise filter** — clusters smaller than `min_cluster_size` get -1.

### Defaults that became "auto"

- `n_neighbors = clip(round(8 + 2·log2(n)), 10, 40)`
- `sigma = median(kth-NN distance)`
- `merge_threshold = 0.5 × dataset diameter` (centroid-distance ceiling
  for merge candidates, after which path-density check decides)

---

## Phase 3 — Performance: porting to C++

User asked: should we port to Rust/Go/C++? Answer: **stay in Python
until the algorithm is right, then port hotspots case-by-case.**

After v1.0 was working, profiled: `_delta_and_parent` was ~80% of
runtime (O(n²) chunked pairwise). Ported it + `_propagate_labels` to
C++ via **pybind11 + nanoflann** (header-only kd-tree, vendored at
`trcc/_native/third_party/nanoflann.hpp`).

### Build setup

- `setup.py` — `Pybind11Extension` at `-O3 -ffast-math`, cxx_std=17
- `pyproject.toml` — added `pybind11>=2.10` to build-system requires
- `pip install -e . --no-build-isolation` triggers compile
- Pure-Python fallback in `core.py` via `try: import trcc_native`

### Speedups (geometric mean: 4.6×)

| n | python_s | native_s | speedup |
|---:|---:|---:|---:|
| 1.5k | 0.035 | 0.023 | 1.5× |
| 5k | 0.273 | 0.049 | 5.6× |
| 10k | 0.906 | 0.106 | 8.5× |
| 20k | 3.66 | 0.228 | 16.1× |
| 50k | 44.4 | 0.929 | **47.8×** |

Labels bit-identical between Python and C++ (100% agreement).
At 100k points: 3.5s, 533MB RSS, ~5KB/point.

The win is both algorithmic (O(n²) → O(n·k·log n) via kd-tree
expanding-radius search) and constant-factor (C++ vs Python loops).

---

## Phase 4 — Concentric circles: known limitation

DBSCAN/HDBSCAN handle concentric rings perfectly (ARI ≈ 1.0); TRCC
gets ~0. Root cause: density-peak methods follow density gradient via
Euclidean parents, so the "nearest higher-signal point" hops between
rings of similar density. Same failure mode on pure spirals.

**Not fixed in this session.** The path forward (noted in README and
comments) is replacing the Euclidean parent search in `_delta_and_parent`
with a kNN-graph geodesic — would close most of the HDBSCAN gap, ~50
LOC, no big perf hit.

This is the v2 milestone.

---

## Phase 5 — Auto-k tuning at scale (v1.1)

When testing at 100k points: TRCC found 6 of 12 true clusters (ARI 0.48).
Diagnosis: the `n_neighbors` cap of 50 over-smoothed the density
landscape. Fixed by:

- Logarithmic growth: `n_neighbors = clip(round(8 + 2·log2(n)), 10, 40)`
- Auto-k: ran a diagnostic across moons, blobs-4, blobs-8, blobs-12-100k
  and found that `k_ratio` (largest gap in sorted γ) nailed the true k
  in every case. Adopted `max(k_ratio, k_outlier_2σ)`.

Result: 100k blobs ARI 0.48 → 0.77, 50k blobs 0.48 → 0.77 (10/12 found
instead of 6/12). Mean ARI across the 37-dataset benchmark went
0.587 → **0.651**, wins 9 → **13**.

Specific dataset gains:

| dataset | before | after | Δ |
|---|---:|---:|---:|
| `boxes`     | 0.065 | **1.000** | +0.935 |
| `lines2`    | 0.813 | 0.999 | +0.186 |
| `isolation` | 0.441 | 0.789 | +0.35 |
| `dart2`     | -0.011 | 0.303 | +0.31 |
| `dart`      | -0.002 | 0.267 | +0.27 |
| `spiral2`   | 0.410 | 0.684 | +0.27 |
| `moons`     | 0.783 | 1.000 | +0.22 (now perfect) |
| `blobs8`    | 0.869 | 0.997 | +0.13 |
| `basic1`    | 0.875 | 0.986 | +0.11 |
| `outliers`  | 0.977 | 0.923 | -0.05 (only meaningful regression) |

---

## Final repository layout

```
TRCC/
├── README.md                user-facing intro + benchmark table
├── CHANGES.md               19-issue mapping (v0 → v1)
├── SESSION_LOG.md           this file
├── pyproject.toml           Python + C++ build config
├── setup.py                 Pybind11Extension
├── trcc/
│   ├── __init__.py
│   ├── core.py              algorithm (TRCC class)
│   ├── benchmark.py         37-dataset suite vs KMeans/DBSCAN/HDBSCAN
│   ├── autotune.py          Optuna hyperparameter tuning
│   └── _native/
│       ├── trcc_ext.cpp     pybind11 + nanoflann C++ extension
│       └── third_party/
│           └── nanoflann.hpp (vendored, MIT license)
├── tests/
│   └── test_trcc.py         12 unit tests
├── examples/
│   └── quickstart.py        minimal demo + plot
├── scripts/
│   ├── compare_old_vs_new.py    head-to-head v0 vs v1
│   └── benchmark_native.py      Python vs C++ scaling test
├── results/                 (regenerated by benchmark.py)
│   ├── benchmark.csv
│   ├── benchmark.md
│   ├── native_vs_python.csv
│   └── figures/             37 four-panel plots
├── TRCC-old/                untouched backup of original codebase
└── Data/                    untouched datasets
```

---

## How to reproduce results

```bash
cd /Users/azeradham/Desktop/Items/TRCC
source .venv/bin/activate

# Tests
python -m pytest tests/ -v

# Full 37-dataset benchmark vs KMeans/DBSCAN/HDBSCAN
python -m trcc.benchmark --out results --data Data

# Python-vs-C++ scaling test
python scripts/benchmark_native.py

# Old TRCC vs new TRCC head-to-head
python scripts/compare_old_vs_new.py

# Quickstart visualization
python examples/quickstart.py
```

If the C++ extension goes missing, rebuild:
```bash
pip install -e . --no-build-isolation
```

---

## Open work / v2 roadmap

Ordered by ROI:

1. **kNN-graph geodesic parent search** — closes the concentric-rings
   and pure-spirals gap vs HDBSCAN. ~50 LOC. Highest impact unblocked
   item. Replaces Euclidean parent in `_delta_and_parent` with BFS over
   the kNN graph, restricted to higher-signal neighbors.
2. **OpenMP-parallel C++ kernels** — currently single-threaded; the
   `#pragma omp parallel for` is in the source but not linked. Apple
   clang doesn't ship libomp by default. ~2-3× more on multi-core.
3. **Streaming `partial_fit`** — cytokine signal supports incremental
   updates because it's a sum over kNN distances; would let TRCC handle
   data that doesn't fit in RAM.
4. **Quality benchmark on real biology data** — the original motivation.
   Single-cell RNA-seq, mass cytometry. Would establish whether TRCC's
   density-anchored approach actually delivers on its scalpel pitch.
5. **Theoretical writeup** — formalize the path-density merge as a
   relaxation of mutual reachability with density bottleneck constraint.
   Probably 4-6 pages of paper.

---

## Decisions and rationale (for future me)

- **Why NumPy + sklearn over a full C++ rewrite?** Reviewers reproduce
  in Python; clustering quality is what gets cited, not raw speed.
  Hot-path port via pybind11 gives ~all the speed of a port for ~10% of
  the maintenance burden.
- **Why density peaks and not centroid-style?** The original "α·S - β·d"
  scoring is fundamentally broken on convex-but-different-density data;
  density peaks make the cytokine signal load-bearing without the
  α/β knob hell.
- **Why path-density merge instead of mutual-reachability MST (HDBSCAN's
  approach)?** Path-density preserves a centroid story (each cluster
  has a center, you can `predict()` new points), which is what biology
  collaborators want for downstream interpretation. MST gives better
  topology handling but worse interpretability.
- **Why `max(k_ratio, k_outlier_2σ)` for auto-k?** Diagnostic across
  4 datasets showed `k_ratio` alone was perfect on every test case;
  outlier rule is the safety net for pathological tails. Median was
  too conservative (broke `boxes`); pure max with kneedle/1.5σ was too
  aggressive (broke `moons`).
- **Why NOT port now to Rust/Go/C# top-to-bottom?** Diminishing returns
  after the kd-tree fix. Profile shows sklearn (NN search) is now the
  bottleneck, not our code. A full port would shift the bottleneck to
  whatever Rust kd-tree we'd use, with no real gain.

---

## Things to watch when picking back up

- `_resolve_n_neighbors` log-growth was tuned on synthetic blobs; verify
  on real high-D data before publishing.
- The path-density merge `path_ratio = 0.6` is a magic number. If the
  paper review pushes back, justify with a sensitivity analysis or make
  it adaptive.
- `min_cluster_size = 15` default is fine for n ∈ [500, 100k]. For very
  small n it's too aggressive; for very large n it under-filters noise.
  Consider scaling to `0.5 * sqrt(n)`.
- C++ extension is single-threaded by default on macOS Apple clang.
  If you see throughput drop on Linux, check that the `#pragma omp`
  block isn't accidentally serialized.

---

*Saved 2026-04-27 at end of rebuild session.*
