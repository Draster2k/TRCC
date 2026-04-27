# TRCC — T-Regulated Cytokine Clustering (v1.1)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19832226.svg)](https://doi.org/10.5281/zenodo.19832226)
[![Paper](https://img.shields.io/badge/Paper-10.5281%2Fzenodo.19831715-blue)](https://doi.org/10.5281/zenodo.19831715)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--9555--1036-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0005-9555-1036)

A density-anchored clustering algorithm. Each point emits a kNN-Gaussian
"cytokine" density signal; cluster centers are discovered as **density
peaks**; non-peak points propagate their label down the density gradient;
and clusters fuse only when a high-density **ridge** connects them
(path-density mutual reachability). The hot path is implemented in
**C++ with a [nanoflann](https://github.com/jlblancoc/nanoflann) kd-tree**
exposed via pybind11 — a pure-Python fallback ships in the same package.

```python
from trcc import TRCC
labels = TRCC().fit_predict(X)        # -1 = noise, 0..K-1 = clusters
```

> **Status.** v1.1 is the version targeted at the technical paper.
> 100,000 points clustered in **3.45 s / 533 MB**, mean ARI **0.65** across
> 37 benchmark datasets. **+61 % accuracy and ≈10× faster than the
> original v0**, second only to HDBSCAN (mean ARI 0.85) — and stronger
> than HDBSCAN on a class of datasets (anisotropic, varied-density,
> outlier-laden) where its USP shines.

---

## 1.  Progression: v0 → v1.1

| Stage | Mean ARI | Wins (37) | Mean runtime | Speedup vs v0 |
|---|---:|---:|---:|---:|
| **v0** original (`TRCC-old/`) | 0.404 | ~3 | 0.92 s | 1× |
| **v1.0** algorithm rewrite | 0.587 | 9 | 0.27 s | 3.4× |
| **v1.0 + C++ extension** | 0.587 | 9 | 0.27 s | bit-identical labels, 4.6× geomean compute speedup |
| **v1.1** auto-K + `√n` neighborhood | **0.639** | **12** | **0.097 s** | **9.5×** |

Each step is reproducible from the repository:

```bash
pip install -e . --no-build-isolation
python -m pytest tests/ -q                    # 12/12
python -m trcc.benchmark --out results --data Data
python scripts/compare_old_vs_new.py          # head-to-head v0 vs v1.1
python scripts/benchmark_native.py            # Python fallback vs C++
python scripts/sensitivity_n_neighbors.py     # ablation table for the paper
```

---

## 2.  Why TRCC: density-peak + path-density vs. HDBSCAN's MST

HDBSCAN and TRCC are **both** density-aware, but they answer two
different questions about the data:

|  | **HDBSCAN** | **TRCC v1.1** |
|---|---|---|
| **Structural primitive** | Mutual-reachability minimum spanning tree | Cytokine signal `S_i` + density-peak parent forest |
| **Cluster identity** | A connected sub-tree of the condensed hierarchy | A density mode (peak) + everything that flows to it |
| **Cluster representation** | Set of points; no distinguished center | Density peak + centroid → enables `predict()` for new points |
| **Cluster fusion criterion** | Edge weight in the MST (point-to-point reachability) | Path-density ridge between **centroids** (cluster-to-cluster) |
| **Best at** | Topologically defined clusters of arbitrary shape (rings, spirals, manifold-like) | Centrally-massed clusters with anisotropy, varying density, or embedded outliers |
| **Inherent failure mode** | Variable-density clusters that share a density floor | Clusters that share a single density mode (concentric rings) |

The trade-off is principled: **HDBSCAN trades interpretability for
topology robustness**; **TRCC trades topology robustness for
density-anchored interpretability**. On a Gaussian-mixture-style dataset,
TRCC produces a centroid-and-peak summary that downstream pipelines
(e.g. cell-type assignment in scRNA-seq) can directly consume. On
concentric rings or pure spirals, HDBSCAN is the right tool; TRCC will
collapse the rings (a known density-peak limitation).

### The algorithmic novelty: **path-density mutual reachability**

Plain density-peak clustering (Rodriguez & Laio, *Science* 2014) tends
to over-fragment when the peak detector picks slightly more peaks than
exist. HDBSCAN solves over-fragmentation via tree condensation. **TRCC
solves it by checking whether two clusters are joined by a high-density
ridge in the data**, not just whether their centroids are close:

```
score for merging clusters A and B  =
        min     S(midpoint of segment A–B)
   midpoints
   ─────────────────────────────────────────
        min(peak signal of A, peak signal of B)

merge if score ≥ 0.6
```

This is a *cluster-to-cluster* analogue of HDBSCAN's mutual
reachability, evaluated on the original cytokine signal rather than on a
graph. It enforces (i) spatial closeness of centroids and (ii) the
existence of a continuous high-density path connecting them — a stricter
criterion than centroid distance alone, and a more permissive one than
single-linkage on the MST. Formal definition in
[`paper/formalization.tex`](paper/formalization.tex), Eqs. (8)–(11).

---

## 3.  Headline benchmark (37 datasets)

```
algorithm   mean ARI   mean silhouette   wins   mean runtime
TRCC v1.1     0.639           0.42        12       0.097 s
HDBSCAN       0.849           0.43        14       0.039 s
DBSCAN        0.578           0.41         7       0.017 s
KMeans        0.572           0.49         4       0.011 s
```

TRCC wins the most on:

| dataset | TRCC | KMeans | DBSCAN | HDBSCAN |
|---|---:|---:|---:|---:|
| `aniso` (sheared blobs)     | **0.998** | 0.610 | 0.975 | 0.932 |
| `triangle`                  | **0.971** | 0.257 | 0.601 | 0.726 |
| `supernova`                 | **0.988** | 0.962 | 0.000 | 0.711 |
| `varied` (different σ)      | **0.934** | 0.809 | 0.877 | 0.844 |
| `boxes`                     | **1.000** | 0.999 | 0.000 | 0.999 |
| `lines2`                    | 0.999 | 0.440 | 0.000 | **0.981** |
| `outliers`                  | 0.923 | 0.643 | 0.907 | **0.997** |

Acknowledged TRCC failure cases: `circles`, `spirals`, `dart`/`dart2`,
`un` — concentric-ring-style topologies on which density peaks cannot
distinguish rings of equal density. Future v2 work (kNN-graph geodesic
parent search) addresses this.

Full table and figures: [`results/benchmark.md`](results/benchmark.md),
[`results/figures/`](results/figures/).

---

## 4.  Performance: C++ extension (pybind11 + nanoflann)

The hot path (`delta_and_parent` + `propagate_labels`) is in C++.
Labels match the Python fallback bit-for-bit.

| n | python | native | speedup |
|---:|---:|---:|---:|
| 1,500 | 35 ms | 23 ms | 1.5× |
| 5,000 | 273 ms | 49 ms | 5.6× |
| 10,000 | 906 ms | 106 ms | 8.5× |
| 20,000 | 3.66 s | 228 ms | **16.1×** |
| 50,000 | 44.4 s | 0.93 s | **47.8×** |

**Geometric mean: 4.6×.** The win is both algorithmic (kd-tree
expanding-radius search → O(n·k·log n) instead of O(n²)) and constant
factor (C++ inner loops). At 100 k points: **3.45 s, 533 MB**.

---

## 5.  Install

```bash
pip install trcc                           # builds the C++ extension
```

Or from source:

```bash
git clone https://github.com/draster2k/trcc.git
cd trcc
pip install -e . --no-build-isolation
python -m pytest tests/                    # 12/12 should pass
```

### No C++ compiler? No problem.

**TRCC is fully functional without a C++ toolchain.** If the build fails
or you want to skip it, set the environment variable before install:

```bash
TRCC_NO_EXTENSION=1 pip install trcc
```

The package then runs entirely in NumPy + scikit-learn. The output is
**bit-identical** to the C++ path (verified across the 37-dataset
benchmark and all unit tests) — only the runtime differs. Expect a
~5–50× slowdown on the kd-tree-bound inner loop for `n ≥ 10⁴`; for
small datasets (`n ≤ 2,000`) the difference is negligible.

```python
# Same code works whether or not the extension was built:
from trcc import TRCC
labels = TRCC().fit_predict(X)
```

### Optional: OpenMP parallelism

To enable OpenMP parallelism in the kd-tree query, set
`TRCC_BUILD_OMP=1` before install. On Linux / GCC this works out of the
box; on macOS Apple clang requires `brew install libomp` first.

### Tested on

- macOS 14 / Apple clang 21
- Linux / gcc 11+
- Python 3.9, 3.10, 3.11, 3.12

---

## 6.  Hyperparameters

All defaults are derived from the data — no manual scale-dependent
tuning required. Override any via the constructor.

| Param | Default | Formula / role |
|---|---|---|
| `n_clusters` | `"auto"` | `max(k_ratio, k_outlier_2σ)` over sorted γ = S·δ |
| `n_neighbors` | `"auto"` | `clip(⌈√n / 2⌉, 10, 200)` (Loftsgaarden–Quesenberry 1965) |
| `sigma` | `"auto"` | `median(k-th NN distance)` |
| `min_cluster_size` | 15 | Smaller clusters → noise (-1) |
| `min_signal_percentile` | 0 | Set >0 to label low-density points as noise |
| `merge_threshold` | `"auto"` | Centroid-distance ceiling for merge candidates |
| `max_clusters` | 50 | Upper bound when `n_clusters="auto"` |
| `random_state` | 42 | Tie-breaking only; main pipeline is deterministic |

Run [`scripts/sensitivity_n_neighbors.py`](scripts/sensitivity_n_neighbors.py)
to verify low sensitivity to `n_neighbors` — output is in
[`paper/sensitivity_table.tex`](paper/sensitivity_table.tex).

---

## 7.  API

```python
from trcc import TRCC

model = TRCC().fit(X)
model.labels_              # (n,)            cluster IDs, -1 = noise
model.cluster_centers_     # (K, d)          centroids
model.signals_             # (n,)            cytokine signal per point
model.peak_indices_        # (K,)            indices of density peaks in X
model.cluster_signal_      # (K,)            mean signal per cluster
model.n_clusters_          # int             K after auto-K + merging
model.predict(X_new)       # (m,)            assign new points to nearest centroid
```

Tuning helper:

```python
from trcc.autotune import tune
res = tune(X, y_true=None, n_trials=40)    # silhouette objective
print(res.params, res.score)
```

---

## 8.  Citing

If you use TRCC in academic work, please cite the archived release:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19832226.svg)](https://doi.org/10.5281/zenodo.19832226)

```bibtex
@software{adham_trcc_2026,
  author       = {Adham, Azar},
  title        = {TRCC v1.1.0: T-Regulated Cytokine Clustering --
                  source code, benchmark suite, and reproducibility scripts},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.1.0},
  doi          = {10.5281/zenodo.19832226},
  url          = {https://doi.org/10.5281/zenodo.19832226},
  orcid        = {https://orcid.org/0009-0005-9555-1036},
  note         = {Density-peak clustering with path-density mutual-reachability merging.}
}
```

The companion paper preprint is archived separately:

```bibtex
@misc{adham_trcc_paper_2026,
  author    = {Adham, Azar},
  title     = {TRCC: T-Regulated Cytokine Clustering with Path-Density Mutual-Reachability Merging},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19831715},
  url       = {https://doi.org/10.5281/zenodo.19831715}
}
```

GitHub also offers a one-click "Cite this repository" button (powered by
the [`CITATION.cff`](CITATION.cff) at the repo root) that exports both
records to BibTeX, RIS, EndNote, and other formats automatically.

The algorithm builds on the density-peak idea of Rodriguez & Laio
(*Science* 2014), the kNN density estimator of Loftsgaarden &
Quesenberry (1965), and the mutual-reachability concept used by
HDBSCAN (Campello et al. 2013).

---

## 9.  Repository

```
trcc/                    package
  ├── core.py            algorithm
  ├── benchmark.py       benchmark suite
  ├── autotune.py        Optuna tuning
  └── _native/           C++ extension (nanoflann + pybind11)
tests/                   12 unit tests
examples/                quickstart + visual
scripts/                 ablations (Python vs C++, v0 vs v1, sensitivity)
results/                 generated outputs (CSV + markdown + PNGs)
paper/                   formalization.tex, pseudocode.tex, sensitivity_table.tex
TRCC-old/                untouched original
```

Documentation:

- [`README.md`](README.md) — this file
- [`CHANGES.md`](CHANGES.md) — 19-issue mapping (v0 → v1)
- [`SESSION_LOG.md`](SESSION_LOG.md) — full rebuild log + decisions
- [`paper/formalization.tex`](paper/formalization.tex) — formal math
- [`paper/pseudocode.tex`](paper/pseudocode.tex) — algorithms
- [`paper/sensitivity_table.tex`](paper/sensitivity_table.tex) — ablation

---

*MIT License. Built with NumPy, scikit-learn, pybind11, and nanoflann.*
