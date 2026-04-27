"""
Reproducible benchmark suite for TRCC vs KMeans, DBSCAN, HDBSCAN.

Datasets: a panel of synthetic shapes (sklearn.datasets) plus all CSVs found
in `Data/` (each treated as a 2D point cloud). Metrics: ARI, NMI (where
ground truth exists), silhouette (always), runtime, cluster count.

Usage
-----
    python -m trcc.benchmark --out results/

Outputs
-------
    results/benchmark.csv      table of all runs
    results/benchmark.md       markdown summary (mean ARI per algorithm)
    results/figures/<name>.png 2D scatter for each dataset (multi-panel)
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

import hdbscan

from .core import TRCC


# ---------------- synthetic dataset panel ----------------

def synthetic_panel(seed: int = 0):
    def aniso():
        X, y = make_blobs(n_samples=1500, centers=3, random_state=170)
        X = X @ np.array([[0.6, -0.6], [-0.4, 0.8]])
        return X, y

    def varied():
        return make_blobs(
            n_samples=1500, cluster_std=[1.0, 2.5, 0.5], random_state=170
        )

    return [
        ("blobs4", lambda: make_blobs(n_samples=1500, centers=4,
                                      cluster_std=0.7, random_state=seed), 4),
        ("blobs8", lambda: make_blobs(n_samples=2000, centers=8,
                                      cluster_std=0.5, random_state=seed), 8),
        ("moons", lambda: make_moons(n_samples=1500, noise=0.06,
                                     random_state=seed), 2),
        ("moons_noisy", lambda: make_moons(n_samples=1500, noise=0.10,
                                           random_state=seed), 2),
        ("circles", lambda: make_circles(n_samples=1500, factor=0.5,
                                         noise=0.05, random_state=seed), 2),
        ("aniso", aniso, 3),
        ("varied", varied, 3),
    ]


# ---------------- CSV dataset loader ----------------

def csv_panel(data_dir: Path):
    """Load every *.csv as (name, X, y_or_None). y comes from a 'label'
    column if present, otherwise None."""
    out = []
    if not data_dir.exists():
        return out
    paths = list(data_dir.glob("*.csv")) + list(data_dir.glob("**/*.csv"))
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]
    for path in sorted(paths):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.shape[1] < 2:
            continue
        # use first 2 numeric columns; treat 'label'/'class'/'y' as ground truth
        y = None
        for label_col in ("label", "class", "color", "Class", "Label", "Color", "target", "Target"):
            if label_col in df.columns:
                y = df[label_col].to_numpy()
                df = df.drop(columns=[label_col])
                break
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            continue
        # drop rows with NaN to satisfy sklearn estimators
        clean_mask = num_df.notna().all(axis=1).to_numpy()
        if y is not None:
            y = np.asarray(y)[clean_mask]
        num = num_df.to_numpy()[clean_mask]
        if len(num) < 50:
            continue
        X = num[:, :2]
        out.append((path.stem, X, y))
    return out


# ---------------- algorithm adapters ----------------

def run_trcc(X):
    return TRCC().fit_predict(X)

def run_kmeans(X, k):
    k = int(np.clip(k, 2, 50))
    return KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(X)

def run_dbscan(X):
    return DBSCAN(eps=0.2, min_samples=10).fit_predict(X)

def run_hdbscan(X):
    return hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(X)


# ---------------- metric helpers ----------------

def metrics(X, labels, y_true=None):
    valid = labels != -1
    n_clusters = int(len(np.unique(labels[valid]))) if valid.any() else 0
    n_noise = int((~valid).sum())
    n_valid = int(valid.sum())
    # silhouette is undefined when n_clusters == n_valid (degenerate
    # singleton clusters) or fewer than 2 clusters
    if 2 <= n_clusters < n_valid and n_valid >= 10:
        sil = float(silhouette_score(X[valid], labels[valid]))
    else:
        sil = float("nan")
    if y_true is not None:
        ari = float(adjusted_rand_score(y_true, labels))
        nmi = float(normalized_mutual_info_score(y_true, labels))
    else:
        ari = float("nan")
        nmi = float("nan")
    return n_clusters, n_noise, ari, nmi, sil


# ---------------- runner ----------------

def benchmark(out_dir: Path, data_dir: Optional[Path]):
    rows = []
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel = []
    for name, gen, k_true in synthetic_panel():
        X, y = gen()
        X = StandardScaler().fit_transform(X)
        panel.append((name, X, y, k_true))

    if data_dir is not None:
        for name, X, y in csv_panel(data_dir):
            X = StandardScaler().fit_transform(X)
            if y is not None:
                try:
                    series = pd.Series(y)
                    y_arr = pd.factorize(series.astype(str).fillna("__nan__"))[0]
                    valid_y = y_arr[y_arr >= 0]
                    n_unique = len(np.unique(valid_y)) if len(valid_y) else 0
                    if n_unique < 2:
                        # single-class column is not real ground truth
                        y = None
                        k_true = 4
                    else:
                        k_true = min(n_unique, 50)
                        y = y_arr
                except Exception:
                    y = None
                    k_true = 4
            else:
                k_true = 4
            panel.append((name, X, y, k_true))

    algos = [
        ("TRCC", lambda X, k: run_trcc(X)),
        ("KMeans", lambda X, k: run_kmeans(X, k)),
        ("DBSCAN", lambda X, k: run_dbscan(X)),
        ("HDBSCAN", lambda X, k: run_hdbscan(X)),
    ]

    for name, X, y, k_true in panel:
        per_dataset_labels = {}
        for algo_name, fn in algos:
            t0 = time.time()
            try:
                labels = fn(X, k_true)
            except Exception as e:
                print(f"[{name}/{algo_name}] failed: {e}")
                continue
            elapsed = time.time() - t0
            n_c, n_noise, ari, nmi, sil = metrics(X, labels, y)
            rows.append(
                dict(
                    dataset=name,
                    algorithm=algo_name,
                    n_clusters=n_c,
                    n_noise=n_noise,
                    ari=ari,
                    nmi=nmi,
                    silhouette=sil,
                    runtime_sec=round(elapsed, 4),
                )
            )
            per_dataset_labels[algo_name] = labels
            print(f"{name:18s} {algo_name:8s} k={n_c:2d} noise={n_noise:5d} "
                  f"ARI={ari:.3f} NMI={nmi:.3f} sil={sil:.3f} t={elapsed:.2f}s")
        # plot 4-panel comparison if 2D
        if X.shape[1] == 2 and len(per_dataset_labels) > 0:
            _plot_panel(X, per_dataset_labels, name, figures_dir / f"{name}.png")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "benchmark.csv", index=False)
    _write_markdown(df, out_dir / "benchmark.md")
    print(f"\nWrote {out_dir/'benchmark.csv'} and {out_dir/'benchmark.md'}")


def _plot_panel(X, labels_dict, title, out_path):
    n = len(labels_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (algo, labels) in zip(axes, labels_dict.items()):
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=4, cmap="tab20", alpha=0.7)
        ax.set_title(f"{algo}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_markdown(df: pd.DataFrame, path: Path):
    if df.empty:
        path.write_text("No results.\n")
        return
    pivot = df.pivot_table(index="dataset", columns="algorithm",
                           values="ari", aggfunc="first")
    sil_pivot = df.pivot_table(index="dataset", columns="algorithm",
                               values="silhouette", aggfunc="first")
    rt_pivot = df.pivot_table(index="dataset", columns="algorithm",
                              values="runtime_sec", aggfunc="first")
    md = ["# TRCC Benchmark Summary", "", "## Adjusted Rand Index (ARI)",
          "Higher is better. NaN = no ground truth.", "",
          pivot.round(3).to_markdown(), "",
          f"**Mean ARI per algorithm** (excluding NaNs):", ""]
    for c in pivot.columns:
        md.append(f"- {c}: {pivot[c].mean(skipna=True):.3f}")
    md += ["", "## Silhouette", "", sil_pivot.round(3).to_markdown(),
           "", "## Runtime (seconds)", "", rt_pivot.round(3).to_markdown(), ""]
    path.write_text("\n".join(md))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results", type=Path)
    p.add_argument("--data", default="Data", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    benchmark(args.out, args.data if args.data.exists() else None)


if __name__ == "__main__":
    main()
