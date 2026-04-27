"""
Benchmark TRCC with the C++ extension vs the pure-Python fallback.
Reports per-dataset runtime, speedup, and verifies ARI is unchanged.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from trcc import core as trcc_core
from trcc import TRCC


def bench(X, y=None):
    # native
    trcc_core._HAS_NATIVE = True
    t0 = time.perf_counter()
    labels_native = TRCC().fit_predict(X)
    t_native = time.perf_counter() - t0

    # python fallback
    trcc_core._HAS_NATIVE = False
    t0 = time.perf_counter()
    labels_python = TRCC().fit_predict(X)
    t_python = time.perf_counter() - t0
    trcc_core._HAS_NATIVE = True

    ari_native = adjusted_rand_score(y, labels_native) if y is not None else float("nan")
    ari_python = adjusted_rand_score(y, labels_python) if y is not None else float("nan")
    return {
        "n": len(X),
        "t_python": t_python,
        "t_native": t_native,
        "speedup": t_python / t_native if t_native > 0 else float("inf"),
        "ari_python": ari_python,
        "ari_native": ari_native,
        "labels_match": float(np.mean(labels_python == labels_native)),
    }


def aniso():
    X, y = make_blobs(n_samples=1500, centers=3, random_state=170)
    return X @ np.array([[0.6, -0.6], [-0.4, 0.8]]), y


cases = [
    ("blobs4_1.5k", lambda: make_blobs(n_samples=1500, centers=4, cluster_std=0.7, random_state=0)),
    ("blobs8_2k",   lambda: make_blobs(n_samples=2000, centers=8, cluster_std=0.5, random_state=0)),
    ("moons_1.5k",  lambda: make_moons(n_samples=1500, noise=0.06, random_state=0)),
    ("aniso_1.5k",  aniso),
    ("blobs_5k",    lambda: make_blobs(n_samples=5000, centers=6, cluster_std=0.6, random_state=0)),
    ("blobs_10k",   lambda: make_blobs(n_samples=10000, centers=8, cluster_std=0.5, random_state=0)),
    ("blobs_20k",   lambda: make_blobs(n_samples=20000, centers=10, cluster_std=0.5, random_state=0)),
    ("blobs_50k",   lambda: make_blobs(n_samples=50000, centers=10, cluster_std=0.5, random_state=0)),
]

rows = []
print(f"{'dataset':14s} {'n':>7s} {'python_s':>10s} {'native_s':>10s} {'speedup':>8s} "
      f"{'ARI_py':>8s} {'ARI_C':>8s} {'agree':>7s}")
print("-" * 80)
for name, gen in cases:
    X, y = gen()
    X = StandardScaler().fit_transform(X)
    r = bench(X, y)
    rows.append({"dataset": name, **r})
    print(f"{name:14s} {r['n']:>7d} {r['t_python']:>10.3f} {r['t_native']:>10.3f} "
          f"{r['speedup']:>7.1f}x {r['ari_python']:>8.3f} {r['ari_native']:>8.3f} "
          f"{r['labels_match']:>6.1%}")

print("-" * 80)
df = pd.DataFrame(rows)
print(f"{'GEOMEAN':14s} {'':>7s} {'':>10s} {'':>10s} "
      f"{np.exp(np.log(df.speedup).mean()):>7.1f}x")
df.round(4).to_csv("results/native_vs_python.csv", index=False)
print(f"\nwrote results/native_vs_python.csv")
