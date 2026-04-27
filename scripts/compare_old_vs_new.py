"""
Head-to-head: TRCC v0 (old, TRCC-old/TRCC_M1.py) vs TRCC v1 (new, trcc/core.py).
Same datasets, same standardization, same metric (ARI).
"""
import sys, os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Make old code importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TRCC-old"))
from TRCC_M1 import trcc_algorithm as old_trcc  # noqa: E402

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.datasets import make_blobs, make_moons, make_circles

from trcc import TRCC


def old_to_labels(X, clusters):
    """Old API returns dict of cluster_id -> ndarray of points; recover labels."""
    labels = np.full(len(X), -1, dtype=np.int64)
    for cid, pts in clusters.items():
        if len(pts) == 0:
            continue
        pts = np.asarray(pts)
        # nearest-point lookup (same approach as the original benchmark code)
        for p in pts:
            d = np.linalg.norm(X - p, axis=1)
            i = int(np.argmin(d))
            labels[i] = cid
    return labels


def aniso():
    X, y = make_blobs(n_samples=1500, centers=3, random_state=170)
    return X @ np.array([[0.6, -0.6], [-0.4, 0.8]]), y


def varied():
    return make_blobs(n_samples=1500, cluster_std=[1.0, 2.5, 0.5], random_state=170)


synth = [
    ("blobs4", lambda: make_blobs(n_samples=1500, centers=4, cluster_std=0.7, random_state=0), 4),
    ("blobs8", lambda: make_blobs(n_samples=2000, centers=8, cluster_std=0.5, random_state=0), 8),
    ("moons", lambda: make_moons(n_samples=1500, noise=0.06, random_state=0), 2),
    ("moons_noisy", lambda: make_moons(n_samples=1500, noise=0.10, random_state=0), 2),
    ("circles", lambda: make_circles(n_samples=1500, factor=0.5, noise=0.05, random_state=0), 2),
    ("aniso", aniso, 3),
    ("varied", varied, 3),
]


def csv_panel(data_dir):
    out = []
    for path in sorted(list(data_dir.glob("**/*.csv"))):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        y = None
        for c in ("label", "class", "color", "Label", "Class", "Color"):
            if c in df.columns:
                y = df[c].to_numpy()
                df = df.drop(columns=[c])
                break
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            continue
        mask = num.notna().all(axis=1).to_numpy()
        if y is not None:
            y = np.asarray(y)[mask]
        X = num.to_numpy()[mask][:, :2]
        if len(X) < 50:
            continue
        if y is not None:
            try:
                y = pd.factorize(pd.Series(y).astype(str).fillna("__nan__"))[0]
                if len(np.unique(y)) < 2:
                    y = None
            except Exception:
                y = None
        out.append((path.stem, X, y))
    return out


rows = []

panel = []
for name, gen, k_true in synth:
    X, y = gen()
    X = StandardScaler().fit_transform(X)
    panel.append((name, X, y, k_true))

data_dir = Path("Data")
if data_dir.exists():
    for name, X, y in csv_panel(data_dir):
        X = StandardScaler().fit_transform(X)
        if y is not None and len(np.unique(y)) >= 2:
            panel.append((name, X, y, len(np.unique(y))))

print(f"{'dataset':24s} {'old_ARI':>8s} {'new_ARI':>8s} {'Δ':>7s} {'old_t':>7s} {'new_t':>7s}")
print("-" * 70)

for name, X, y, k_true in panel:
    if y is None:
        continue
    # old TRCC
    t0 = time.time()
    try:
        clusters = old_trcc(X, k=100, max_iter=10,
                            max_cluster_distance=0.7, min_points_per_cluster=10,
                            epsilon=0.4, sigma=0.6, alpha=1.2, beta=1.0)
        old_labels = old_to_labels(X, clusters)
        old_ari = adjusted_rand_score(y, old_labels)
    except Exception as e:
        old_ari = float("nan")
    old_t = time.time() - t0

    # new TRCC
    t0 = time.time()
    try:
        new_labels = TRCC().fit_predict(X)
        new_ari = adjusted_rand_score(y, new_labels)
    except Exception:
        new_ari = float("nan")
    new_t = time.time() - t0

    delta = new_ari - old_ari
    rows.append({"dataset": name, "old_ari": old_ari, "new_ari": new_ari,
                 "delta": delta, "old_t": old_t, "new_t": new_t})
    print(f"{name:24s} {old_ari:8.3f} {new_ari:8.3f} {delta:+7.3f} {old_t:7.2f} {new_t:7.2f}")

df = pd.DataFrame(rows)
print("-" * 70)
print(f"{'mean':24s} {df.old_ari.mean():8.3f} {df.new_ari.mean():8.3f} "
      f"{(df.new_ari - df.old_ari).mean():+7.3f} "
      f"{df.old_t.mean():7.2f} {df.new_t.mean():7.2f}")
print()
n = len(df)
new_wins = int((df.new_ari > df.old_ari + 0.01).sum())
old_wins = int((df.old_ari > df.new_ari + 0.01).sum())
ties = n - new_wins - old_wins
print(f"new wins: {new_wins}/{n}   old wins: {old_wins}/{n}   ties: {ties}/{n}")

df.round(3).to_csv("results/old_vs_new.csv", index=False)
print("wrote results/old_vs_new.csv")
