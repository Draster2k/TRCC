"""Minimal example: fit TRCC on a synthetic dataset and visualize."""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from trcc import TRCC

X, y = make_blobs(n_samples=2000, centers=5, cluster_std=0.6, random_state=0)
X = StandardScaler().fit_transform(X)

model = TRCC().fit(X)
print(f"n_clusters = {model.n_clusters_}")
print(f"n_noise    = {(model.labels_ == -1).sum()}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X[:, 0], X[:, 1], c=model.labels_, s=8, cmap="tab10", alpha=0.7)
peaks = model.peak_indices_
ax.scatter(X[peaks, 0], X[peaks, 1], c="black", marker="x", s=100, label="density peaks")
ax.set_title(f"TRCC: {model.n_clusters_} clusters")
ax.legend()
fig.tight_layout()
fig.savefig("examples/quickstart.png", dpi=120)
print("wrote examples/quickstart.png")
