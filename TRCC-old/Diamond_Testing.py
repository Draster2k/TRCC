"""
Diamond Testing Script for TRCC.
Evaluates TRCC vs KMeans on synthetic blobs generated to simulate test datasets.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from TRCC_M1 import trcc_algorithm
from sklearn.datasets import load_iris  # Replace with actual diamond dataset loader

# --- LOAD DATA ---
# Replace this block with your actual diamond dataset loading logic
# X = your_data_array
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=20000, centers=5, cluster_std=1.5, random_state=42)

# --- NORMALIZATION ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- RUN TRCC ---
print("Running TRCC...")
trcc_clusters = trcc_algorithm(
    X_scaled,
    k=100,
    max_iter=10,
    max_cluster_distance=0.8,
    min_points_per_cluster=50,
    epsilon=0.5,
    sigma=1.0
)

# --- FORMAT TRCC OUTPUT ---
trcc_labels = np.full(len(X_scaled), -1)
for cluster_id, points in trcc_clusters.items():
    for point in points:
        idx = np.where((X_scaled == point).all(axis=1))[0]
        if len(idx) > 0:
            trcc_labels[idx[0]] = cluster_id

# --- TRCC METRICS ---
valid_trcc = trcc_labels != -1
trcc_score = silhouette_score(X_scaled[valid_trcc], trcc_labels[valid_trcc])
print(f"\n[TRCC] Silhouette Score: {trcc_score:.3f} | Clusters: {len(trcc_clusters)}")

# --- RUN KMEANS FOR COMPARISON ---
print("\nRunning KMeans...")
kmeans = KMeans(n_clusters=len(trcc_clusters), n_init='auto', random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_score = silhouette_score(X_scaled, kmeans_labels)
print(f"[KMeans] Silhouette Score: {kmeans_score:.3f} | Clusters: {len(np.unique(kmeans_labels))}")

# --- VISUALIZE ---
def plot_clusters(X, labels, title):
    X_pca = PCA(n_components=2).fit_transform(X)
    cmap = plt.colormaps.get_cmap('tab10').resampled(len(np.unique(labels)))
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, s=2)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

plot_clusters(X_scaled[valid_trcc], trcc_labels[valid_trcc], f'TRCC Clustering (Score: {trcc_score:.3f})')
plot_clusters(X_scaled, kmeans_labels, f'KMeans Clustering (Score: {kmeans_score:.3f})')
plt.show()