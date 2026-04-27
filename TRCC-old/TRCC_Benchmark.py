"""
TRCC Benchmarking Script.
Generates synthetic datasets (Moons, Circles, Classification) and compares TRCC
performance against KMeans and DBSCAN using the Silhouette Score.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from TRCC_M1 import trcc_algorithm

def evaluate_and_plot(X, dataset_name, trcc_params):
    """
    Evaluates TRCC, KMeans, and DBSCAN on dataset X and plots the results.
    
    Args:
        X (ndarray): The input dataset.
        dataset_name (str): The name of the dataset for labeling plots.
        trcc_params (dict): Hyperparameters for the TRCC algorithm.
    """
    X = StandardScaler().fit_transform(X)

    # --- TRCC ---
    trcc_clusters = trcc_algorithm(
        X,
        k=100,
        max_iter=10,
        max_cluster_distance=trcc_params["max_cluster_distance"],
        min_points_per_cluster=10,
        epsilon=trcc_params["epsilon"],
        sigma=trcc_params["sigma"],
        alpha=trcc_params["alpha"],
        beta=trcc_params["beta"]
    )

    trcc_labels = np.full(len(X), -1)
    for cluster_id, points in trcc_clusters.items():
        for p in points:
            idx = np.where((X == p).all(axis=1))[0]
            if len(idx) > 0:
                trcc_labels[idx[0]] = cluster_id

    valid_trcc = trcc_labels != -1
    trcc_score = silhouette_score(X[valid_trcc], trcc_labels[valid_trcc])

    # --- KMeans ---
    kmeans = KMeans(n_clusters=len(np.unique(trcc_labels[valid_trcc])), n_init='auto', random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_score = silhouette_score(X, kmeans_labels)

    # --- DBSCAN ---
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    db_labels = dbscan.fit_predict(X)
    valid_db = db_labels != -1
    db_score = silhouette_score(X[valid_db], db_labels[valid_db]) if len(set(db_labels[valid_db])) > 1 else -1

    # --- Plot ---
    def plot_clusters(title, X, labels, score):
        plt.figure()
        plt.title(f"{title} (score={score:.3f})")
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=5, cmap='tab10')
        plt.grid(True)
        plt.tight_layout()

    plot_clusters(f"TRCC - {dataset_name}", X[valid_trcc], trcc_labels[valid_trcc], trcc_score)
    plot_clusters(f"KMeans - {dataset_name}", X, kmeans_labels, kmeans_score)
    plot_clusters(f"DBSCAN - {dataset_name}", X[valid_db], db_labels[valid_db], db_score)

    print(f"{dataset_name} | TRCC: {trcc_score:.3f}, KMeans: {kmeans_score:.3f}, DBSCAN: {db_score:.3f}")

if __name__ == "__main__":
    # Use best params from Optuna or set manually
    trcc_best_params = {
        "epsilon": 0.4,
        "sigma": 0.6,
        "max_cluster_distance": 0.7,
        "alpha": 1.2,
        "beta": 1.0
    }

    datasets = {
        "Moons": make_moons(n_samples=2000, noise=0.07, random_state=42),
        "Circles": make_circles(n_samples=2000, factor=0.5, noise=0.05, random_state=42),
        "Classification": make_classification(n_samples=2000, n_features=2, n_redundant=0,
                                              n_clusters_per_class=1, n_classes=3, random_state=42)
    }

    for name, (X, _) in datasets.items():
        evaluate_and_plot(X, name, trcc_best_params)

    plt.show()