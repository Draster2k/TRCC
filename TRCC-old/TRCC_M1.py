"""
TRCC (T-Regulated Cytokine Clustering) Algorithm (Numpy Implementation)
This module provides a CPU-optimized implementation of the TRCC algorithm.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import time


def trcc_algorithm(X, k=100, max_iter=10, max_cluster_distance=0.8,
                   min_points_per_cluster=50, epsilon=0.5, sigma=1.0,
                   alpha=1.0, beta=1.0):
    """
    Executes the TRCC algorithm to cluster data points.
    
    Args:
        X: Input data array.
        k: Initial number of random seeds for clustering.
        max_iter: Maximum number of assignment iterations.
        max_cluster_distance: Distance threshold under which to merge clusters.
        min_points_per_cluster: Minimum bounds for a valid cluster.
        epsilon: Radius for nearest neighbors cytokine signal calculation.
        sigma: Spread of the Gaussian used for cytokine calculation.
        alpha: Weight of cytokine signal in the assignment score.
        beta: Weight of euclidean distance in the assignment score.
        
    Returns:
        dict: A dictionary mapping cluster IDs to their constituent data points.
    """
    def initialize_clusters(X, k):
        """Randomly initialize k cluster seed points."""
        np.random.seed(42)
        k = min(k, len(X))
        indices = np.random.choice(len(X), k, replace=False)
        return {i: [X[idx]] for i, idx in enumerate(indices)}

    def calculate_cytokine_signals(X, epsilon, sigma):
        """Calculate density-based 'cytokine' signals for all points within epsilon radius."""
        nbrs = NearestNeighbors(radius=epsilon, algorithm='auto', n_jobs=-1).fit(X)
        distances, indices = nbrs.radius_neighbors(X)

        signals = np.zeros(len(X))
        for i, (dist_list, idx_list) in enumerate(zip(distances, indices)):
            if len(dist_list) > 0:
                signals[i] = np.sum(np.exp(-np.square(dist_list) / (2 * sigma ** 2)))
        return signals

    def assign_points_to_clusters(X, signals, clusters):
        """Assign data points to clusters based on a weighted sum of distance and local density (signal)."""
        cluster_centers = np.array([np.mean(points, axis=0) for points in clusters.values()])
        assignments = [[] for _ in range(len(cluster_centers))]

        for i, x in enumerate(X):
            signal = signals[i]
            scores = -np.linalg.norm(cluster_centers - x, axis=1) * beta + signal * alpha
            best_cluster = np.argmax(scores)
            assignments[best_cluster].append(x)

        return {i: np.array(points) for i, points in enumerate(assignments)}

    def merge_clusters(clusters, max_distance):
        """Merge nearby clusters whose centroids are closer than max_distance."""
        centroids = np.array([np.mean(points, axis=0) for points in clusters.values()])
        merged = set()
        new_clusters = {}

        cluster_keys = list(clusters.keys())
        cluster_count = 0

        for i, key_i in enumerate(cluster_keys):
            if key_i in merged:
                continue
            current_points = list(clusters[key_i])
            for j, key_j in enumerate(cluster_keys):
                if i != j and key_j not in merged:
                    dist = np.linalg.norm(np.mean(current_points, axis=0) - np.mean(clusters[key_j], axis=0))
                    if dist < max_distance:
                        current_points.extend(clusters[key_j])
                        merged.add(key_j)
            new_clusters[cluster_count] = np.array(current_points)
            cluster_count += 1

        print(f"After merging: {len(new_clusters)} clusters remain")
        return new_clusters

    def filter_small_clusters(clusters, min_points):
        """Filter out clusters that do not satisfy the minimum points requirement."""
        filtered_clusters = {i: points for i, points in clusters.items() if len(points) >= min_points}
        print(f"After filtering: {len(filtered_clusters)} clusters remain")
        return filtered_clusters

    def clusters_equal(c1, c2):
        if len(c1) != len(c2):
            return False
        for key in c1:
            if key not in c2 or len(c1[key]) != len(c2[key]):
                return False
        return True

    # --- Main TRCC Loop ---
    clusters = initialize_clusters(X, k)
    prev_clusters = None

    for iteration in range(max_iter):
        start = time.time()
        print(f"Iteration {iteration + 1}")

        signals = calculate_cytokine_signals(X, epsilon, sigma)
        clusters = assign_points_to_clusters(X, signals, clusters)
        print(f"Number of clusters after assignment: {len(clusters)}")

        clusters = merge_clusters(clusters, max_cluster_distance)
        clusters = filter_small_clusters(clusters, min_points_per_cluster)

        if prev_clusters is not None and clusters_equal(prev_clusters, clusters):
            print("Convergence reached.")
            break

        prev_clusters = clusters.copy()
        print(f"Iteration time: {round(time.time() - start, 2)} seconds\n")

    return clusters