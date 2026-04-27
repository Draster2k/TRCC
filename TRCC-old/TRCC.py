"""
PyTorch-based Object-Oriented TRCC (T-Regulated Cytokine Clustering) Algorithm Implementation.
This module provides a class-based structure for TRCC, utilizing PyTorch tensors.
"""

import torch
from sklearn.neighbors import NearestNeighbors


class TRCC:
    def __init__(self, k=20, max_iter=15, max_cluster_distance=1.5,
                 min_points_per_cluster=10, epsilon=0.5, sigma=1.0,
                 alpha=1.0, beta=1.0, device="cpu"):
        """
        Manual-tuning TRCC clustering algorithm.

        Parameters
        ----------
        k : int
            Initial number of clusters (like KMeans n_init).
        max_iter : int
            Maximum iterations for assignment.
        max_cluster_distance : float
            Merge clusters if centroid distance < this value.
        min_points_per_cluster : int
            Remove clusters smaller than this size.
        epsilon : float
            Radius for cytokine signal neighbor search.
        sigma : float
            Spread for Gaussian kernel in cytokine signals.
        alpha : float
            Weight for signal strength.
        beta : float
            Weight for distance.
        device : str
            "cpu" or "cuda" (GPU). Stick to "cpu" on M1/Intel for now.
        """
        self.k = k
        self.max_iter = max_iter
        self.max_cluster_distance = max_cluster_distance
        self.min_points_per_cluster = min_points_per_cluster
        self.epsilon = epsilon
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.labels_ = None

    # --- Helpers ---
    def _initialize_clusters(self, X):
        """Randomly pick k points from X as initial cluster seeds."""
        torch.manual_seed(42)
        k = min(self.k, len(X))
        indices = torch.randperm(len(X), device=self.device)[:k]
        return {i: X[idx:idx+1] for i, idx in enumerate(indices)}

    def _calculate_signals(self, X):
        """
        Calculates density-based signals for all points using sklearn NearestNeighbors
        on CPU and applying a Gaussian kernel to the distances.
        """
        # Use sklearn NearestNeighbors on CPU
        X_cpu = X.cpu().numpy()
        nbrs = NearestNeighbors(radius=self.epsilon, n_jobs=-1).fit(X_cpu)
        distances, _ = nbrs.radius_neighbors(X_cpu)

        signals = torch.zeros(len(X), dtype=torch.float32, device=self.device)
        for i, d in enumerate(distances):
            if len(d) > 0:
                d_t = torch.tensor(d, dtype=torch.float32, device=self.device)
                signals[i] = torch.exp(-d_t**2 / (2 * self.sigma**2)).sum()
        return signals

    def _assign_points(self, X, signals, clusters):
        """Assigns points to clusters by maximizing score (alpha * signal - beta * distance)."""
        if not clusters:
            return {}

        cluster_centers = torch.stack([pts.mean(dim=0) for pts in clusters.values()])
        dists = torch.cdist(X, cluster_centers)
        scores = -dists * self.beta + signals.unsqueeze(1) * self.alpha
        best_clusters = scores.argmax(dim=1)

        new_clusters = {i: [] for i in range(len(cluster_centers))}
        for idx, c in enumerate(best_clusters.tolist()):
            new_clusters[c].append(X[idx])

        return {i: torch.stack(pts) if pts else torch.empty((0, X.shape[1]), device=self.device)
                for i, pts in new_clusters.items()}

    def _merge_clusters(self, clusters):
        """Merges clusters that have centroids closer than max_cluster_distance."""
        if not clusters:
            return {}

        centroids = torch.stack([pts.mean(dim=0) for pts in clusters.values()])
        used = set()
        new_clusters, cluster_count = {}, 0
        keys = list(clusters.keys())

        for i, key_i in enumerate(keys):
            if key_i in used:
                continue
            current_points = [clusters[key_i]]
            for j, key_j in enumerate(keys):
                if i != j and key_j not in used:
                    dist = torch.norm(centroids[i] - centroids[j])
                    if dist < self.max_cluster_distance:
                        current_points.append(clusters[key_j])
                        used.add(key_j)
            new_clusters[cluster_count] = torch.cat(current_points, dim=0)
            cluster_count += 1
        return new_clusters

    def _filter_clusters(self, clusters):
        """Removes clusters lacking min_points_per_cluster."""
        return {i: pts for i, pts in clusters.items()
                if pts.shape[0] >= self.min_points_per_cluster}

    def _clusters_equal(self, c1, c2):
        if len(c1) != len(c2):
            return False
        c1_hash = [(len(v), torch.round(v.mean(dim=0), decimals=6).cpu().numpy().tobytes())
                   for v in c1.values()]
        c2_hash = [(len(v), torch.round(v.mean(dim=0), decimals=6).cpu().numpy().tobytes())
                   for v in c2.values()]
        return set(c1_hash) == set(c2_hash)

    def _clusters_to_labels(self, X, clusters):
        labels = -torch.ones(len(X), dtype=torch.long, device=self.device)
        for cid, pts in clusters.items():
            if pts.shape[0] > 0:
                for p in pts:
                    dists = torch.norm(X - p, dim=1)
                    idx = torch.argmin(dists).item()
                    labels[idx] = cid
        return labels.cpu().numpy()

    # --- Main fit_predict ---
    def fit_predict(self, X):
        """
        Fits the TRCC model to the provided data X and returns the cluster labels.
        
        Args:
            X: Input array/tensor containing the points.
            
        Returns:
            numpy.ndarray: An array of cluster labels mapped back to the CPU.
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        clusters = self._initialize_clusters(X)
        prev_clusters = None

        for _ in range(self.max_iter):
            signals = self._calculate_signals(X)
            clusters = self._assign_points(X, signals, clusters)
            clusters = self._merge_clusters(clusters)
            clusters = self._filter_clusters(clusters)

            if prev_clusters is not None and self._clusters_equal(prev_clusters, clusters):
                break
            prev_clusters = {i: v.clone() for i, v in clusters.items()}

        self.labels_ = self._clusters_to_labels(X, clusters)
        return self.labels_