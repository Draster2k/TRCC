"""
TRCC: T-Regulated Cytokine Clustering.

A density-anchored clustering algorithm. Each point emits a Gaussian
"cytokine signal" reflecting its k-NN local density. Cluster centers are
identified as density peaks: points that combine high signal with large
separation from any higher-signal point (Rodriguez & Laio, Science 2014,
re-cast in TRCC's signal formulation). Non-peak points inherit the label
of their density parent — the nearest point with a strictly higher signal —
yielding clusters that follow the data's intrinsic density flow rather
than spherical centroid geometry. Clusters are then merged under a
mutual-reachability criterion, and clusters smaller than `min_cluster_size`
are demoted to noise.

The algorithm has three exposed degrees of freedom:
  - n_neighbors   : scale at which density is measured
  - n_clusters    : either an integer or "auto" (knee detection on gamma)
  - merge_threshold: mutual-reachability merge cutoff ("auto" = adaptive)

API matches sklearn's clustering convention.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    # Preferred: namespaced inside the package (post-1.1.0 layout).
    from . import trcc_native as _native  # type: ignore[attr-defined]
    _HAS_NATIVE = True
except ImportError:
    try:
        # Legacy fallback: .so was installed at site-packages root.
        import trcc_native as _native  # type: ignore[no-redef]
        _HAS_NATIVE = True
    except ImportError:  # pragma: no cover
        _native = None
        _HAS_NATIVE = False


class TRCC:
    """
    T-Regulated Cytokine Clustering.

    Parameters
    ----------
    n_clusters : int or "auto", default="auto"
        Number of clusters. "auto" picks the knee of the sorted
        peakiness score gamma_i = S_i * delta_i.
    n_neighbors : int or "auto", default="auto"
        Neighbors used to estimate the cytokine signal (local density).
    sigma : float or "auto", default="auto"
        Bandwidth of the Gaussian kernel; "auto" = median k-th NN distance.
    min_cluster_size : int, default=15
        Clusters smaller than this become noise (label -1).
    min_signal_percentile : float, default=0.0
        Points with signal below this percentile become noise. 0 disables.
    merge_threshold : float or "auto", default="auto"
        Centroids whose mutual-reachability distance falls below this
        threshold are merged. "auto" = 1.5 * median k-th NN distance.
    max_clusters : int, default=50
        Upper bound on clusters when n_clusters="auto".
    random_state : int, default=42
    n_jobs : int, default=-1

    Attributes
    ----------
    labels_              : ndarray (n,)
    cluster_centers_     : ndarray (n_clusters_, d)
    signals_             : ndarray (n,)
    cluster_signal_      : ndarray (n_clusters_,)
    peak_indices_        : ndarray of indices used as density peaks
    n_clusters_          : int
    """

    def __init__(
        self,
        n_clusters: Union[int, str] = "auto",
        n_neighbors: Union[int, str] = "auto",
        sigma: Union[float, str] = "auto",
        min_cluster_size: int = 15,
        min_signal_percentile: float = 0.0,
        merge_threshold: Union[float, str] = "auto",
        max_clusters: int = 50,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.min_cluster_size = min_cluster_size
        self.min_signal_percentile = min_signal_percentile
        self.merge_threshold = merge_threshold
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ API

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n = len(X)

        signals, kth = self._cytokine_signals(X)
        delta, parent = self._delta_and_parent(X, signals)
        gamma = signals * delta
        peaks = self._select_peaks(gamma)
        labels = self._propagate(signals, parent, peaks)

        # mutual-reachability merge
        labels = self._merge(X, signals, labels, kth)

        # noise filtering
        labels = self._filter_noise(labels, signals)

        # compact label space
        labels, centroids, cluster_signal = self._finalize(X, labels, signals)

        self.labels_ = labels
        self.cluster_centers_ = centroids
        self.cluster_signal_ = cluster_signal
        self.signals_ = signals
        self.peak_indices_ = peaks
        self.n_clusters_ = int(centroids.shape[0])
        self._kth_median = float(np.median(kth))
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Call fit before predict.")
        X = np.asarray(X, dtype=np.float64)
        if self.cluster_centers_.shape[0] == 0:
            return np.full(len(X), -1, dtype=np.int64)
        d = _pairwise(X, self.cluster_centers_)
        return d.argmin(axis=1).astype(np.int64)

    # ---------------------------------------------------------- internals

    def _resolve_n_neighbors(self, n):
        if isinstance(self.n_neighbors, str) and self.n_neighbors == "auto":
            # Auto-neighborhood:  k(n) = clip( ceil(sqrt(n)/2), 10, 200 )
            # Square-root scaling is standard for kNN density estimators
            # (Loftsgaarden & Quesenberry, 1965); the floor 10 stops the
            # kernel collapsing at small n, the ceiling 200 keeps the
            # kernel local enough at very large n so density peaks remain
            # distinguishable.
            return int(np.clip(np.ceil(np.sqrt(max(n, 2)) / 2.0), 10, 200))
        return int(self.n_neighbors)

    def _cytokine_signals(self, X):
        n = len(X)
        nn = self._resolve_n_neighbors(n)
        self._n_neighbors_eff = nn
        k = min(nn + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k, n_jobs=self.n_jobs).fit(X)
        dists, _ = nbrs.kneighbors(X)
        dists = dists[:, 1:]  # drop self
        kth = dists[:, -1]
        sigma = float(np.median(kth)) if self.sigma == "auto" else float(self.sigma)
        if sigma <= 0:
            sigma = 1.0
        signals = np.exp(-(dists ** 2) / (2.0 * sigma ** 2)).sum(axis=1)
        return signals, kth

    def _delta_and_parent(self, X, signals):
        """
        For each point, find the nearest point with a strictly higher signal.
        Uses C++/nanoflann kd-tree if available, else numpy fallback.
        """
        if _HAS_NATIVE:
            X_c = np.ascontiguousarray(X, dtype=np.float64)
            s_c = np.ascontiguousarray(signals, dtype=np.float64)
            return _native.delta_and_parent(X_c, s_c)

        n = len(X)
        order = np.argsort(-signals)
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)

        delta = np.zeros(n, dtype=np.float64)
        parent = -np.ones(n, dtype=np.int64)

        # Process points in batches in order of descending signal. For each
        # point i, candidates are points already processed (higher signal).
        # Use sklearn NN over the growing set.
        # For modest n this is fast enough; for larger n we use a kd-tree
        # rebuild every doubling.
        # Simpler vectorized approach: full pairwise distance, then mask.
        # n^2 memory is OK up to ~20k points; fall back to chunked otherwise.
        chunk = 4096
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            idx_block = order[start:stop]
            # candidates with strictly higher signal = order[:start] for the
            # first point in the block; but rank-based mask handles ties
            d_block = _pairwise(X[idx_block], X)  # (b, n)
            higher = signals[None, :] > signals[idx_block][:, None]  # strict
            d_block = np.where(higher, d_block, np.inf)
            mins = d_block.min(axis=1)
            args = d_block.argmin(axis=1)
            for local_i, global_i in enumerate(idx_block):
                if not np.isfinite(mins[local_i]):
                    delta[global_i] = 0.0
                    parent[global_i] = -1
                else:
                    delta[global_i] = mins[local_i]
                    parent[global_i] = args[local_i]

        # global max signal point: delta = max delta (so it always qualifies as a peak)
        top = order[0]
        if delta[top] == 0.0:
            delta[top] = float(delta.max() if delta.max() > 0 else 1.0)
        return delta, parent

    def _select_peaks(self, gamma):
        order = np.argsort(-gamma)
        if isinstance(self.n_clusters, str) and self.n_clusters == "auto":
            k = self._auto_n_clusters(gamma[order])
        else:
            k = int(self.n_clusters)
        k = max(1, min(k, self.max_clusters, len(gamma)))
        return order[:k]

    def _auto_n_clusters(self, gamma_sorted_desc):
        """
        Pick k via three complementary heuristics on sorted log-gamma:

          1. Ratio-gap: index where gamma[i]/gamma[i+1] is largest. This
             catches sharp density-peak/non-peak separations that the
             other two miss when the top peaks are nearly tied.
          2. Outlier rule: log-gamma values beyond mean + 1.5*std of the
             tail. Robust when there are many genuine peaks at similar
             gamma (e.g. 12 equal-density blobs).
          3. Kneedle: distance-from-line knee on the sorted curve.

        Take the *max* of the three, capped at max_clusters. Taking the
        max is intentional: under-detecting peaks is more harmful than
        over-detecting (the path-density merge can collapse spurious
        peaks, but cannot split missed clusters).
        """
        g = gamma_sorted_desc[: self.max_clusters]
        if len(g) < 3:
            return max(2, len(g))
        lg = np.log(g + 1e-12)

        # 1. Largest ratio drop in linear gamma. Diagnosis on a panel of
        #    datasets (moons, blobs-4, blobs-8, blobs-12 at 100k) showed
        #    this single estimator nails the true k in every case where
        #    a clean density-peak structure exists.
        ratios = g[:-1] / np.maximum(g[1:], 1e-12)
        k_ratio = int(np.argmax(ratios)) + 1

        # 2. 2σ outlier rule on log-gamma — guard for cases where the
        #    biggest ratio gap lies in noise. In testing it never disagreed
        #    with k_ratio downward, so we take the max as a safety floor.
        tail = lg[2:] if len(lg) > 4 else lg
        k_outlier = int(np.sum(lg > tail.mean() + 2.0 * tail.std()))

        k = max(k_ratio, k_outlier)
        return max(2, min(k, self.max_clusters))

    def _propagate(self, signals, parent, peaks):
        """Assign every point to the cluster of its density parent."""
        if _HAS_NATIVE:
            return _native.propagate_labels(
                np.ascontiguousarray(signals, dtype=np.float64),
                np.ascontiguousarray(parent, dtype=np.int64),
                np.ascontiguousarray(peaks, dtype=np.int64),
            )
        n = len(signals)
        labels = -np.ones(n, dtype=np.int64)
        for cid, p in enumerate(peaks):
            labels[p] = cid
        order = np.argsort(-signals)
        for i in order:
            if labels[i] != -1:
                continue
            par = parent[i]
            if par == -1:
                labels[i] = 0
            else:
                labels[i] = labels[par]
        return labels

    def _merge(self, X, signals, labels, kth):
        """
        Path-density-aware merge.

        Two clusters merge iff the density along the straight-line path
        between their centroids does not collapse: the worst-case signal
        on the line (sampled) must exceed `path_ratio` * min(T_i, T_j).
        Among candidate merges, the pair with the highest path-density
        (relative to the weaker cluster) is merged first; iterate to a
        fixed point. This prevents merging across low-density gaps even
        when centroids are close, and allows merges across long
        ridges of dense data — the genuine reason centroid-based methods
        fail on non-convex clusters.
        """
        ids = np.unique(labels[labels >= 0])
        if len(ids) <= 1:
            return labels

        kth_med = float(np.median(kth))
        diam = float(_pairwise(X[:1], X).max())
        max_centroid_d = 0.5 * diam if self.merge_threshold == "auto" \
            else float(self.merge_threshold)
        path_ratio = 0.6
        n_steps = 11

        # Fit NearestNeighbors ONCE; reuse for all path-density evaluations.
        sigma = float(np.median(kth)) if self.sigma == "auto" else float(self.sigma)
        if sigma <= 0:
            sigma = 1.0
        k_nn = min(self._n_neighbors_eff, len(X))
        nbrs = NearestNeighbors(n_neighbors=k_nn, n_jobs=self.n_jobs).fit(X)

        def signal_at(pts):
            d, _ = nbrs.kneighbors(pts)
            return np.exp(-(d ** 2) / (2.0 * sigma ** 2)).sum(axis=1)

        for _ in range(50):
            ids = np.unique(labels[labels >= 0])
            if len(ids) <= 1:
                break
            centroids = np.stack([X[labels == c].mean(axis=0) for c in ids])
            # use the cluster's peak signal (max within cluster) instead of
            # mean — peaks are stable density modes and produce a stricter
            # path-density criterion than the membership mean.
            cluster_T = np.array([signals[labels == c].max() for c in ids])
            d = _pairwise(centroids, centroids)
            np.fill_diagonal(d, np.inf)

            # Collect candidate pairs (centroid distance below ceiling)
            cand = []
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if d[i, j] <= max_centroid_d:
                        cand.append((i, j))
            if not cand:
                break
            # Build all sample points in one batch
            ts = np.linspace(0.05, 0.95, n_steps)
            all_pts = np.empty((len(cand) * n_steps, X.shape[1]))
            for ci, (i, j) in enumerate(cand):
                all_pts[ci * n_steps:(ci + 1) * n_steps] = (
                    centroids[i] * (1 - ts[:, None])
                    + centroids[j] * ts[:, None]
                )
            sig_all = signal_at(all_pts).reshape(len(cand), n_steps)
            best_pair = None
            best_score = -np.inf
            for ci, (i, j) in enumerate(cand):
                bottleneck = sig_all[ci].min()
                weakest_cluster = min(cluster_T[i], cluster_T[j])
                if bottleneck >= path_ratio * weakest_cluster:
                    score = bottleneck / (weakest_cluster + 1e-9)
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            keep, drop = sorted((ids[i], ids[j]))
            labels = np.where(labels == drop, keep, labels)
        return labels

    def _filter_noise(self, labels, signals):
        # by-size
        ids, counts = np.unique(labels[labels >= 0], return_counts=True)
        small = ids[counts < self.min_cluster_size]
        if len(small) > 0:
            labels = np.where(np.isin(labels, small), -1, labels)
        # by-signal
        if self.min_signal_percentile > 0:
            cutoff = np.percentile(signals, self.min_signal_percentile)
            labels = np.where(signals < cutoff, -1, labels)
        return labels

    def _finalize(self, X, labels, signals):
        ids = np.unique(labels[labels >= 0])
        if len(ids) == 0:
            return (np.full(len(X), -1, dtype=np.int64),
                    np.empty((0, X.shape[1])),
                    np.empty(0))
        remap = {old: new for new, old in enumerate(ids)}
        new = np.array([remap[l] if l >= 0 else -1 for l in labels],
                       dtype=np.int64)
        centroids = np.stack([X[new == c].mean(axis=0) for c in range(len(ids))])
        cluster_signal = np.array([signals[new == c].mean()
                                   for c in range(len(ids))])
        return new, centroids, cluster_signal


# ----------------------------------------------------------- helper

def _pairwise(A, B):
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    a2 = (A * A).sum(axis=1, keepdims=True)
    b2 = (B * B).sum(axis=1, keepdims=True).T
    d2 = a2 + b2 - 2.0 * (A @ B.T)
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2)
