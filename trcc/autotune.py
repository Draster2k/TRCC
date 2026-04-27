"""
Hyperparameter tuning for TRCC using Optuna.

Tunes (n_neighbors, sigma, min_cluster_size, merge_threshold) against a
user-provided dataset. Default objective is silhouette on a sub-sample
(O(N^2) full silhouette is impractical at scale); supports ARI when
ground-truth labels are provided.

Usage
-----
    from trcc.autotune import tune
    best = tune(X, y_true=None, n_trials=40)
    print(best.params)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import optuna
from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score,
                             silhouette_score)

from .core import TRCC


@dataclass
class TuneResult:
    params: dict
    score: float
    study: optuna.Study


def _silhouette_subsample(X, labels, max_n=4000, rng=None):
    valid = labels != -1
    if valid.sum() < 10:
        return -1.0
    n_clusters = len(np.unique(labels[valid]))
    if n_clusters < 2:
        return -1.0
    if valid.sum() > max_n:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(np.where(valid)[0], size=max_n, replace=False)
        return float(silhouette_score(X[idx], labels[idx]))
    return float(silhouette_score(X[valid], labels[valid]))


def tune(
    X,
    y_true: Optional[np.ndarray] = None,
    n_trials: int = 40,
    objective: str = "auto",
    timeout: Optional[float] = None,
    seed: int = 0,
    verbose: bool = False,
) -> TuneResult:
    """
    Parameters
    ----------
    X : (n, d) array
    y_true : optional ground-truth labels. If provided and `objective="auto"`,
        ARI is maximized; otherwise silhouette.
    n_trials : Optuna trial budget.
    objective : "auto" | "silhouette" | "ari" | "nmi".
    """
    if objective == "auto":
        objective = "ari" if y_true is not None else "silhouette"

    rng = np.random.default_rng(seed)

    def obj(trial):
        params = dict(
            n_neighbors=trial.suggest_int("n_neighbors", 8, 60),
            sigma=trial.suggest_float("sigma", 0.05, 2.0, log=True),
            min_cluster_size=trial.suggest_int("min_cluster_size", 5, 60),
            merge_threshold=trial.suggest_float("merge_threshold", 0.05, 5.0,
                                                log=True),
            random_state=seed,
        )
        try:
            labels = TRCC(**params).fit_predict(X)
        except Exception as e:
            if verbose:
                print(f"trial failed: {e}")
            return -1.0

        valid = labels != -1
        n_c = len(np.unique(labels[valid])) if valid.any() else 0
        if n_c < 2:
            return -1.0

        if objective == "silhouette":
            return _silhouette_subsample(X, labels, rng=rng)
        if objective == "ari":
            return float(adjusted_rand_score(y_true, labels))
        if objective == "nmi":
            return float(normalized_mutual_info_score(y_true, labels))
        raise ValueError(objective)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(obj, n_trials=n_trials, timeout=timeout, show_progress_bar=verbose)
    return TuneResult(params=study.best_params,
                      score=study.best_value,
                      study=study)


if __name__ == "__main__":
    import argparse
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler

    p = argparse.ArgumentParser()
    p.add_argument("--n_trials", type=int, default=40)
    args = p.parse_args()

    X, y = make_blobs(n_samples=2000, centers=5, cluster_std=0.7, random_state=0)
    X = StandardScaler().fit_transform(X)
    res = tune(X, y_true=y, n_trials=args.n_trials, verbose=True)
    print("best ARI:", round(res.score, 4))
    print("best params:", res.params)
