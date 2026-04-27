"""
Hyperparameter Tuning for TRCC using Optuna.
Optimizes the parameters of TRCC_M1 on a synthetic test dataset to maximize the silhouette score.
"""
import optuna
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from TRCC_M1 import trcc_algorithm

# --- 1. Load dataset (replace this block with your real data if needed) ---
print("Loading synthetic test dataset (make_blobs)...")
X, _ = make_blobs(n_samples=20000, centers=5, cluster_std=1.5, random_state=42)
X = StandardScaler().fit_transform(X)

# --- 2. Define the Optuna objective function ---
def objective(trial):
    """
    Optuna objective function to evaluate a given set of TRCC hyperparameters.
    Maximizes the Silhouette Score.
    """
    # Hyperparameter search space
    epsilon = trial.suggest_float("epsilon", 0.1, 2.0)
    sigma = trial.suggest_float("sigma", 0.1, 2.0)
    max_cluster_distance = trial.suggest_float("max_cluster_distance", 0.1, 1.5)
    alpha = trial.suggest_float("alpha", 0.5, 2.0)
    beta = trial.suggest_float("beta", 0.5, 2.0)

    try:
        clusters = trcc_algorithm(
            X,
            k=100,
            max_iter=10,
            max_cluster_distance=max_cluster_distance,
            min_points_per_cluster=50,
            epsilon=epsilon,
            sigma=sigma,
            alpha=alpha,
            beta=beta
        )
    except Exception as e:
        print(f"[ERROR] Trial failed: {e}")
        return -1.0

    # Convert to label array for silhouette scoring
    labels = np.full(len(X), -1)
    for cluster_id, points in clusters.items():
        for p in points:
            idx = np.where((X == p).all(axis=1))[0]
            if len(idx) > 0:
                labels[idx[0]] = cluster_id

    # Filter invalid points
    valid_mask = labels != -1
    unique_clusters = np.unique(labels[valid_mask])

    # Reject trivial clusterings
    if len(unique_clusters) < 2 or np.sum(valid_mask) < 10:
        return -1.0

    score = silhouette_score(X[valid_mask], labels[valid_mask])
    print(f"Trial: score={score:.4f} | eps={epsilon:.3f} | sig={sigma:.3f} | dist={max_cluster_distance:.3f} | α={alpha:.2f} β={beta:.2f}")
    return score

# --- 3. Run the Optuna optimization ---
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# --- 4. Print the best result ---
print("\n✔ Best Silhouette Score:", round(study.best_value, 4))
print("✔ Best Hyperparameters:")
for key, val in study.best_params.items():
    print(f"   - {key}: {val}")