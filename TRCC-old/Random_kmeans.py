"""
Random Dataset Testing (KMeans).
Runs KMeans on various synthetic CSV datasets and saves scatter plots.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ---- Paths ----
DATA_DIR = "Data/Clustering_Data"
RESULTS_DIR = "results"
KMEANS_DIR = os.path.join(RESULTS_DIR, "kmeans")

# Create output directory
os.makedirs(KMEANS_DIR, exist_ok=True)


# ---- Utility: plot and save ----
def save_plot(X, labels, title, path):
    """Utility to quickly save a scatter plot."""
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab20", s=10)
    plt.title(title)
    plt.savefig(path)
    plt.close()


# ---- Load datasets and cluster ----
def main():
    """Main execution block iterating over datasets."""
    dataset_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    for file in dataset_files:
        name = os.path.splitext(os.path.basename(file))[0]
        print(f"Processing {name}...")

        # Load dataset
        df = pd.read_csv(file)
        X = df.values

        # --- KMeans ---
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        y_kmeans = kmeans.fit_predict(X)

        # Save visualization
        save_plot(X, y_kmeans, f"KMeans - {name}", os.path.join(KMEANS_DIR, f"{name}.png"))

    print("KMeans clustering finished. Results saved in results/kmeans/")


if __name__ == "__main__":
    main()