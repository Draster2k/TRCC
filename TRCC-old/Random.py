"""
Random Dataset Testing (TRCC).
Runs the TRCC algorithm on various synthetic CSV datasets and saves the output plots.
"""
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TRCC import TRCC   # import the manual TRCC class


DATA_DIR = "Data/Clustering_Data"
RESULTS_DIR = "results"
TRCC_DIR = os.path.join(RESULTS_DIR, "trcc_manual")

os.makedirs(TRCC_DIR, exist_ok=True)


def save_plot(X, labels, title, path):
    """Utility to quickly save a scatter plot."""
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab20", s=10)
    plt.title(title)
    plt.savefig(path)
    plt.close()


def main():
    """Main execution block iterating over datasets."""
    dataset_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    # Skip problematic slow datasets for now
    skip_list = {"basic4", "boxes3", "boxes2", "wave", "chrome"}

    for file in dataset_files:
        name = os.path.splitext(os.path.basename(file))[0]
        if name in skip_list:
            print(f"\nSkipping {name} (too slow for testing)")
            continue

        print(f"\nProcessing {name}...")

        df = pd.read_csv(file)
        X = df.values

        trcc = TRCC(
            k=10,
            max_iter=20,
            max_cluster_distance=1.5,
            min_points_per_cluster=5,
            epsilon=0.5,
            sigma=1.0,
            alpha=1.0,
            beta=1.0,
            device="cpu"
        )

        start = time.time()
        y_trcc = trcc.fit_predict(X)
        elapsed = round(time.time() - start, 2)

        print(f"  Time taken: {elapsed} seconds")
        print(f"  Final clusters: {len(set(y_trcc))}")

        save_plot(X, y_trcc, f"TRCC - {name}", os.path.join(TRCC_DIR, f"{name}.png"))

    print("\nAll datasets processed. Results saved in results/trcc_manual/")


if __name__ == "__main__":
    main()