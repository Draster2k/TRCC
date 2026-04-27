"""
Boston Crime Data Clustering using TRCC.
Loads geospatial crime data and clusters incident locations using the TRCC algorithm.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from TRCC_M1 import trcc_algorithm

# Paths to the dataset
DATA_DIR = 'archive-4'
DATA_FILE = os.path.join(DATA_DIR, 'crime.csv')


def load_boston_crime_data(filepath):
    """Load and preprocess the Boston crime dataset."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback to a different encoding if UTF-8 fails
        df = pd.read_csv(filepath, encoding='ISO-8859-1')

    # Extract relevant columns (latitude and longitude)
    df = df[['Lat', 'Long']].dropna()

    # Filter out invalid coordinates
    df = df[(df['Lat'] > 42) & (df['Lat'] < 43)]
    df = df[(df['Long'] > -71.2) & (df['Long'] < -70.9)]

    return df


def plot_clusters(X, clusters, title="TRCC Clustering of Boston Crime Data"):
    """Plot the geospatial clusters on a scatter plot."""
    plt.figure(figsize=(10, 8))
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(clusters)))

    for i, points in clusters.items():
        points = np.array(points)
        plt.scatter(points[:, 1], points[:, 0], label=f'Cluster {i}', color=colors[i], alpha=0.6)
        centroid = np.mean(points, axis=0)
        plt.scatter(centroid[1], centroid[0], color='black', marker='x', s=100)

    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """Main execution block: Load data, scale, run TRCC, and visualize."""
    # Load the data
    df = load_boston_crime_data(DATA_FILE)
    X = df.values

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = load_boston_crime_data(DATA_FILE)
    X_scaled = X_scaled[:5000]  # Use only first 5000 rows

    # Run the TRCC algorithm
    clusters = trcc_algorithm(
        X_scaled,
        k=10,  # Reduce number of initial clusters
        max_iter=25,  # Lower iterations (was 50)
        max_cluster_distance=2.0,  # Increase merging distance
        min_points_per_cluster=5,  # Reduce required cluster size
        epsilon=0.5,  # Reduce neighborhood radius
        sigma=0.5  # Reduce signal influence
    )
    # Plot the clusters
    plot_clusters(X, clusters, title="TRCC Clustering of Boston Crime Data")


if __name__ == "__main__":
    main()