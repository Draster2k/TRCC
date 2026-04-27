"""
K-Means Clustering on Diamonds dataset.
Imputes, normalizes and clusters the cleaned diamonds dataset.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load dataset
data = pd.read_csv('diamonds (cleaned).csv')

# Handle missing values
# Impute numeric columns with mean
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# Impute categorical columns with most frequent value
categorical_cols = data.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

# Normalize numeric data
numeric_data = data[numeric_cols]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# PCA for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Apply K-means clustering
k = 4  # Adjust as needed
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)
clusters = kmeans.labels_

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x', label='Centroids')
plt.title(f'K-Means Clustering with k={k}')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()