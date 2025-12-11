import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Read data (format: 366 rows × 24 columns, each row represents 24-hour wind speed for one day)
data = pd.read_excel('wind2024.xlsx', engine='openpyxl')  # Date used as index
X = data.values

# Standardize data (keep original scale for visualization if needed)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute WCSS (within-cluster sum of squares) for different K values
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Build elbow-method table
elbow_df = pd.DataFrame({
    'K': k_range,
    'WCSS': wcss,
    'Error_Change_Rate': [0] + [abs(wcss[i] - wcss[i-1]) / wcss[i-1] for i in range(1, len(wcss))]
})

print("Elbow method result table:")
print(elbow_df.round(2))

# Select K=4 according to the elbow method (example choice)
best_k = 4
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Inverse transform centroids back to original scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
hours = np.arange(24)

# Export cluster mean curves table
mean_curves_table = pd.DataFrame(
    centroids,
    columns=[f'Hour_{i+1}' for i in range(24)],
    index=[f'Cluster_{i+1}' for i in range(best_k)]
)

mean_curves_table.to_csv("mean_curves_wind.csv", index=True)

# Compute distances from each sample to each cluster center
distances = kmeans.transform(X_scaled)

# Convert distances to probabilities (Softmax normalization)
prob_matrix = np.exp(-distances) / np.sum(np.exp(-distances), axis=1, keepdims=True)

membership_df = pd.DataFrame(
    prob_matrix,
    columns=[f'Cluster_{i}' for i in range(best_k)],
    index=data.index
)
result_df = pd.concat([data, membership_df], axis=1)

# Export full membership probability table to CSV (including original data)
result_df.to_csv("clustering_results_wind.csv", index=True)

# Elbow method plot
plt.figure(figsize=(10, 5))
plt.plot(k_range, wcss, 'bo-', markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
plt.title('Elbow Method for Optimal K', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(k_range)
plt.show()

# Membership probability heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(membership_df.T, cmap="YlGnBu", cbar_kws={'label': 'Membership Probability'})
plt.xlabel('Day Index', fontsize=12)
plt.ylabel('Cluster', fontsize=12)
plt.title('Membership Probability Heatmap', fontsize=14)
plt.show()
