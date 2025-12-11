import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read data (format: 366 rows × 24 columns, each row represents 24-hour output for one day)
data = pd.read_excel('pv2024.xlsx', engine='openpyxl')  # Date used as index
X = data.values

# Standardize data (remove scale effects)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.mixture import GaussianMixture

n_components = range(1, 11)
aic_values = []

for k in n_components:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    aic_values.append(gmm.aic(X_scaled))

# Output AIC results for the elbow method
elbow_table = pd.DataFrame({
    'Number of Clusters (K)': n_components,
    'AIC': aic_values,
})
print("Elbow method AIC results:")
print(elbow_table.round(2))

best_k = 3  # Choose the actual value based on the AIC curve
gmm = GaussianMixture(n_components=best_k, random_state=42)
labels = gmm.fit_predict(X_scaled)

# Get mean profiles for each cluster (inverse transform to original scale)
cluster_means = scaler.inverse_transform(gmm.means_)

# Export table of cluster mean profiles
mean_curves_table = pd.DataFrame(
    cluster_means,
    columns=[f'Hour_{i+1}' for i in range(24)],
    index=[f'Cluster_{i+1}' for i in range(best_k)]
)
mean_curves_table.to_csv("mean_curves_pv.csv", index=True)

# Get membership probability matrix (n_samples × n_clusters)
prob_matrix = gmm.predict_proba(X_scaled)

# Combine original data with probability matrix
prob_df = pd.DataFrame(
    prob_matrix,
    columns=[f'Cluster_{i}_prob' for i in range(gmm.n_components)],
    index=data.index
)
result_df = pd.concat([data, prob_df], axis=1)

# Export full membership probability table to CSV (including original data)
result_df.to_csv("clustering_results_pv.csv", index=True)
