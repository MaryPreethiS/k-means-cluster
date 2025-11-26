# ============================================================
#   K-MEANS CLUSTERING FROM SCRATCH (NUMPY ONLY)
#   Full Implementation + WCSS + Elbow Method + Silhouette Score
#   + Final Cluster Visualization
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# ------------------------------
# 1. Initialize Centroids
# ------------------------------
def initialize_centroids(X, k):
    np.random.seed(42)
    random_indices = np.random.choice(len(X), k, replace=False)
    return X[random_indices]

# ------------------------------
# 2. Compute Euclidean Distances
# ------------------------------
def compute_distances(X, centroids):
    return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

# ------------------------------
# 3. Assign Cluster Labels
# ------------------------------
def assign_clusters(distances):
    return np.argmin(distances, axis=1)

# ------------------------------
# 4. Update Centroids
# ------------------------------
def update_centroids(X, clusters, k):
    return np.array([X[clusters == i].mean(axis=0) for i in range(k)])

# ------------------------------
# 5. Main K-Means Function
# ------------------------------
def kmeans_from_scratch(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)

    for _ in range(max_iters):
        old_centroids = centroids.copy()
        distances = compute_distances(X, centroids)
        clusters = assign_clusters(distances)
        centroids = update_centroids(X, clusters, k)

        if np.allclose(centroids, old_centroids):  # convergence check
            break

    return centroids, clusters

# ============================================================
#               DATA GENERATION (500 points, 5 clusters)
# ============================================================
X, y = make_blobs(n_samples=500, centers=5, cluster_std=0.9, random_state=42)

# ============================================================
#     RUN K-MEANS FOR K = 2 to 10 AND COMPUTE WCSS VALUES
# ============================================================
def compute_wcss(X, centroids, clusters):
    wcss = 0
    for i in range(len(centroids)):
        cluster_points = X[clusters == i]
        wcss += np.sum((cluster_points - centroids[i]) ** 2)
    return wcss

wcss_values = []
k_range = range(2, 11)

for k in k_range:
    centroids, clusters = kmeans_from_scratch(X, k)
    wcss = compute_wcss(X, centroids, clusters)
    wcss_values.append(wcss)

# ============================================================
#                     ELBOW PLOT
# ============================================================
plt.plot(k_range, wcss_values, marker='o')
plt.title("Elbow Plot (WCSS vs K)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid()
plt.show()

# Based on the elbow plot, optimal K ≈ 5
optimal_k = 5

# ============================================================
#        RUN K-MEANS AGAIN FOR THE OPTIMAL K = 5
# ============================================================
centroids, clusters = kmeans_from_scratch(X, optimal_k)

# ============================================================
#     SILHOUETTE SCORE FOR THE CHOSEN K
# ============================================================
sil_score = silhouette_score(X, clusters)
print(f"Silhouette Score for K={optimal_k}: {sil_score:.4f}")

# ============================================================
#               FINAL CLUSTER VISUALIZATION
# ============================================================
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='tab10')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='X')
plt.title(f"Final K-Means Clustering (K = {optimal_k})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid()
plt.show()
