import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# =============================================================================
# 2. K-Means Clustering
# =============================================================================
# GOAL: Group a set of data points into a predefined number of clusters (K).
# THE LOOP: This is a two-step iterative process:
#   1. Assignment Step: Assign each data point to its closest cluster center (centroid).
#   2. Update Step: Move each centroid to the average position of the points
#      assigned to it.
# We repeat this until the centroids stop moving.
# -----------------------------------------------------------------------------

def kmeans_clustering_loop(X, K=3, max_iters=100):
    """
    Implements K-Means clustering from scratch.

    Args:
        X (np.array): The data to cluster.
        K (int): The number of clusters to create.
        max_iters (int): Maximum number of iterations to prevent infinite loops.

    Returns:
        tuple: The final cluster assignments and centroid locations.
    """
    print("\n--- Starting K-Means Clustering Loop ---")
    n_samples, n_features = X.shape

    # Step 1: Initialize centroids randomly from the data points
    random_indices = np.random.choice(n_samples, K, replace=False)
    centroids = X[random_indices]

    # This is the core "learning loop"!
    for i in range(max_iters):
        # Step 2: Assignment Step
        # Create a list to store the cluster index for each point
        cluster_assignments = []
        for point in X:
            # Calculate distance from the point to each centroid
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            # Assign the point to the cluster with the closest centroid
            closest_centroid_index = np.argmin(distances)
            cluster_assignments.append(closest_centroid_index)

        cluster_assignments = np.array(cluster_assignments)

        # Store old centroids to check for convergence
        old_centroids = np.copy(centroids)

        # Step 3: Update Step
        for k in range(K):
            # Get all points assigned to this cluster
            points_in_cluster = X[cluster_assignments == k]
            # If a cluster has points, update its centroid to the mean of those points
            if len(points_in_cluster) > 0:
                centroids[k] = np.mean(points_in_cluster, axis=0)

        # Step 4: Check for convergence (if centroids haven't moved)
        if np.all(centroids == old_centroids):
            print(f"Converged at iteration {i}!")
            break

        if i % 10 == 0:
            print(f"Iteration {i}: Centroids are updating...")

    print("--- K-Means Clustering Loop Finished ---")
    return cluster_assignments, centroids


# --- Example Usage for K-Means ---
# Generate some sample data with distinct blobs
X_cluster, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=1.0)

# Run the algorithm
labels, final_centroids = kmeans_clustering_loop(X_cluster, K=3)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7, label='Data Points')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', s=200, marker='X', edgecolor='black',
            label='Final Centroids')
plt.title('K-Means Clustering from Scratch')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()