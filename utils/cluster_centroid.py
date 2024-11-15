import numpy as np

def find_centroid(coordinates, clusters):
    # Find the unique cluster labels
    unique_clusters = np.unique(clusters)
    # Initialize a dictionary to store centroids
    centroids = []

    # Calculate the centroid for each cluster
    for cluster in unique_clusters:
        if cluster != -1:  # Exclude noise points
            points_in_cluster = coordinates[clusters == cluster]
            centroid = points_in_cluster.mean(axis=0)
            centroids.append(centroid)
    return np.array(centroids)