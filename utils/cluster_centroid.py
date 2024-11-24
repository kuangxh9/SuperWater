import numpy as np
from scipy.spatial import distance_matrix

def find_centroids(pred_coords, coords_prob, threshold=0.5, cluster_distance=1.52, use_weighted_avg=True, clash_distance=1.52):
    valid_indices = np.where(coords_prob >= threshold)[0]
    refined_coords = pred_coords[valid_indices]
    refined_probs = coords_prob[valid_indices]
    
    if len(refined_coords) == 0:
        return None
    
    dist_matrix = distance_matrix(refined_coords, refined_coords)
    
    clusters = []
    visited = set() 
    for i in range(len(refined_coords)):
        if i in visited:
            continue
        
        neighbors = np.where((dist_matrix[i] < cluster_distance) & (dist_matrix[i] > 0))[0]
        
        if len(neighbors) > 0:
            neighbor_probs = refined_probs[neighbors]
            if refined_probs[i] < np.max(neighbor_probs):
                continue
        
        cluster = [i] 
        
        for neighbor in neighbors:
            if neighbor not in visited:
                cluster.append(neighbor)
                visited.add(neighbor)
        
        clusters.append(cluster)
    
    final_centroids = []
    final_probs = []
    
    for cluster in clusters:
        cluster_coords = refined_coords[cluster]
        cluster_probs = refined_probs[cluster]
        
        best_idx = np.argmax(cluster_probs)
        best_coord = cluster_coords[best_idx]
        best_prob = cluster_probs[best_idx]

        if use_weighted_avg and len(cluster_coords) > 1:
            normalized_probs = cluster_probs / np.sum(cluster_probs)
            weighted_centroid = np.average(cluster_coords, axis=0, weights=normalized_probs)
            final_centroids.append(weighted_centroid)
        else:
            final_centroids.append(best_coord)
        
        final_probs.append(best_prob)
    
    final_centroids = np.array(final_centroids)
    final_probs = np.array(final_probs)

    dist_matrix = distance_matrix(final_centroids, final_centroids)
    np.fill_diagonal(dist_matrix, np.inf) 
    to_delete = set()

    for i in range(len(final_centroids)):
        close_points = np.where(dist_matrix[i] <= clash_distance)[0] # 2.25
        if len(close_points) > 0:
            max_prob_idx = close_points[np.argmax(final_probs[close_points])]
            to_delete.update(close_points[close_points != max_prob_idx])

    final_centroids = np.delete(final_centroids, list(to_delete), axis=0)
    final_probs = np.delete(final_probs, list(to_delete))
    return np.array(final_centroids)