import numpy as np
from scipy.spatial import distance_matrix, cKDTree
from scipy.spatial.distance import pdist

def find_centroids(pred_coords, coords_prob, threshold=0.5, 
                   cluster_distance=1.52, use_weighted_avg=True, clash_distance=2.2,
                   dedupe_decimals=6, tol=1e-8):
    """
    Returns centroids after clustering, weighted/best selection, and clash removal.
    Clash removal uses global greedy non-maximum suppression (NMS) with prob sorting,
    ensuring minimum pairwise distance >= clash_distance - tol.
    """
    valid_indices = np.where(coords_prob >= threshold)[0]
    refined_coords = pred_coords[valid_indices]
    refined_probs  = coords_prob[valid_indices]
    if refined_coords.size == 0:
        return None

    dist_mat = distance_matrix(refined_coords, refined_coords)
    clusters = []
    visited = set()
    for i in range(len(refined_coords)):
        if i in visited:
            continue
        neighbors = np.where((dist_mat[i] < cluster_distance) & (dist_mat[i] > 0))[0]
        if neighbors.size > 0 and refined_probs[i] < np.max(refined_probs[neighbors]):
            continue
        cluster = [i]
        for nb in neighbors:
            if nb not in visited:
                cluster.append(nb)
                visited.add(nb)
        clusters.append(cluster)

    finals = []
    final_probs = []
    for cl in clusters:
        cc = refined_coords[cl]
        pp = refined_probs[cl]
        best_idx = np.argmax(pp)
        best_prob = pp[best_idx]
        if use_weighted_avg and len(cl) > 1:
            w = pp / (pp.sum() + 1e-12)
            centroid = np.average(cc, axis=0, weights=w)
        else:
            centroid = cc[best_idx]
        finals.append(centroid)
        final_probs.append(best_prob)

    if len(finals) == 0:
        return None

    final_centroids = np.asarray(finals, dtype=float)
    final_probs     = np.asarray(final_probs, dtype=float)

    tree = cKDTree(final_centroids)
    neighbor_lists = tree.query_ball_point(final_centroids, r=clash_distance - tol)

    order = np.argsort(-final_probs)
    blocked = np.zeros(len(final_centroids), dtype=bool)
    keep_idx = []

    for idx in order:
        if blocked[idx]:
            continue
        keep_idx.append(idx)
        for nb in neighbor_lists[idx]:
            blocked[nb] = True

    final_centroids = final_centroids[keep_idx]
    final_probs     = final_probs[keep_idx]

    if final_centroids.shape[0] > 1:
        rounded = np.round(final_centroids, dedupe_decimals)
        uniq, uniq_idx = np.unique(rounded, axis=0, return_index=True)
        order2 = np.argsort(uniq_idx)
        final_centroids = final_centroids[uniq_idx[order2]]
        # final_probs     = final_probs[uniq_idx[order2]]

    return final_centroids
