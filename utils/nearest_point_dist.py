import torch

def get_nearest_point_distances(set1, set2):
    points1 = torch.tensor(set1, dtype=torch.float)
    points2 = torch.tensor(set2, dtype=torch.float)
    dists = torch.cdist(points1, points2) 
    min_dists, indices = torch.min(dists, dim=1) 
    min_dists_np = min_dists.numpy()
    return min_dists_np, indices