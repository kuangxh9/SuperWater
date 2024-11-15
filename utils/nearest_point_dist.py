import torch

def get_nearest_point_distances(set1, set2):
    # set2 is target
    # Ensure set1 and set2 are tensors and of float type
    points1 = torch.tensor(set1, dtype=torch.float)
    points2 = torch.tensor(set2, dtype=torch.float)

    # Calculate distances using broadcasting
    # set2 is broadcasted to match the first dimension of set1
    dists = torch.cdist(points1, points2)  # Removing the singleton dimension for proper broadcasting
    
    # Get the minimum distances and the indices of the points that provide these minimum distances for each batch
    min_dists, indices = torch.min(dists, dim=1)  # Adjust dimension for min calculation
    
    # Convert min_dists from a PyTorch tensor to a NumPy array
    min_dists_np = min_dists.numpy()
    
    return min_dists_np, indices