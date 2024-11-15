import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

def match_points_and_get_distances(data1, data2):
    """
    Finds the optimal one-to-one matching between points in data1 and data2
    to minimize the total Euclidean distance, and returns the distances of
    these matched pairs.

    Parameters:
    - data1: A NumPy array of shape (n, 3) representing n points in 3D space.
    - data2: A NumPy array of shape (m, 3) representing m points in 3D space.

    Returns:
    - A list of distances corresponding to the minimum distances between
      optimally matched pairs from data1 to data2.
    """
    
    # Calculate pairwise Euclidean distances using PyTorch
    dist_matrix = cdist(data1, data2)

    # Solve the assignment problem using SciPy's linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # Collect the minimum distances for each optimal pairing into a list
    min_distances = np.array([dist_matrix[i, j] for i, j in zip(row_ind, col_ind)])

    # Optionally, print the matches and their distances
    # for i, j in zip(row_ind, col_ind):
    #     print(f"data1 point {i} is matched with data2 point {j}, distance: {dist_matrix[i, j]:.2f}")

    # Return the list of minimum distances
    return min_distances

# # Example usage
# data1 = torch.tensor([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0), (10.0, 11.0, 12.0)])
# data2 = torch.tensor([(2.0, 3.0, 4.0), (5.0, 6.0, 7.0), (8.0, 9.0, 10.0), (11.0, 12.0, 13.0)])

# # Call the function
# min_distances = match_points_and_get_distances(data1, data2)

