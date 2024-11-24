import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

def match_points_and_get_distances(data1, data2):
    dist_matrix = cdist(data1, data2)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    min_distances = np.array([dist_matrix[i, j] for i, j in zip(row_ind, col_ind)])
    return min_distances

