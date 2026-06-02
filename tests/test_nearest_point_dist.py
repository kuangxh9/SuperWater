"""Unit test for the nearest-neighbour distance helper used to score samples."""
import numpy as np

from superwater.utils.nearest_point_dist import get_nearest_point_distances


def test_nearest_point_distances_and_indices():
    set1 = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    set2 = np.array([[0.0, 0.0, 1.0], [5.0, 0.0, 0.0]])
    dists, indices = get_nearest_point_distances(set1, set2)
    assert np.allclose(dists, [1.0, 0.0])
    assert indices.tolist() == [0, 1]
