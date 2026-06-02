"""Unit tests for the post-processing clustering / clash-removal step."""
import numpy as np

from superwater.utils.cluster_centroid import find_centroids


def test_threshold_filters_everything_returns_none():
    coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    probs = np.array([0.1, 0.2])
    assert find_centroids(coords, probs, threshold=0.5) is None


def test_nearby_points_merge_far_points_kept():
    # p0,p1 are within cluster_distance (1.52) -> one centroid; p2 is far -> second centroid.
    coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [10.0, 0.0, 0.0]])
    probs = np.array([0.9, 0.8, 0.7])
    centroids = find_centroids(coords, probs, threshold=0.5,
                               cluster_distance=1.52, clash_distance=2.2)
    assert centroids is not None
    assert centroids.shape == (2, 3)
    # The far point must survive unchanged.
    assert np.any(np.all(np.isclose(centroids, [10.0, 0.0, 0.0]), axis=1))


def test_clash_removal_suppresses_lower_probability():
    # 2.0 A apart: not clustered (>1.52) but clashing (<2.2) -> keep higher-prob only.
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    probs = np.array([0.9, 0.6])
    centroids = find_centroids(coords, probs, threshold=0.5,
                               cluster_distance=1.52, clash_distance=2.2)
    assert centroids.shape == (1, 3)
    assert np.allclose(centroids[0], [0.0, 0.0, 0.0])


def test_min_pairwise_distance_respects_clash_distance():
    rng = np.random.default_rng(0)
    coords = rng.uniform(-15, 15, size=(200, 3))
    probs = rng.uniform(0.0, 1.0, size=200)
    clash = 2.2
    centroids = find_centroids(coords, probs, threshold=0.05, clash_distance=clash)
    if centroids is not None and len(centroids) > 1:
        from scipy.spatial.distance import pdist
        assert pdist(centroids).min() >= clash - 1e-6
