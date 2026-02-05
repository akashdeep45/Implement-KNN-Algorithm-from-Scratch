import numpy as np

from src.kdtree import KDTree
from src.distances import euclidean_distance


def test_kdtree_single_point():
    X = np.array([[1.0, 2.0]])
    y = np.array([0])
    tree = KDTree(X, y)
    dists, labels = tree.query(np.array([1.0, 2.0]), k=1)
    assert dists.shape == (1,)
    assert labels.shape == (1,)
    assert dists[0] == 0.0
    assert labels[0] == 0


def test_kdtree_nearest_neighbor_matches_bruteforce():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = rng.integers(0, 3, size=50)
    tree = KDTree(X, y)

    q = rng.normal(size=3)
    dists_tree, labels_tree = tree.query(q, k=3, distance_func=euclidean_distance)

    # brute force
    dists_all = np.array([euclidean_distance(q, p) for p in X])
    idx = np.argsort(dists_all)[:3]
    dists_brute = dists_all[idx]
    labels_brute = y[idx]

    assert np.allclose(np.sort(dists_tree), np.sort(dists_brute))
    assert set(labels_tree.tolist()) == set(labels_brute.tolist())

