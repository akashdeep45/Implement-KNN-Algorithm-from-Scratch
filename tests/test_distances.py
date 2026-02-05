import numpy as np

from src.distances import euclidean_distance, manhattan_distance


def test_euclidean_basic():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert euclidean_distance(a, b) == 5.0


def test_manhattan_basic():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert manhattan_distance(a, b) == 7.0


def test_distance_shape_mismatch():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    for func in (euclidean_distance, manhattan_distance):
        try:
            func(a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for shape mismatch")

