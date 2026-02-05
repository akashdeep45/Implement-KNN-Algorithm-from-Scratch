from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np


@dataclass
class KDNode:
    point: np.ndarray
    label: int
    axis: int
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


class KDTree:
    """
    Simple k-d tree for k-NN queries in moderate dimensions.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same length")
        self.n_features = X.shape[1]
        self.root = self._build(X, y, depth=0)

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Optional[KDNode]:
        if X.shape[0] == 0:
            return None
        axis = depth % self.n_features
        idx = np.argsort(X[:, axis])
        X_sorted = X[idx]
        y_sorted = y[idx]
        median = X_sorted.shape[0] // 2
        return KDNode(
            point=X_sorted[median],
            label=int(y_sorted[median]),
            axis=axis,
            left=self._build(X_sorted[:median], y_sorted[:median], depth + 1),
            right=self._build(X_sorted[median + 1 :], y_sorted[median + 1 :], depth + 1),
        )

    def query(
        self, q: np.ndarray, k: int = 1, distance_func=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors of q.

        Returns (distances, labels) sorted by increasing distance.
        """
        if distance_func is None:
            from .distances import euclidean_distance as distance_func

        q = np.asarray(q, dtype=float)
        if q.ndim != 1 or q.shape[0] != self.n_features:
            raise ValueError("Query point must be 1D with same number of features")

        # max-heap of (negative_distance, label)
        best: List[Tuple[float, int]] = []

        def search(node: Optional[KDNode]):
            if node is None:
                return

            dist = distance_func(q, node.point)
            import heapq

            if len(best) < k:
                heapq.heappush(best, (-dist, node.label))
            else:
                if dist < -best[0][0]:
                    heapq.heapreplace(best, (-dist, node.label))

            axis = node.axis
            diff = q[axis] - node.point[axis]
            first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)

            search(first)

            # check if we need to explore the other branch
            if len(best) < k or abs(diff) < -best[0][0]:
                search(second)

        search(self.root)

        # convert heap to sorted arrays
        best_sorted = sorted(best, key=lambda x: -x[0])
        dists = np.array([-d for d, _ in best_sorted], dtype=float)
        labels = np.array([lbl for _, lbl in best_sorted], dtype=int)
        return dists, labels

