from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .distances import DISTANCE_FUNCS
from .kdtree import KDTree


class KNNClassifier:
    """
    Basic k-NN classifier with optional k-d tree acceleration.
    """

    def __init__(
        self,
        k: int = 5,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        use_kdtree: bool = True,
    ):
        if k <= 0:
            raise ValueError("k must be positive")
        if metric not in DISTANCE_FUNCS:
            raise ValueError(f"Unsupported metric: {metric}")
        self.k = k
        self.metric = metric
        self.use_kdtree = use_kdtree
        self._tree: Optional[KDTree] = None
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same length")
        self._X = X
        self._y = y
        if self.use_kdtree and self.metric == "euclidean":
            # Standard k-d tree relies on L2 geometry
            self._tree = KDTree(X, y)
        else:
            self._tree = None
        return self

    def _predict_point(self, x: np.ndarray) -> int:
        if self._X is None or self._y is None:
            raise RuntimeError("Classifier not fitted")

        dist_func = DISTANCE_FUNCS[self.metric]
        k = min(self.k, self._X.shape[0])

        if self._tree is not None:
            dists, labels = self._tree.query(x, k=k, distance_func=dist_func)
        else:
            # brute-force
            dists = np.array([dist_func(x, p) for p in self._X], dtype=float)
            idx = np.argsort(dists)[:k]
            dists = dists[idx]
            labels = self._y[idx]

        # majority vote (ties resolved by smallest distance)
        unique_labels = np.unique(labels)
        best_label = None
        best_count = -1
        best_min_dist = float("inf")
        for label in unique_labels:
            mask = labels == label
            count = int(np.sum(mask))
            min_dist = float(np.min(dists[mask]))
            if count > best_count or (count == best_count and min_dist < best_min_dist):
                best_label = label
                best_count = count
                best_min_dist = min_dist
        return int(best_label)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return np.array([self._predict_point(X)], dtype=int)
        preds = [self._predict_point(row) for row in X]
        return np.asarray(preds, dtype=int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y)
        y_pred = self.predict(X)
        if y_pred.shape != y.shape:
            raise ValueError("Predicted and true labels shape mismatch")
        return float(np.mean(y_pred == y))

