import numpy as np

from src.knn import KNNClassifier
from src.evaluate import load_iris_dataset


def _simple_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, seed: int = 0):
    """Minimal train/test split using NumPy only."""
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_test = int(n_samples * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def test_knn_simple_perfect_fit():
    # simple linearly separable data
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array([0, 0, 1, 1])
    clf = KNNClassifier(k=1, metric="euclidean", use_kdtree=True).fit(X, y)
    acc = clf.score(X, y)
    assert acc == 1.0


def test_knn_iris_reasonable_accuracy():
    X, y = load_iris_dataset()
    X_train, X_test, y_train, y_test = _simple_train_test_split(X, y, test_size=0.3, seed=0)
    clf = KNNClassifier(k=5, metric="euclidean", use_kdtree=True).fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    # Should be much better than random (1/3)
    assert acc > 0.8


def test_knn_manhattan_metric():
    X = np.array([[0.0], [2.0], [5.0]])
    y = np.array([0, 1, 1])
    clf = KNNClassifier(k=1, metric="manhattan", use_kdtree=False).fit(X, y)
    pred = clf.predict(np.array([[1.1]]))
    assert pred.shape == (1,)
    assert pred[0] == 1

