from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple, List

import numpy as np
import pandas as pd

from .knn import KNNClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
IRIS_PATH = BASE_DIR / "iris" / "iris.data"


@dataclass
class EvalResult:
    dataset: str
    metric: str
    use_kdtree: bool
    k_neighbors: int
    n_splits: int
    mean_accuracy: float
    std_accuracy: float
    total_runtime_sec: float


def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Iris dataset from the local iris/iris.data file.

    The file is expected to be in the standard UCI Iris format with
    4 numeric features and a string class label.
    """
    df = pd.read_csv(
        IRIS_PATH,
        header=None,
        names=["sepal_length", "sepal_width", "petal_length", "petal_width", "label"],
    )
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy(
        dtype=float
    )
    # Map string labels to integers 0, 1, 2
    label_mapping = {name: idx for idx, name in enumerate(sorted(df["label"].unique()))}
    y = df["label"].map(label_mapping).to_numpy(dtype=int)
    return X, y


def _stratified_kfold_indices(
    y: np.ndarray, n_splits: int, random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Simple stratified k-fold splitter implemented with NumPy only.
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    # Collect indices per class
    class_indices = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        class_indices.append(idx)

    # Split each class indices into n_splits chunks
    class_chunks: List[List[np.ndarray]] = []
    for idx in class_indices:
        splits = np.array_split(idx, n_splits)
        class_chunks.append(splits)

    for fold in range(n_splits):
        test_idx_parts = [chunks[fold] for chunks in class_chunks]
        test_idx = np.concatenate(test_idx_parts)
        train_mask = np.ones_like(y, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        folds.append((train_idx, test_idx))

    return folds


def evaluate_knn(
    dataset: str = "iris",
    metric: Literal["euclidean", "manhattan"] = "euclidean",
    use_kdtree: bool = True,
    k_neighbors: int = 5,
    n_splits: int = 5,
    random_state: int = 42,
) -> EvalResult:
    if dataset != "iris":
        raise ValueError("Only the local 'iris' dataset is supported in this project.")

    X, y = load_iris_dataset()
    folds = _stratified_kfold_indices(y, n_splits=n_splits, random_state=random_state)

    accuracies: list[float] = []
    start = time.perf_counter()

    for train_idx, test_idx in folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = KNNClassifier(k=k_neighbors, metric=metric, use_kdtree=use_kdtree)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        accuracies.append(acc)

    total_runtime = time.perf_counter() - start
    accuracies_arr = np.array(accuracies, dtype=float)

    return EvalResult(
        dataset=dataset,
        metric=metric,
        use_kdtree=use_kdtree,
        k_neighbors=k_neighbors,
        n_splits=n_splits,
        mean_accuracy=float(accuracies_arr.mean()),
        std_accuracy=float(accuracies_arr.std()),
        total_runtime_sec=float(total_runtime),
    )


def run_default_experiments() -> pd.DataFrame:
    """
    Run a small set of experiments on the local Iris dataset.
    """
    configs = [
        ("iris", "euclidean", True),
        ("iris", "euclidean", False),
        ("iris", "manhattan", False),  # k-d tree off for non-Euclidean
    ]
    results = []
    for dataset, metric, use_kdtree in configs:
        res = evaluate_knn(
            dataset=dataset,
            metric=metric,
            use_kdtree=use_kdtree,
            k_neighbors=5,
            n_splits=5,
        )
        results.append(res.__dict__)
    return pd.DataFrame(results)


def main():
    df = run_default_experiments()
    print("k-NN Evaluation Results")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

