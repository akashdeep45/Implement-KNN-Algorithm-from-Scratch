# Implement KNN Algorithm from Scratch

A comprehensive implementation of the k-Nearest Neighbors (k-NN) algorithm built from scratch in Python, featuring KD-Tree optimization for efficient nearest neighbor search.

## 🎯 Overview

This project implements a complete k-NN classifier without relying on scikit-learn's implementation. It includes:

- **Custom k-NN Classifier** with configurable k and distance metrics
- **KD-Tree Implementation** for O(log n) nearest neighbor queries
- **Multiple Distance Metrics** (Euclidean, Manhattan)
- **Comprehensive Test Suite** with pytest
- **Performance Evaluation** on the Iris dataset

## 📁 Project Structure

```
project/
├── src/
│   ├── knn.py           # Main KNN classifier implementation
│   ├── kdtree.py        # KD-Tree data structure for fast queries
│   ├── distances.py     # Distance metric functions
│   ├── evaluate.py      # Model evaluation utilities
│   └── report.py        # Performance reporting
├── tests/
│   ├── test_knn.py      # KNN classifier tests
│   ├── test_kdtree.py   # KD-Tree tests
│   └── test_distances.py # Distance metrics tests
├── iris/
│   └── iris.data        # Iris dataset
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## 🚀 Features

### 1. KNNClassifier
- Configurable number of neighbors (k)
- Support for multiple distance metrics
- Optional KD-Tree acceleration
- Scikit-learn compatible API (fit, predict, score)

### 2. KD-Tree Optimization
- Efficient nearest neighbor search
- Automatic tree construction
- Handles multi-dimensional data
- Significant speedup for large datasets

### 3. Distance Metrics
- **Euclidean Distance**: Standard L2 norm
- **Manhattan Distance**: L1 norm (city block distance)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Implement-KNN-Algorithm-from-Scratch.git
cd Implement-KNN-Algorithm-from-Scratch
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Example

```python
from src.knn import KNNClassifier
import numpy as np

# Create sample data
X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Initialize and train the classifier
knn = KNNClassifier(k=3, metric='euclidean', use_kdtree=True)
knn.fit(X_train, y_train)

# Make predictions
X_test = np.array([[2, 2], [7, 7]])
predictions = knn.predict(X_test)
print(predictions)  # Output: [0 1]

# Calculate accuracy
accuracy = knn.score(X_test, np.array([0, 1]))
print(f"Accuracy: {accuracy:.2f}")  # Output: Accuracy: 1.00
```

### Iris Dataset Example

```python
from src.knn import KNNClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Load Iris dataset
data = pd.read_csv('iris/iris.data', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate
knn = KNNClassifier(k=5, metric='euclidean', use_kdtree=True)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2%}")
```

## 🧪 Running Tests

Run the complete test suite:

```bash
pytest tests/ -v
```

Run with coverage report:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## 📊 Performance Evaluation

Evaluate the classifier on the Iris dataset:

```bash
python evaluate.py
```

This will output:
- Training and test accuracy
- Confusion matrix
- Classification metrics (precision, recall, F1-score)
- Performance comparison with/without KD-Tree

## 🔬 Implementation Details

### KNN Classifier (`src/knn.py`)
- Implements majority voting for classification
- Handles tie-breaking using minimum distance
- Supports both brute-force and KD-Tree search
- Input validation and error handling

### KD-Tree (`src/kdtree.py`)
- Binary space partitioning tree
- Recursive construction with median splitting
- Efficient k-nearest neighbor queries
- Works optimally with Euclidean distance

### Distance Functions (`src/distances.py`)
- Vectorized implementations using NumPy
- Extensible design for adding new metrics
- Optimized for performance

## 📈 Results

On the Iris dataset (train/test split: 70/30):

| Configuration | Accuracy |
|--------------|----------|
| k=5, Euclidean, KD-Tree | ~96-98% |
| k=5, Manhattan, Brute-force | ~94-97% |
| k=3, Euclidean, KD-Tree | ~95-97% |

*Results may vary based on random seed*

## 🛠️ Dependencies

- `numpy>=1.26`: Numerical computing
- `pandas>=2.1`: Data manipulation
- `pytest>=8.0`: Testing framework
- `pytest-cov>=4.1`: Code coverage
- `tabulate>=0.9`: Pretty-print tables
- `reportlab>=4.0`: PDF report generation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Created as part of a machine learning implementation project.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## ⭐ Acknowledgments

- Iris dataset from UCI Machine Learning Repository
- Inspired by scikit-learn's KNN implementation
- KD-Tree algorithm based on classic computational geometry texts
