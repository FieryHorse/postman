from sklearn.datasets import make_classification
import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent))
from my_code.decision_tree import build_unrestricted_tree, predict

X, y = make_classification(
        n_samples=150,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        flip_y=0.15,  # Introduces 15% label noise
        random_state=42
    )

def train_test_split(X, y, test_size=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * (1 - test_size))
    train, test = idx[:cut], idx[cut:]
    return X[train], X[test], y[train], y[test]


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


X_train, X_test, y_train, y_test = train_test_split(X, y, seed=1)

unrestricted_tree = build_unrestricted_tree(X_train, y_train)

print(f"Unrestricted Tree Train Accuracy: {accuracy(y_train, predict(unrestricted_tree, X_train)) * 100:.2f}%")
print(f"Unrestricted Tree Test Accuracy:  {accuracy(y_test, predict(unrestricted_tree, X_test)) * 100:.2f}%")