import numpy as np
from random_forest import RandomForestClassifier
from regularised_tree import build_tree, predict
from sklearn.datasets import make_classification

#New Dataset
X, y = make_classification(
    n_samples=200,      # Total data points
    n_features=8,       # Total features per sample
    n_informative=4,   # Features carrying actual signal
    n_redundant=2,     # Linear combinations of informative features
    n_classes=2,       # Binary classification (0 or 1)
    flip_y=0.1,        # 10% label noise to simulate real-world data
    random_state=42    # Fixed seed for reproducible results
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

model = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_split=12)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_acc = accuracy(y_train, y_train_pred)
test_acc = accuracy(y_test, y_test_pred)
print(f"Random forest train accuracy: {train_acc*100:.2f}%")
print(f"Random forest test accuracy: {test_acc*100:.2f}%")

tree = build_tree(X_train, y_train)

print(f"Regularised Tree Train Accuracy: {accuracy(y_train, predict(tree, X_train)) * 100:.2f}%")
print(f"Regularised Tree Test Accuracy:  {accuracy(y_test, predict(tree, X_test)) * 100:.2f}%")