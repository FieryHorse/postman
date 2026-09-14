from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X, y = make_classification(
        n_samples=150,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        flip_y=0.15,  # Introduces 15% label noise
        random_state=42
    )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

sklearn_tree = DecisionTreeClassifier(max_depth=3, min_samples_split=10, random_state=42)
sklearn_tree.fit(X_train, y_train)
sklearn_train_preds = sklearn_tree.predict(X_train)
sklearn_test_preds = sklearn_tree.predict(X_test)

sklearn_acc = np.mean(sklearn_test_preds == y_test)

print(f"\nTrain Accuracy Sklearn: {np.mean(sklearn_train_preds == y_train)*100:.2f}%")
print(f"Test Accuracy Sklearn: {sklearn_acc * 100:.2f}%")

