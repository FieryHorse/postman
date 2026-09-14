# Postman

A from-scratch implementation of **Decision Trees and Random Forests** using NumPy, built to understand the underlying mechanics of tree-based machine learning rather than relying on scikit-learn's model implementations.

The project explores how decision trees choose splits using **Gini impurity**, how unrestricted trees can overfit, how regularisation controls tree complexity, and how **Random Forests** improve robustness through bootstrap sampling and random feature selection.

## Overview

This repository contains implementations of:

* 🌳 An unrestricted Decision Tree
* 🌱 A regularised Decision Tree
* 🌲 A Random Forest Classifier
* 📊 Demonstrations comparing the models
* 🔬 Experiments illustrating overfitting and regularisation

The models are implemented manually with **NumPy**, while `scikit-learn` is used only for generating synthetic datasets and, in one demonstration, providing a reference Decision Tree implementation.

---

## Project Structure

```text
postman/
│
├── my_code/
│   ├── __init__.py
│   ├── decision_tree.py
│   ├── regularised_tree.py
│   ├── random_forest.py
│   └── tree_vs_forest.py
│
├── demonstration/
│   ├── comparison_4.6.py
│   ├── overfitting_demo.py
│   └── regularised_tree_demo.py
│
├── README.md
├── WRITEUP.md
└── .gitignore
```

### `my_code/`

Contains the core implementations.

#### `decision_tree.py`

Implements an unrestricted decision tree.

Main components:

* `gini()` — calculates Gini impurity
* `candidate_thresholds()` — generates candidate split thresholds
* `best_split()` — searches for the split producing the largest Gini reduction
* `Node` — represents a tree node
* `majority()` — determines the majority class
* `build_unrestricted_tree()` — recursively constructs the tree
* `predict()` — generates predictions

The unrestricted tree continues splitting until a node contains samples belonging to only one class or no useful split can be found.

#### `regularised_tree.py`

Implements a constrained decision tree.

In addition to the splitting logic used by the unrestricted tree, it supports:

* `max_depth`
* `min_samples_split`
* Random feature subsets

These constraints prevent the tree from growing excessively complex and help reduce overfitting.

#### `random_forest.py`

Implements a Random Forest classifier from scratch.

Each tree is trained using:

1. **Bootstrap sampling** — randomly samples the training data with replacement.
2. **Random feature selection** — each split can consider only a subset of features.
3. **Multiple decision trees** — several trees are trained independently.
4. **Majority voting** — predictions from the individual trees are combined to produce the final classification.

The default configuration is:

```python
RandomForestClassifier(
    n_estimators=15,
    max_depth=3,
    min_samples_split=10,
    max_features="sqrt"
)
```

#### `tree_vs_forest.py`

Generates a synthetic classification dataset and compares the performance of a regularised decision tree against the implemented random forest.

---

## How the Decision Tree Works

The tree uses **Gini impurity** to evaluate potential splits.

For a node containing samples from different classes, Gini impurity is calculated as:

$$
G = 1 - \sum_{i=1}^{C} p_i^2
$$

where \(p_i\) is the proportion of samples belonging to class \(i\).

For every candidate feature and threshold, the implementation calculates the weighted impurity of the two resulting child nodes.

The best split is the one that produces the largest reduction in impurity:

$$
\text{Gain} = G_{\text{parent}} - G_{\text{split}}
$$

The process is repeated recursively until a stopping condition is reached.

---

## Regularisation

An unrestricted decision tree can continue splitting until it fits the training data extremely closely.

This can lead to **overfitting**:

```text
Training accuracy  → very high
Test accuracy      → significantly lower
```

The regularised implementation limits tree complexity using parameters such as:

### Maximum depth

```python
max_depth=3
```

Prevents the tree from growing beyond a specified depth.

### Minimum samples for splitting

```python
min_samples_split=10
```

Prevents nodes containing too few samples from being split further.

Together, these restrictions encourage the tree to learn broader patterns instead of memorising individual training examples.

---

## Random Forest

The Random Forest implementation combines multiple decision trees.

For every tree, the training data is sampled with replacement:

```python
indices = np.random.choice(
    n_samples,
    size=n_samples,
    replace=True
)
```

A random subset of features can also be selected for each split.

For example:

```python
max_features="sqrt"
```

uses approximately:

$$
\sqrt{N}
$$

features when there are \(N\) available features.

After all trees make their predictions, the forest chooses the class receiving the most votes.

This reduces the dependence on any single tree and generally produces a more robust classifier.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/FieryHorse/postman.git
cd postman
```

Install the required Python packages:

```bash
pip install numpy scikit-learn
```

### Requirements

* Python 3.x
* NumPy
* scikit-learn

The core tree and forest implementations rely primarily on NumPy.

---

## Running the Demonstrations

### 1. Unrestricted Tree — Overfitting

Run:

```bash
python demonstration/overfitting_demo.py
```

This creates a synthetic classification dataset, trains an unrestricted decision tree, and reports its training and test accuracy.

The experiment demonstrates how an unrestricted tree can fit the training data extremely closely while performing worse on unseen data.

---

### 2. Regularised Decision Tree

Run:

```bash
python demonstration/regularised_tree_demo.py
```

This trains a constrained decision tree and reports:

```text
Regularised Tree Train Accuracy: ...
Regularised Tree Test Accuracy: ...
```

The purpose is to observe how limiting tree complexity can improve generalisation.

---

### 3. Decision Tree vs Random Forest

Run:

```bash
python my_code/tree_vs_forest.py
```

This experiment creates a synthetic binary classification dataset and compares:

* Random Forest
* Regularised Decision Tree

The random forest uses:

```python
n_estimators=200
max_depth=4
min_samples_split=12
```

and `sqrt` feature selection.

---

### 4. Comparison with scikit-learn

Run:

```bash
python demonstration/comparison_4.6.py
```

This compares the behaviour of a scikit-learn decision tree with the corresponding self-implemented decision tree results.

The demonstration uses:

```python
DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    random_state=42
)
```

---

## Example Usage

### Decision Tree

```python
import numpy as np
from my_code.regularised_tree import build_tree, predict

X = np.array([
    [1, 2],
    [2, 1],
    [8, 9],
    [9, 8]
])

y = np.array([0, 0, 1, 1])

tree = build_tree(
    X,
    y,
    max_depth=3,
    min_samples_split=2
)

predictions = predict(tree, X)

print(predictions)
```

### Random Forest

```python
from my_code.random_forest import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    min_samples_split=10,
    max_features="sqrt"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

## Concepts Demonstrated

This project focuses on several important machine-learning concepts:

| Concept                     | Implementation                   |
| --------------------------- | -------------------------------- |
| Gini impurity               | `gini()`                         |
| Split selection             | `best_split()`                   |
| Recursive tree construction | `build_tree()`                   |
| Majority-class prediction   | `majority()`                     |
| Tree regularisation         | `max_depth`, `min_samples_split` |
| Bootstrap sampling          | Random Forest                    |
| Random feature selection    | Random Forest                    |
| Ensemble learning           | Majority voting                  |
| Overfitting                 | `overfitting_demo.py`            |
| Model comparison            | Demonstration scripts            |

---

## Design Philosophy

The purpose of this project is not to reproduce every feature of production-grade machine-learning libraries.

Instead, it focuses on making the fundamental algorithms understandable and explicit.

Rather than calling:

```python
DecisionTreeClassifier(...)
```

the implementation exposes the major steps involved in training a tree:

```text
Dataset
   │
   ▼
Calculate Gini impurity
   │
   ▼
Generate candidate splits
   │
   ▼
Evaluate every split
   │
   ▼
Choose the best split
   │
   ▼
Recursively build child nodes
   │
   ▼
Stop according to constraints
   │
   ▼
Predict using the resulting tree
```

The Random Forest then builds upon this tree implementation.

---

## Limitations

This is an educational implementation and intentionally has a smaller feature set than mature libraries such as scikit-learn.

For example, the current implementation does not provide the full range of functionality typically expected from a production machine-learning library, such as:

* Regression trees
* Probability estimates
* Missing-value handling
* Categorical feature handling
* Feature importance APIs
* Advanced split optimisation
* Model serialisation
* Extensive parameter validation

Candidate thresholds are also limited to a maximum number of values/quantiles to keep the split search manageable.

---

## Future Improvements

Possible extensions include:

* [ ] Add regression trees
* [ ] Add feature importance
* [ ] Add prediction probabilities
* [ ] Add configurable random seeds
* [ ] Improve threshold selection
* [ ] Add out-of-bag evaluation
* [ ] Add unit tests
* [ ] Add visualisation of decision trees
* [ ] Add performance benchmarks against scikit-learn
* [ ] Improve package/module imports
* [ ] Add type hints and documentation

---

## Learning Goals

This repository is intended as a practical exploration of:

**Decision Trees → Overfitting → Regularisation → Ensemble Learning → Random Forests**

The implementation provides a way to see how these ideas work internally rather than treating the algorithms as black boxes.

---

## License

See the repository for licensing information.
