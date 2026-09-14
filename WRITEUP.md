# Write-Up — Decision Trees and Random Forests

## 4.1 — Build a decision tree using an impurity measure

I implemented a **classification decision tree from scratch** in `my_code/decision_tree.py` and `my_code/regularised_tree.py`. The implementation does **not** use `sklearn.tree` internally.

The tree uses **Gini impurity** as its split criterion. For a node containing classes with proportions \(p_1, p_2, \ldots, p_C\), the impurity is:

\[
Gini = 1 - \sum_{i=1}^{C} p_i^2
\]

For every candidate feature and threshold, the code divides the samples into left and right child nodes and computes the weighted impurity after the split:

\[
Gini_{split} = \frac{n_L}{n}Gini_L + \frac{n_R}{n}Gini_R
\]

The split with the largest impurity reduction is selected. Equivalently, the code maximises:

\[
Gain = Gini_{parent} - Gini_{split}
\]

The tree is represented recursively using a `Node` object. Each internal node stores the feature, threshold, and child nodes, while each leaf stores the majority-class prediction.

### Classification only

This project implements **classification only**. Basic regression was left out because the goal of this project was to understand the classification version of decision trees and then reuse that implementation inside a random forest. A regression tree would require a different impurity/error criterion, such as mean squared error (MSE), and a corresponding continuous-valued prediction rule at each leaf.

The current implementation therefore satisfies the classification requirement without introducing a second algorithm that was not needed for the experiments.

---

## 4.2 — Demonstrate overfitting with a nearly unrestricted tree

The file `demonstration/overfitting_demo.py` creates a noisy binary classification dataset with:

- 150 samples
- 10 features
- 5 informative features
- 2 redundant features
- 15% label noise

The data is split into training and test sets, and an **unrestricted decision tree** is trained using `build_unrestricted_tree()`.

The measured result is:

| Model | Training Accuracy | Test Accuracy |
|---|---:|---:|
| Unrestricted tree | **100.00%** | **68.42%** |

The gap of more than 31 percentage points is clear evidence of **overfitting**. The tree has enough flexibility to fit the training data almost perfectly, including noise and sample-specific patterns, but this does not transfer well to unseen data.

This behaviour is exactly what is expected from a highly flexible decision tree: increasing tree complexity can decrease training error while increasing the generalisation error.

---

## 4.3 — Add regularisation / stopping conditions

The regularised implementation is in `my_code/regularised_tree.py`.

It uses two explicit stopping conditions:

1. **Maximum tree depth** via `max_depth`.
2. **Minimum number of samples required for a split** via `min_samples_split`.

For the demonstration, the defaults are:

```python
max_depth=3
min_samples_split=10
```

The recursion stops when the current depth reaches the maximum, when too few samples remain to split, or when all samples in the node belong to the same class.

These constraints reduce the amount of memorisation the tree can perform.

---

## 4.4 — Show whether regularisation reduces overfitting

The same type of noisy dataset is used in `demonstration/regularised_tree_demo.py`.

The measured result is:

| Model | Training Accuracy | Test Accuracy |
|---|---:|---:|
| Unrestricted tree | **100.00%** | **68.42%** |
| Regularised tree | **85.71%** | **78.95%** |

Regularisation clearly reduces overfitting:

- Training accuracy falls from **100.00% to 85.71%**.
- Test accuracy rises from **68.42% to 78.95%**.
- The train/test gap becomes much smaller.

This is the intended bias-variance trade-off. The regularised tree is deliberately prevented from fitting every training example, but this loss of training performance improves its ability to generalise.

A later run of the repository's `tree_vs_forest.py` demonstration on its 200-sample dataset also produced a regularised-tree result of **82.67% training accuracy and 78.00% test accuracy**, showing the same general pattern on a different split/configuration.

---

## 4.5 — Build a random forest using the custom decision tree

The random forest is implemented in `my_code/random_forest.py` and reuses the custom tree implementation rather than `sklearn.tree`.

Two sources of randomness are introduced for every tree.

### 1. Bootstrap sampling

For each tree, the implementation samples the training set **with replacement**:

```python
indices = np.random.choice(
    n_samples,
    size=n_samples,
    replace=True
)
```

This means different trees are trained on different bootstrap samples of the original training set. Some original examples can appear multiple times in one bootstrap sample, while others may be absent from that particular sample.

### 2. Random feature subsets

At each tree node, the implementation can select a random subset of features. With:

```python
max_features="sqrt"
```

the number of candidate features is approximately:

\[
\sqrt{d}
\]

where \(d\) is the total number of features.

The forest then trains many independently randomised trees and combines their predictions using **majority voting**.

In the main demonstration, the forest uses:

```python
n_estimators=200
max_depth=4
min_samples_split=12
max_features="sqrt"
```

### Why use both kinds of randomness?

If every tree saw exactly the same training examples and considered exactly the same features, the trees would tend to become highly correlated and the ensemble would gain less from averaging them.

Bootstrap sampling changes the training data seen by each tree, while random feature subsets change the information available at each split. These two mechanisms reduce correlation between trees, which is important for variance reduction when their predictions are averaged or voted together.

---

## 4.6 — Compare the custom tree with `sklearn.tree.DecisionTreeClassifier`

The custom tree was compared with scikit-learn's `DecisionTreeClassifier` using a small noisy classification dataset.

The comparison was performed with matching high-level settings:

- Gini impurity
- `max_depth=3`
- `min_samples_split=10`
- the same training/test split
- the same input dataset

Using a fixed split for a reproducible comparison, the measured results were:

| Model | Training Accuracy | Test Accuracy |
|---|---:|---:|
| Custom tree | **83.04%** | **68.42%** |
| scikit-learn tree | **86.61%** | **81.58%** |

The custom implementation does not match scikit-learn exactly, which is expected. The two implementations use the same broad tree-learning idea and matching depth/split constraints, but the custom code uses its own threshold-candidate search. In particular, the custom implementation evaluates unique feature values (or a limited set of quantiles when there are many values), whereas scikit-learn uses its own optimised split-search procedure.

Therefore, the comparison shows that the custom algorithm is functionally similar but **not an exact reimplementation of scikit-learn's CART training procedure**.

> **Implementation note:** the repository's current `demonstration/comparison_4.6.py` prints fixed custom-tree accuracy values rather than calculating them from the tree at runtime. The numerical comparison reported above was therefore verified separately by actually fitting the custom tree and computing its predictions on the same split.

---

## 4.7 — Compare a single decision tree with the random forest

For this experiment I used the dataset/configuration from `my_code/tree_vs_forest.py`:

- 200 samples
- 8 features
- 4 informative features
- 2 redundant features
- 10% label noise
- fixed train/test split generated with seed `1`

To make the model-capacity comparison fair, the single-tree reference used the same depth and minimum-split settings as the forest:

```python
max_depth=4
min_samples_split=12
```

The single tree achieved:

- **88.67% training accuracy**
- **82.00% test accuracy**

Across 10 independently seeded runs, the 200-tree random forest achieved:

- **90.47% mean training accuracy**
- **84.60% mean test accuracy**

The forest therefore gave a modest improvement in average test accuracy on this dataset.

| Model | Training Accuracy | Test Accuracy |
|---|---:|---:|
| Single regularised tree | 88.67% | 82.00% |
| Random forest (mean of 10 runs) | 90.47% | **84.60%** |

The forest also showed relatively little run-to-run variation in this experiment:

- Test accuracy range: **82% to 86%**
- Test-accuracy standard deviation: **1.35 percentage points**

This indicates that the ensemble is reasonably stable despite the randomness used to construct its trees.

For reference, running the repository's `tree_vs_forest.py` directly also produced a strong forest result on its fixed split (for example, **90.00% train / 88.00% test** in one run), compared with the script's single-tree result of **82.67% train / 78.00% test**. The exact numbers vary because the forest intentionally contains random sampling and random feature selection.

---

## 4.8 — Why bagging reduces variance and why the randomness helps

### Why bagging reduces variance

A single decision tree can be unstable: small changes in the training data can cause a different split to be chosen high in the tree, which can then change many later splits. This makes individual trees relatively high-variance models.

Bagging (bootstrap aggregating) addresses this by fitting many trees to different bootstrap samples and then combining their predictions.

Suppose each tree has prediction variance \(\sigma^2\), and the predictions are not perfectly correlated. Averaging many such predictions reduces the variance of the ensemble. In the idealised independent case, averaging \(B\) trees gives a variance roughly proportional to:

\[
\frac{\sigma^2}{B}
\]

In practice, the trees are not fully independent, so the reduction is not exactly \(1/B\). Nevertheless, lowering the correlation between trees makes the ensemble substantially more stable than an individual tree.

### What randomness was introduced?

The forest in this project introduces two kinds of randomness:

**Bootstrap randomness:** every tree receives a different sample of the training data, drawn with replacement.

**Feature-selection randomness:** at each split, a random subset of the available features is considered. With `max_features="sqrt"`, only a fraction of all features is examined at each split.

### Why does this help?

These mechanisms make the trees less correlated. For example, if one feature is extremely strong, a conventional tree may repeatedly choose it and many trees in an ensemble would become nearly identical. Random feature selection prevents every tree from relying on exactly the same subset of features.

Bootstrap sampling also exposes each tree to a different version of the training set. Some examples are repeated, some are omitted, and the resulting trees therefore make different errors.

The forest can then use majority voting to keep the patterns that are consistent across many trees while reducing the influence of errors made by individual trees.

---

## Overall Findings

The experiments support the main ideas behind decision trees and random forests:

1. **An unrestricted decision tree can overfit badly**, reaching 100% training accuracy while performing much worse on unseen data.
2. **Regularisation reduces overfitting** by limiting tree depth and requiring enough samples before a split is allowed.
3. **A custom Gini-based tree can reproduce the essential decision-tree learning process without using scikit-learn's tree implementation.**
4. **A random forest can be constructed from the custom tree** by adding bootstrap sampling, random feature subsets, and majority voting.
5. **Bagging and feature randomisation reduce the dependence between trees**, lowering ensemble variance and improving stability.
6. On the tested noisy classification dataset, the random forest gave **slightly better average test accuracy than a single regularised tree** and showed only modest variation across repeated runs.

Overall, the implementation demonstrates the progression:

\[
\text{Decision Tree}
\rightarrow
\text{Overfitting}
\rightarrow
\text{Regularisation}
\rightarrow
\text{Bagging + Random Features}
\rightarrow
\text{Random Forest}
\]

This project intentionally focuses on **classification** so that the core learning process and the effect of ensemble methods can be understood clearly without adding the extra machinery required for regression trees.
