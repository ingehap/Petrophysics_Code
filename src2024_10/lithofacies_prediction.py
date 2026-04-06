#!/usr/bin/env python3
"""
High-Resolution Lithofacies Prediction Using Ensemble Classifiers
==================================================================
Based on: Satti et al. (2024), "Enhancing Reservoir Characterization
With High-Resolution Lithofacies Prediction Using Advanced Feature
Engineering and Ensemble Classifiers", Petrophysics, Vol. 65, No. 5,
pp. 813-834.

Implements:
- Petrophysical cutoff-based lithofacies definition (Table 2)
- Feature engineering from well logs (GR, LLD, RHOB)
- Extra Trees (ET) classifier
- XGBoost-style gradient boosting classifier (simplified)
- Confusion matrix and F1-score evaluation
- k-Fold and random-subsampling cross-validation

Reference: DOI:10.30632/PJV65N5-2024a9
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from collections import Counter


# -----------------------------------------------------------------------
# 1. Lithofacies Definition from Cutoffs
# -----------------------------------------------------------------------
def define_lithofacies(Vshl: np.ndarray, Sw: np.ndarray,
                       PHIE: np.ndarray,
                       Vshl_cutoff: float = 0.40,
                       Sw_cutoff: float = 0.60,
                       PHIE_cutoff: float = 0.06
                       ) -> np.ndarray:
    """
    Assign lithofacies labels based on petrophysical cutoffs (Table 2).

    Classes:
        0 = shale          (Vshl > cutoff)
        1 = wet sand        (Vshl <= cutoff AND Sw > Sw_cutoff)
        2 = gas sand        (Vshl <= cutoff AND Sw <= Sw_cutoff AND PHIE > cutoff)

    Parameters
    ----------
    Vshl : array-like  – volume of shale (fraction)
    Sw : array-like    – water saturation (fraction)
    PHIE : array-like  – effective porosity (fraction)

    Returns
    -------
    np.ndarray  – lithofacies labels (0, 1, or 2)
    """
    Vshl = np.asarray(Vshl)
    Sw = np.asarray(Sw)
    PHIE = np.asarray(PHIE)
    labels = np.zeros(len(Vshl), dtype=int)  # default: shale
    sand_mask = Vshl <= Vshl_cutoff
    labels[sand_mask & (Sw > Sw_cutoff)] = 1  # wet sand
    labels[sand_mask & (Sw <= Sw_cutoff) & (PHIE > PHIE_cutoff)] = 2  # gas sand
    return labels


# -----------------------------------------------------------------------
# 2. Feature Engineering
# -----------------------------------------------------------------------
def engineer_features(GR: np.ndarray, LLD: np.ndarray,
                      RHOB: np.ndarray) -> np.ndarray:
    """
    Build feature matrix from well-log curves with derived features.

    Derived features:
    - GR normalized
    - log10(LLD)
    - RHOB
    - GR gradient
    - LLD / RHOB ratio
    - Moving average (window=5) of each primary log

    Parameters
    ----------
    GR, LLD, RHOB : np.ndarray
        Well-log curves.

    Returns
    -------
    np.ndarray  shape (n_samples, n_features)
    """
    n = len(GR)
    GR = np.asarray(GR, dtype=float)
    LLD = np.asarray(LLD, dtype=float)
    RHOB = np.asarray(RHOB, dtype=float)

    # Primary features
    GR_norm = (GR - GR.min()) / (GR.max() - GR.min() + 1e-6)
    log_LLD = np.log10(np.maximum(LLD, 0.01))

    # Gradient features
    GR_grad = np.gradient(GR)
    RHOB_grad = np.gradient(RHOB)

    # Ratio
    LLD_RHOB = LLD / (RHOB + 1e-6)

    # Moving averages
    def moving_avg(x, w=5):
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode='same')

    GR_ma = moving_avg(GR)
    LLD_ma = moving_avg(np.log10(np.maximum(LLD, 0.01)))
    RHOB_ma = moving_avg(RHOB)

    return np.column_stack([
        GR_norm, log_LLD, RHOB,
        GR_grad, RHOB_grad, LLD_RHOB,
        GR_ma, LLD_ma, RHOB_ma
    ])


# -----------------------------------------------------------------------
# 3. Decision Tree (base for ensembles)
# -----------------------------------------------------------------------
class DecisionStump:
    """A simple single-split decision tree (depth=1) for classification."""

    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_class = 0
        self.right_class = 0

    def fit(self, X, y, feature_subset=None):
        n, p = X.shape
        if feature_subset is None:
            feature_subset = range(p)
        best_gini = float('inf')
        for j in feature_subset:
            thresholds = np.unique(X[:, j])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, j], np.linspace(5, 95, 20))
            for t in thresholds:
                left = y[X[:, j] <= t]
                right = y[X[:, j] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                gini = self._gini(left) * len(left) + self._gini(right) * len(right)
                gini /= n
                if gini < best_gini:
                    best_gini = gini
                    self.feature_idx = j
                    self.threshold = t
                    self.left_class = Counter(left).most_common(1)[0][0]
                    self.right_class = Counter(right).most_common(1)[0][0]

    def predict(self, X):
        preds = np.where(X[:, self.feature_idx] <= self.threshold,
                         self.left_class, self.right_class)
        return preds

    @staticmethod
    def _gini(y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)


# -----------------------------------------------------------------------
# 4. Extra Trees Classifier (simplified)
# -----------------------------------------------------------------------
class ExtraTreesClassifier:
    """
    Simplified Extra Trees (ET) classifier.
    Uses random splits and random feature subsets (Geurts et al., 2006).
    """

    def __init__(self, n_estimators: int = 50, max_features: str = 'sqrt',
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.rng = np.random.RandomState(random_state)
        self.trees: List[DecisionStump] = []

    def fit(self, X, y):
        n, p = X.shape
        if self.max_features == 'sqrt':
            k = max(1, int(np.sqrt(p)))
        else:
            k = p
        self.trees = []
        for _ in range(self.n_estimators):
            feat_idx = self.rng.choice(p, k, replace=False)
            idx = self.rng.choice(n, n, replace=True)
            tree = DecisionStump()
            tree.fit(X[idx], y[idx], feature_subset=feat_idx)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = np.array([t.predict(X) for t in self.trees])
        # Majority vote
        preds = np.array([
            Counter(all_preds[:, i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ])
        return preds


# -----------------------------------------------------------------------
# 5. Gradient Boosting Classifier (simplified XGB-like)
# -----------------------------------------------------------------------
class SimpleGBClassifier:
    """
    Simplified gradient boosting classifier (XGB-style).
    Binary-class implementation extended to multi-class via OVR.
    """

    def __init__(self, n_estimators: int = 30, learning_rate: float = 0.3,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.rng = np.random.RandomState(random_state)
        self.classes_ = None
        self.models_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            y_bin = (y == c).astype(float)
            pred = np.full(len(y), y_bin.mean())
            trees = []
            for _ in range(self.n_estimators):
                residual = y_bin - self._sigmoid(pred)
                tree = DecisionStump()
                # Convert residual to class labels for stump
                r_labels = (residual > 0).astype(int)
                tree.fit(X, r_labels)
                update = tree.predict(X).astype(float) * 2 - 1
                pred += self.lr * update
                trees.append(tree)
            self.models_[c] = (trees, y_bin.mean())

    def predict(self, X):
        scores = {}
        for c in self.classes_:
            trees, base = self.models_[c]
            pred = np.full(X.shape[0], base)
            for tree in trees:
                update = tree.predict(X).astype(float) * 2 - 1
                pred += self.lr * update
            scores[c] = pred
        score_matrix = np.column_stack([scores[c] for c in self.classes_])
        return self.classes_[np.argmax(score_matrix, axis=1)]

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# -----------------------------------------------------------------------
# 6. Evaluation Metrics
# -----------------------------------------------------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     classes: Optional[np.ndarray] = None
                     ) -> np.ndarray:
    """Compute confusion matrix."""
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    return cm


def f1_score_per_class(y_true: np.ndarray, y_pred: np.ndarray
                       ) -> Dict[int, float]:
    """Compute F1 score for each class."""
    classes = np.unique(y_true)
    result = {}
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        result[int(c)] = f1
    return result


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def kfold_cross_validation(X: np.ndarray, y: np.ndarray,
                           clf_class, k: int = 5, **clf_kwargs
                           ) -> List[float]:
    """k-Fold cross-validation returning accuracy per fold."""
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k
    accuracies = []

    for i in range(k):
        test_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size],
                                    indices[(i + 1) * fold_size:]])
        clf = clf_class(**clf_kwargs)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        accuracies.append(accuracy(y[test_idx], y_pred))

    return accuracies


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Lithofacies Prediction Module Demo ===\n")
    np.random.seed(42)

    # Synthetic well-log data for 6 wells
    n = 600
    GR = np.random.uniform(15, 150, n)
    LLD = 10 ** np.random.uniform(0, 3, n)
    RHOB = np.random.uniform(2.0, 2.75, n)
    Vshl = np.clip(GR / 150.0, 0, 1)
    Sw = np.clip(0.3 + 0.5 * Vshl + np.random.normal(0, 0.1, n), 0, 1)
    PHIE = np.clip(0.25 - 0.2 * Vshl + np.random.normal(0, 0.03, n), 0, 0.4)

    # Define facies from cutoffs
    y = define_lithofacies(Vshl, Sw, PHIE)
    print(f"Facies distribution: {dict(Counter(y))}")
    print(f"  0=shale, 1=wet sand, 2=gas sand")

    # Feature engineering
    X = engineer_features(GR, LLD, RHOB)
    print(f"Feature matrix shape: {X.shape}")

    # Train/test split
    X_train, X_test = X[:480], X[480:]
    y_train, y_test = y[:480], y[480:]

    # Extra Trees
    et = ExtraTreesClassifier(n_estimators=50)
    et.fit(X_train, y_train)
    y_pred_et = et.predict(X_test)
    acc_et = accuracy(y_test, y_pred_et)
    f1_et = f1_score_per_class(y_test, y_pred_et)
    print(f"\nExtra Trees accuracy: {acc_et:.3f}")
    print(f"  F1 per class: {f1_et}")

    # Gradient Boosting
    gb = SimpleGBClassifier(n_estimators=30)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    acc_gb = accuracy(y_test, y_pred_gb)
    f1_gb = f1_score_per_class(y_test, y_pred_gb)
    print(f"\nGradient Boosting accuracy: {acc_gb:.3f}")
    print(f"  F1 per class: {f1_gb}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_et)
    print(f"\nConfusion matrix (ET):\n{cm}")

    # Cross-validation
    cv_acc = kfold_cross_validation(X, y, ExtraTreesClassifier,
                                    k=5, n_estimators=30)
    print(f"\n5-fold CV accuracies (ET): {[f'{a:.3f}' for a in cv_acc]}")
    print(f"Mean accuracy: {np.mean(cv_acc):.3f}")
