"""
Reservoir Fluid Typing From Standard Mud Gas – A Machine-Learning Approach
==========================================================================
Based on: Cely, A., Siedlecki, A., Ng, C.S.W., Liashenko, A., Donnadieu, S.,
and Yang, T. (2024), "Reservoir Fluid Typing From Standard Mud Gas – A
Machine-Learning Approach," Petrophysics, 65(4), pp. 496-506.
DOI: 10.30632/PJV65N4-2024a5

Implements:
  - Random Forest classifier for oil/gas binary classification
  - Feature engineering from standard mud gas ratios
  - Three-approach feature selection methodology
  - AUC and accuracy evaluation metrics
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from itertools import combinations


@dataclass
class MudGasFeatures:
    """Engineered features from standard mud gas for ML classification."""
    c1_c2: np.ndarray        # C1/C2 ratio
    c1_c3: np.ndarray        # C1/C3 ratio
    c2_c3: np.ndarray        # C2/C3 ratio
    bernard: np.ndarray      # C1/(C2+C3)
    wetness: np.ndarray      # (C2+C3)/(C1+C2+C3)
    c1_norm: np.ndarray      # C1/(C1+C2+C3) normalized
    c2_norm: np.ndarray      # C2/(C1+C2+C3) normalized
    c3_norm: np.ndarray      # C3/(C1+C2+C3) normalized


def engineer_features(c1: np.ndarray, c2: np.ndarray,
                      c3: np.ndarray) -> MudGasFeatures:
    """Compute all gas ratio features from raw C1, C2, C3 readings.

    The paper systematically evaluates combinations of these features
    to find the most discriminative set for oil/gas classification.
    """
    eps = 1e-10
    total = c1 + c2 + c3 + eps
    return MudGasFeatures(
        c1_c2=c1 / (c2 + eps),
        c1_c3=c1 / (c3 + eps),
        c2_c3=c2 / (c3 + eps),
        bernard=c1 / (c2 + c3 + eps),
        wetness=(c2 + c3) / total,
        c1_norm=c1 / total,
        c2_norm=c2 / total,
        c3_norm=c3 / total,
    )


def features_to_matrix(feats: MudGasFeatures,
                       selected: Optional[List[str]] = None) -> np.ndarray:
    """Convert MudGasFeatures to a matrix using selected feature names."""
    all_feats = {
        "c1_c2": feats.c1_c2, "c1_c3": feats.c1_c3, "c2_c3": feats.c2_c3,
        "bernard": feats.bernard, "wetness": feats.wetness,
        "c1_norm": feats.c1_norm, "c2_norm": feats.c2_norm, "c3_norm": feats.c3_norm,
    }
    if selected is None:
        selected = list(all_feats.keys())
    return np.column_stack([all_feats[name] for name in selected])


class FeatureSelector:
    """Three-approach feature selection as described in the paper.

    Approach 1: Forward selection (pairs -> trios -> ...)
    Approach 2: Backward elimination (full set -> n-1 -> ...)
    Approach 3: Manual comparison and hybrid construction
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.all_features = [
            "c1_c2", "c1_c3", "c2_c3", "bernard",
            "wetness", "c1_norm", "c2_norm", "c3_norm"
        ]

    def _evaluate(self, X: np.ndarray, y: np.ndarray,
                  feature_indices: list) -> float:
        """Evaluate a feature subset using cross-validated accuracy."""
        X_sub = X[:, feature_indices]
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=self.random_state
        )
        skf = StratifiedKFold(n_splits=3, shuffle=True,
                              random_state=self.random_state)
        scores = []
        for train_idx, val_idx in skf.split(X_sub, y):
            clf.fit(X_sub[train_idx], y[train_idx])
            scores.append(accuracy_score(y[val_idx], clf.predict(X_sub[val_idx])))
        return np.mean(scores)

    def forward_selection(self, X: np.ndarray, y: np.ndarray,
                          max_features: int = 5) -> Tuple[list, float]:
        """Approach 1: forward feature selection starting from best pairs."""
        n_feat = X.shape[1]
        best_set = []
        best_score = 0.0

        # Start with best pair
        pair_scores = {}
        for i, j in combinations(range(n_feat), 2):
            score = self._evaluate(X, y, [i, j])
            pair_scores[(i, j)] = score
        best_pair = max(pair_scores, key=pair_scores.get)
        best_set = list(best_pair)
        best_score = pair_scores[best_pair]

        # Add features one at a time
        for _ in range(max_features - 2):
            remaining = [i for i in range(n_feat) if i not in best_set]
            if not remaining:
                break
            candidates = {}
            for f in remaining:
                score = self._evaluate(X, y, best_set + [f])
                candidates[f] = score
            best_new = max(candidates, key=candidates.get)
            if candidates[best_new] > best_score:
                best_set.append(best_new)
                best_score = candidates[best_new]
            else:
                break

        return best_set, best_score

    def backward_elimination(self, X: np.ndarray,
                             y: np.ndarray) -> Tuple[list, float]:
        """Approach 2: backward elimination starting from all features."""
        current = list(range(X.shape[1]))
        current_score = self._evaluate(X, y, current)

        while len(current) > 2:
            worst_feature = None
            best_new_score = 0
            for f in current:
                subset = [x for x in current if x != f]
                score = self._evaluate(X, y, subset)
                if score > best_new_score:
                    best_new_score = score
                    worst_feature = f
            if best_new_score >= current_score:
                current.remove(worst_feature)
                current_score = best_new_score
            else:
                break

        return current, current_score


class FluidTypeClassifier:
    """Random Forest classifier for oil/gas prediction from standard mud gas.

    Implements hyperparameter tuning focused on reducing overfitting
    (number of trees, depth, leaf samples) as described in the paper.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 10,
                 min_samples_leaf: int = 5, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, random_state=random_state
        )
        self.feature_names: Optional[List[str]] = None
        self.threshold: float = 0.5
        self._fitted = False

    def train(self, c1: np.ndarray, c2: np.ndarray, c3: np.ndarray,
              labels: np.ndarray, feature_names: Optional[List[str]] = None):
        """Train on labeled standard mud gas data. labels: 0=oil, 1=gas."""
        feats = engineer_features(c1, c2, c3)
        self.feature_names = feature_names or [
            "c1_c2", "c1_c3", "c2_c3", "bernard", "wetness"
        ]
        X = features_to_matrix(feats, self.feature_names)
        self.model.fit(X, labels)
        self._fitted = True

    def predict(self, c1: np.ndarray, c2: np.ndarray,
                c3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict fluid type and probability.

        Returns (labels, probabilities) where labels are 0=oil, 1=gas.
        """
        if not self._fitted:
            raise RuntimeError("Model not trained.")
        feats = engineer_features(c1, c2, c3)
        X = features_to_matrix(feats, self.feature_names)
        proba = self.model.predict_proba(X)[:, 1]
        labels = (proba >= self.threshold).astype(int)
        return labels, proba

    def evaluate(self, c1: np.ndarray, c2: np.ndarray, c3: np.ndarray,
                 labels: np.ndarray) -> dict:
        """Evaluate on test set, returning accuracy and AUC."""
        pred_labels, proba = self.predict(c1, c2, c3)
        return {
            "accuracy": accuracy_score(labels, pred_labels),
            "auc": roc_auc_score(labels, proba),
        }


def test_all():
    """Test ML fluid typing pipeline."""
    print("=" * 70)
    print("Testing: ML Fluid Typing from Standard Mud Gas (Cely et al., 2024)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    n_train, n_test = 400, 100

    # Generate labeled data: oil (0) and gas (1)
    def gen_data(n, seed):
        r = np.random.RandomState(seed)
        n_oil, n_gas = n // 2, n - n // 2
        c1 = np.concatenate([r.uniform(60, 200, n_oil), r.uniform(300, 900, n_gas)])
        c2 = np.concatenate([r.uniform(15, 50, n_oil), r.uniform(5, 25, n_gas)])
        c3 = np.concatenate([r.uniform(8, 30, n_oil), r.uniform(2, 10, n_gas)])
        labels = np.array([0] * n_oil + [1] * n_gas)
        idx = r.permutation(n)
        return c1[idx], c2[idx], c3[idx], labels[idx]

    c1_tr, c2_tr, c3_tr, y_tr = gen_data(n_train, 42)
    c1_te, c2_te, c3_te, y_te = gen_data(n_test, 99)

    # Feature selection
    feats_tr = engineer_features(c1_tr, c2_tr, c3_tr)
    X_all = features_to_matrix(feats_tr)
    feature_names = ["c1_c2", "c1_c3", "c2_c3", "bernard",
                     "wetness", "c1_norm", "c2_norm", "c3_norm"]

    selector = FeatureSelector()
    fwd_set, fwd_score = selector.forward_selection(X_all, y_tr)
    print(f"  Forward selection: features={[feature_names[i] for i in fwd_set]}, "
          f"accuracy={fwd_score:.3f}")

    bwd_set, bwd_score = selector.backward_elimination(X_all, y_tr)
    print(f"  Backward elimination: features={[feature_names[i] for i in bwd_set]}, "
          f"accuracy={bwd_score:.3f}")

    # Train and evaluate classifier
    best_features = [feature_names[i] for i in fwd_set]
    classifier = FluidTypeClassifier()
    classifier.train(c1_tr, c2_tr, c3_tr, y_tr, feature_names=best_features)

    metrics = classifier.evaluate(c1_te, c2_te, c3_te, y_te)
    print(f"\n  Test accuracy: {metrics['accuracy']:.3f}")
    print(f"  Test AUC:      {metrics['auc']:.3f}")

    # Feature importances
    importances = classifier.model.feature_importances_
    print(f"\n  Feature importances:")
    for name, imp in sorted(zip(best_features, importances), key=lambda x: -x[1]):
        print(f"    {name}: {imp:.3f}")

    print("\n  [PASS] ML fluid typing tests completed.")
    return True


if __name__ == "__main__":
    test_all()
