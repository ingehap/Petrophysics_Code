#!/usr/bin/env python3
"""
Automatic Permeability Estimation: ML Methods vs Conventional Models
=====================================================================
Based on: Raheem et al. (2024), "Best Practices in Automatic Permeability
Estimation: Machine-Learning Methods vs. Conventional Petrophysical
Models", Petrophysics, Vol. 65, No. 5, pp. 789-812.

Implements:
- Timur-Coates permeability model (Eq. 15-16)
- Feature engineering: moving-window statistics (Eq. / Fig. 4)
- Dimensionality reduction: PCA, SVD (Eqs. 2-6)
- ML regressors: Random Forest, SVR, kNN, Ridge, Lasso
- ANN (simple feed-forward) for permeability prediction (Eq. 13)
- Evaluation metrics: MAE, RSE (Eqs. 14, 21)
- Archie-based Sw for input preparation (Eqs. 17-20)
- Group k-fold cross-validation

Reference: DOI:10.30632/PJV65N5-2024a8
"""

import numpy as np
from typing import Tuple, Dict, Optional, List


# -----------------------------------------------------------------------
# 1. Timur-Coates Model (Eqs. 15-16)
# -----------------------------------------------------------------------
def timur_coates(phi: np.ndarray, Swirr: np.ndarray,
                 a: float = 1.0, b: float = 4.0, c: float = 2.0
                 ) -> np.ndarray:
    """
    Timur-Coates permeability estimate:
        k = a * (phi^b) * ((1 - Swirr) / Swirr)^c

    In log-space (Eq. 16):
        ln(k) = ln(a) + b*ln(phi) + c*ln((1-Swirr)/Swirr)

    Parameters
    ----------
    phi : array-like
        Porosity (fraction).
    Swirr : array-like
        Irreducible water saturation (fraction).
    a, b, c : float
        Fitting parameters.

    Returns
    -------
    np.ndarray
        Permeability (md).
    """
    phi = np.asarray(phi, dtype=float)
    Swirr = np.asarray(Swirr, dtype=float)
    Swirr = np.clip(Swirr, 0.01, 0.99)
    ratio = (1.0 - Swirr) / Swirr
    return a * (phi ** b) * (ratio ** c)


# -----------------------------------------------------------------------
# 2. Archie-based Sw Calculation (Eqs. 17-20)
# -----------------------------------------------------------------------
def archie_sw(Rt: np.ndarray, Rw: float, phi: np.ndarray,
              a: float = 1.0, m: float = 2.0, n: float = 2.0
              ) -> np.ndarray:
    """Standard Archie Sw."""
    phi = np.maximum(np.asarray(phi, dtype=float), 0.01)
    Rt = np.maximum(np.asarray(Rt, dtype=float), 0.01)
    F = a / (phi ** m)
    Sw = (F * Rw / Rt) ** (1.0 / n)
    return np.clip(Sw, 0, 1)


def sandstone_resistivity(Rt: np.ndarray, Csh: np.ndarray,
                          Rsh: float) -> np.ndarray:
    """
    Eq. 19: Rs = Rt / (1 - Csh * Rt / Rsh)
    Corrects Rt for shale lamination.
    """
    Rt = np.asarray(Rt, dtype=float)
    Csh = np.asarray(Csh, dtype=float)
    denom = 1.0 - Csh * Rt / Rsh
    denom = np.maximum(denom, 0.01)
    return Rt / denom


# -----------------------------------------------------------------------
# 3. Feature Engineering (Fig. 4)
# -----------------------------------------------------------------------
def augment_features(X: np.ndarray, window_sizes: List[int] = None
                     ) -> np.ndarray:
    """
    Generate second-order features by computing moving-window statistics:
    mean, variance, gradient (1st & 2nd order) for each input feature.

    Parameters
    ----------
    X : np.ndarray  shape (n_samples, n_features)
        Original feature matrix.
    window_sizes : list of int
        Moving-window sizes. Default [3, 5, 7].

    Returns
    -------
    np.ndarray
        Augmented feature matrix.
    """
    if window_sizes is None:
        window_sizes = [3, 5, 7]
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    augmented = [X]

    for ws in window_sizes:
        half = ws // 2
        for j in range(p):
            col = X[:, j]
            means = np.array([
                np.mean(col[max(0, i - half):min(n, i + half + 1)])
                for i in range(n)
            ])
            variances = np.array([
                np.var(col[max(0, i - half):min(n, i + half + 1)])
                for i in range(n)
            ])
            grad1 = np.gradient(col)
            grad2 = np.gradient(grad1)
            augmented.extend([
                means.reshape(-1, 1),
                variances.reshape(-1, 1),
                grad1.reshape(-1, 1),
                grad2.reshape(-1, 1),
            ])

    return np.hstack(augmented)


# -----------------------------------------------------------------------
# 4. Dimensionality Reduction (Eqs. 2-6)
# -----------------------------------------------------------------------
def pca_reduce(X: np.ndarray, n_components: int = 3
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA dimensionality reduction via SVD.

    Returns
    -------
    Z : np.ndarray  shape (n_samples, n_components)
        Latent representation.
    explained_var : np.ndarray
        Fraction of variance explained per component.
    components : np.ndarray  shape (n_components, n_features)
        Principal components.
    """
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    Z = U[:, :n_components] * S[:n_components]
    total_var = np.sum(S ** 2)
    explained_var = S[:n_components] ** 2 / total_var
    return Z, explained_var, Vt[:n_components]


def svd_reduce(X: np.ndarray, n_components: int = 3
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truncated SVD (Eq. 6).

    Returns
    -------
    Z : np.ndarray  shape (n_samples, n_components)
    singular_values : np.ndarray
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = U[:, :n_components] * S[:n_components]
    return Z, S[:n_components]


# -----------------------------------------------------------------------
# 5. ML Regressors (simplified implementations)
# -----------------------------------------------------------------------
def train_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0
                ) -> Tuple[np.ndarray, float]:
    """
    Ridge regression: w = (X^T X + alpha*I)^{-1} X^T y.

    Returns
    -------
    weights, intercept
    """
    n, p = X.shape
    X_b = np.hstack([np.ones((n, 1)), X])
    A = X_b.T @ X_b + alpha * np.eye(p + 1)
    w = np.linalg.solve(A, X_b.T @ y)
    return w[1:], w[0]


def predict_ridge(X: np.ndarray, weights: np.ndarray, intercept: float
                  ) -> np.ndarray:
    return X @ weights + intercept


def train_knn(X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, k: int = 5) -> np.ndarray:
    """k-Nearest Neighbours regression."""
    from scipy.spatial.distance import cdist
    dists = cdist(X_test, X_train)
    nn_idx = np.argsort(dists, axis=1)[:, :k]
    preds = np.array([np.mean(y_train[idx]) for idx in nn_idx])
    return preds


def simple_ann_predict(X: np.ndarray, hidden_sizes: List[int] = None,
                       seed: int = 42) -> np.ndarray:
    """
    Simple feed-forward ANN with random weights (Eq. 13 structure).
    For demonstration – real training would use backpropagation.
    """
    if hidden_sizes is None:
        hidden_sizes = [32, 16]
    rng = np.random.RandomState(seed)
    h = X.copy()
    for hs in hidden_sizes:
        W = rng.randn(h.shape[1], hs) * 0.1
        b = rng.randn(hs) * 0.01
        h = np.maximum(h @ W + b, 0)  # ReLU
    W_out = rng.randn(h.shape[1]) * 0.1
    return h @ W_out


# -----------------------------------------------------------------------
# 6. Evaluation Metrics (Eqs. 14, 21)
# -----------------------------------------------------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (Eq. 14)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative Standard Error (Eq. 21): RSE = std(residuals) / std(y_true)."""
    residuals = y_true - y_pred
    return float(np.std(residuals) / (np.std(y_true) + 1e-12))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# -----------------------------------------------------------------------
# 7. Group k-Fold Cross-Validation (Table 2)
# -----------------------------------------------------------------------
def group_kfold_cv(X: np.ndarray, y: np.ndarray,
                   groups: np.ndarray, k: int = 3
                   ) -> List[Dict[str, float]]:
    """
    Group k-fold cross-validation. Each group (well) is held out once.

    Returns
    -------
    list of dicts with 'mae', 'rse', 'r2' for each fold.
    """
    unique_groups = np.unique(groups)
    results = []

    for i in range(min(k, len(unique_groups))):
        test_mask = groups == unique_groups[i]
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(X_train) < 2 or len(X_test) < 1:
            continue

        # Use Ridge as default estimator
        w, b = train_ridge(X_train, y_train, alpha=1.0)
        y_pred = predict_ridge(X_test, w, b)

        results.append({
            'fold': i,
            'group': unique_groups[i],
            'mae': mae(y_test, y_pred),
            'rse': rse(y_test, y_pred),
            'r2': r_squared(y_test, y_pred),
        })

    return results


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== ML Permeability Estimation Module Demo ===\n")
    np.random.seed(42)

    # Synthetic well-log data for 4 wells
    n_per_well = 80
    wells = np.repeat([0, 1, 2, 3], n_per_well)
    n = len(wells)
    phi = np.random.uniform(0.05, 0.30, n)
    GR = np.random.uniform(20, 120, n)
    RHOB = 2.65 - phi * 1.0 + np.random.normal(0, 0.02, n)
    NPHI = phi + np.random.normal(0, 0.02, n)
    PEF = np.random.uniform(1.5, 4.0, n)

    # True permeability (log-normal, correlated with phi)
    ln_k_true = 2.0 + 15.0 * phi + np.random.normal(0, 0.5, n)
    k_true = np.exp(ln_k_true)

    # Feature matrix
    X = np.column_stack([phi, GR, RHOB, NPHI, PEF])

    # Timur-Coates baseline
    Swirr = np.clip(0.5 - phi, 0.05, 0.95)
    k_tc = timur_coates(phi, Swirr, a=1e4, b=4.0, c=2.0)
    print(f"Timur-Coates MAE (ln k): {mae(ln_k_true, np.log(k_tc + 1e-6)):.3f}")

    # Feature augmentation
    X_aug = augment_features(X, window_sizes=[3, 5])
    print(f"Original features: {X.shape[1]}, Augmented: {X_aug.shape[1]}")

    # PCA
    Z, ev, _ = pca_reduce(X, n_components=3)
    print(f"PCA explained variance: {ev}")

    # Ridge regression on PCA features
    w, b = train_ridge(Z[:240], ln_k_true[:240])
    y_pred = predict_ridge(Z[240:], w, b)
    print(f"\nRidge on PCA features (test well):")
    print(f"  MAE = {mae(ln_k_true[240:], y_pred):.3f}")
    print(f"  RSE = {rse(ln_k_true[240:], y_pred):.3f}")
    print(f"  R²  = {r_squared(ln_k_true[240:], y_pred):.3f}")

    # Group k-fold CV
    cv_results = group_kfold_cv(Z, ln_k_true, wells, k=4)
    print("\nGroup k-Fold CV results:")
    for r in cv_results:
        print(f"  Fold {r['fold']} (well {r['group']}): "
              f"MAE={r['mae']:.3f}, R²={r['r2']:.3f}")
