#!/usr/bin/env python3
"""
Comparative Analysis of TOC Logging Evaluation Methods Using Machine Learning
=============================================================================
Implements the methodology from:
  Dong, M., Shang, J., Tian, L., Wu, M., and Nie, X., 2025,
  "Comparative Analysis of TOC Logging Evaluation Methods Using
  Machine Learning – A Case Study of the Ordos Basin-Yanchang Formation,"
  Petrophysics, Vol. 66, No. 3, pp. 425–448.

Key ideas implemented:
  - ΔlogR method for TOC prediction (Passey et al., 1990) – Eqs. 9–14.
  - Dual-shale-content method (Nie et al., 2017) – Eqs. 4–8.
  - Improved stacking ensemble ML model with hierarchical evaluation.
  - Automatic core homing (sliding-window depth matching).
  - Cook's distance outlier removal.

References:
  Passey, Q.R. et al., 1990 (ΔlogR method).
  Nie, X. et al., 2017 (dual shale content).
  Schmoker, J.W. and Hester, T.C., 1983.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List


# ============================================================
# ΔlogR Method (Passey et al., 1990)
# ============================================================

def delta_logr_ac(rt: np.ndarray, ac: np.ndarray,
                  rt_baseline: float, ac_baseline: float) -> np.ndarray:
    """
    ΔlogR using acoustic transit time overlay (Eq. 9).

    ΔlogR = log10(Rt/Rt_baseline) + 0.02 * (Δt - Δt_baseline)
    """
    return np.log10(rt / rt_baseline) + 0.02 * (ac - ac_baseline)


def delta_logr_den(rt: np.ndarray, den: np.ndarray,
                   rt_baseline: float, den_baseline: float) -> np.ndarray:
    """
    ΔlogR using density overlay (Eq. 11).

    ΔlogR = log10(Rt/Rt_baseline) - 2.50 * (ρb - ρ_baseline)
    """
    return np.log10(rt / rt_baseline) - 2.50 * (den - den_baseline)


def delta_logr_cnl(rt: np.ndarray, cnl: np.ndarray,
                   rt_baseline: float, cnl_baseline: float) -> np.ndarray:
    """
    ΔlogR using compensated neutron overlay (Eq. 10).

    ΔlogR = log10(Rt/Rt_baseline) + 4.0 * (N - N_baseline)
    """
    return np.log10(rt / rt_baseline) + 4.0 * (cnl - cnl_baseline)


def delta_logr_gr(rt: np.ndarray, gr: np.ndarray,
                  rt_baseline: float, gr_baseline: float) -> np.ndarray:
    """
    ΔlogR using gamma ray overlay (Eq. 12).

    ΔlogR = log10(Rt/Rt_baseline) + 0.01 * (GR - GR_baseline)
    """
    return np.log10(rt / rt_baseline) + 0.01 * (gr - gr_baseline)


def toc_from_delta_logr(delta_logr: np.ndarray,
                         lom: float) -> np.ndarray:
    """
    TOC from ΔlogR and maturity (Eq. 13).

    TOC = ΔlogR · 10^(2.297 - 0.1688·LOM)
    """
    return delta_logr * 10.0 ** (2.297 - 0.1688 * lom)


def lom_from_ro(ro: float) -> float:
    """
    Level of Organic Maturity from vitrinite reflectance (Eq. 14).

    LOM ≈ linear mapping used in Passey et al., 1990.
    Approximate: LOM = 10.0 + 5.0 * log10(Ro)  (simplified)
    """
    return 10.0 + 5.0 * np.log10(max(ro, 0.1))


# ============================================================
# Dual-Shale-Content Method (Nie et al., 2017)
# ============================================================

def shale_content_gr(gr: np.ndarray,
                     gr_min: float, gr_max: float,
                     gcur: float = 2.0) -> np.ndarray:
    """
    Total shale content from GR (Eqs. 4-5).

    I_sh = (GR - GR_min) / (GR_max - GR_min)
    V_sh = (2^(GCUR·I_sh) - 1) / (2^GCUR - 1)
    """
    i_sh = np.clip((gr - gr_min) / (gr_max - gr_min + 1e-10), 0, 1)
    v_sh = (2.0 ** (gcur * i_sh) - 1.0) / (2.0 ** gcur - 1.0)
    return v_sh


def shale_content_rt(rt: np.ndarray,
                     c: float, d: float = 1.5) -> np.ndarray:
    """
    Water-bearing shale content from RT (Eq. 6).

    V_shw = (c / RT)^d
    """
    return np.clip((c / rt) ** d, 0, 1)


def toc_dual_shale(v_sh: np.ndarray, v_shw: np.ndarray,
                    phi: np.ndarray, den: np.ndarray,
                    de_om: float = 1.2) -> np.ndarray:
    """
    TOC from dual shale content (Eqs. 7-8).

    V_sho = V_sh - V_shw
    TOC = V_sho · φ · DEN / DE_OM
    """
    v_sho = np.clip(v_sh - v_shw, 0, None)
    toc = v_sho * phi * den / de_om
    return toc


# ============================================================
# Core Homing (Sliding Window Algorithm)
# ============================================================

def sliding_window_core_homing(core_depths: np.ndarray,
                                core_values: np.ndarray,
                                log_depths: np.ndarray,
                                log_curves: np.ndarray,
                                max_shift: float = 2.0,
                                step: float = 0.125) -> Tuple[np.ndarray, float]:
    """
    Automatic core homing via sliding-window depth matching.
    Finds the depth shift that maximises the correlation between
    core values and the best-matching log curve.

    Parameters
    ----------
    core_depths : depth of core samples
    core_values : measured core property (e.g. TOC)
    log_depths  : depth of log samples (regular grid)
    log_curves  : 2-D array (n_depths × n_curves) of log values
    max_shift   : maximum shift in depth units
    step        : sliding step

    Returns
    -------
    adjusted_depths, best_shift
    """
    shifts = np.arange(-max_shift, max_shift + step, step)
    best_corr = -1
    best_shift = 0

    for s in shifts:
        shifted = core_depths + s
        # Interpolate log curves at shifted core depths
        for c in range(log_curves.shape[1]):
            interp_vals = np.interp(shifted, log_depths, log_curves[:, c])
            r = np.corrcoef(core_values, interp_vals)[0, 1]
            if abs(r) > best_corr:
                best_corr = abs(r)
                best_shift = s

    return core_depths + best_shift, best_shift


# ============================================================
# Cook's Distance Outlier Removal
# ============================================================

def cooks_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Cook's distance for simple linear regression (Eq. 2).

    Parameters
    ----------
    x : predictor (1-D)
    y : response (1-D)

    Returns
    -------
    D : array of Cook's distances
    """
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    residuals = y - y_hat
    mse = np.sum(residuals ** 2) / (n - 2)
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H)
    p = 2  # number of parameters
    D = (residuals ** 2 / (p * mse)) * (h / (1 - h) ** 2)
    return D


def remove_outliers_cooks(x: np.ndarray, y: np.ndarray,
                           threshold_factor: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers where Cook's distance > threshold_factor × mean(D).
    """
    D = cooks_distance(x, y)
    mask = D < threshold_factor * np.mean(D)
    return x[mask], y[mask]


# ============================================================
# Stacking Ensemble Model (simplified)
# ============================================================

class SimpleStackingTOC:
    """
    Simplified stacking ensemble for TOC prediction.
    Uses ridge regression + polynomial features as base learners
    and a linear meta-learner (mimicking the paper's approach
    without requiring sklearn).
    """

    def __init__(self, n_folds: int = 5, alpha: float = 1.0):
        self.n_folds = n_folds
        self.alpha = alpha
        self.base_weights = []
        self.meta_weights = None

    def _ridge_fit(self, X, y, alpha):
        """Ridge regression: w = (X'X + αI)^{-1} X'y."""
        n_feat = X.shape[1]
        I = np.eye(n_feat)
        I[0, 0] = 0  # don't regularise intercept
        w = np.linalg.solve(X.T @ X + alpha * I, X.T @ y)
        return w

    def _augment(self, X, degree=2):
        """Add polynomial features up to given degree."""
        feats = [X]
        if degree >= 2:
            feats.append(X ** 2)
        return np.column_stack([np.ones(X.shape[0])] + feats)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the stacking model."""
        n = len(y)
        # Base learners with different feature sets
        configs = [
            {"degree": 1, "alpha": 0.1},
            {"degree": 2, "alpha": 1.0},
            {"degree": 1, "alpha": 10.0},
            {"degree": 2, "alpha": 0.01},
        ]
        meta_features = np.zeros((n, len(configs)))

        for c_idx, cfg in enumerate(configs):
            fold_size = n // self.n_folds
            weights_list = []
            for fold in range(self.n_folds):
                val_idx = slice(fold * fold_size, (fold + 1) * fold_size)
                train_mask = np.ones(n, dtype=bool)
                train_mask[val_idx] = False

                X_aug = self._augment(X[train_mask], cfg["degree"])
                w = self._ridge_fit(X_aug, y[train_mask], cfg["alpha"])
                weights_list.append(w)

                X_val_aug = self._augment(X[val_idx], cfg["degree"])
                meta_features[val_idx, c_idx] = X_val_aug @ w

            # Final fit on full data for this base learner
            X_aug = self._augment(X, cfg["degree"])
            w_full = self._ridge_fit(X_aug, y, cfg["alpha"])
            self.base_weights.append((cfg, w_full))

        # Meta-learner: simple linear on stacked predictions
        meta_X = np.column_stack([np.ones(n), meta_features])
        self.meta_weights = self._ridge_fit(meta_X, y, 0.1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict TOC."""
        n = X.shape[0]
        meta_features = np.zeros((n, len(self.base_weights)))
        for c_idx, (cfg, w) in enumerate(self.base_weights):
            X_aug = self._augment(X, cfg["degree"])
            meta_features[:, c_idx] = X_aug @ w
        meta_X = np.column_stack([np.ones(n), meta_features])
        return meta_X @ self.meta_weights


# ============================================================
# Hierarchical Model Evaluation
# ============================================================

def hierarchical_model_selection(metrics: Dict[str, Dict[str, float]],
                                  weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
                                  ) -> str:
    """
    Three-level hierarchical analysis for model selection.

    Parameters
    ----------
    metrics : {model_name: {"R2": float, "MAE": float, "RMSE": float}}
    weights : relative importance of (R2, MAE, RMSE)

    Returns
    -------
    best_model_name
    """
    w_r2, w_mae, w_rmse = weights
    scores = {}
    for name, m in metrics.items():
        # Normalise: R2 higher is better, MAE/RMSE lower is better
        score = w_r2 * m["R2"] - w_mae * m["MAE"] - w_rmse * m["RMSE"]
        scores[name] = score
    return max(scores, key=scores.get)


# ============================================================
# Test
# ============================================================

def test_all():
    """Test all functions with synthetic data."""
    print("=" * 60)
    print("Testing toc_prediction module (Dong et al., 2025)")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n = 100

    # Synthetic logs
    depth = np.linspace(2000, 2100, n)
    ac = 280 + 40 * rng.randn(n)
    den = 2.4 + 0.15 * rng.randn(n)
    cnl = 0.15 + 0.05 * rng.randn(n)
    gr = 80 + 30 * rng.randn(n)
    rt = np.exp(1.5 + 0.8 * rng.randn(n))
    phi = 0.08 + 0.03 * rng.randn(n)

    # 1. ΔlogR methods
    dlr_ac = delta_logr_ac(rt, ac, rt_baseline=5.0, ac_baseline=260.0)
    dlr_den = delta_logr_den(rt, den, rt_baseline=5.0, den_baseline=2.55)
    toc_dlr = toc_from_delta_logr(dlr_ac, lom=lom_from_ro(0.8))
    print(f"\n1) ΔlogR-AC TOC: mean={np.mean(toc_dlr):.2f}, "
          f"std={np.std(toc_dlr):.2f}")
    print(f"   LOM(Ro=0.8) = {lom_from_ro(0.8):.2f}")

    # 2. Dual-shale-content
    v_sh = shale_content_gr(gr, gr_min=20, gr_max=150)
    v_shw = shale_content_rt(rt, c=3.0, d=1.5)
    toc_ds = toc_dual_shale(v_sh, v_shw, phi, den)
    print(f"\n2) Dual-shale TOC: mean={np.mean(toc_ds):.2f}")

    # 3. Core homing
    core_d = np.array([2020, 2040, 2060, 2080])
    core_v = np.array([3.5, 5.2, 2.1, 4.8])
    log_curves = np.column_stack([ac, gr])
    adj_d, shift = sliding_window_core_homing(core_d, core_v,
                                               depth, log_curves,
                                               max_shift=1.0)
    print(f"\n3) Core homing: best shift = {shift:.3f}")

    # 4. Cook's distance
    x = rng.randn(50)
    y = 2 * x + 1 + 0.5 * rng.randn(50)
    y[0] = 100  # outlier
    x_clean, y_clean = remove_outliers_cooks(x, y)
    print(f"\n4) Cook's distance: {len(x)} → {len(x_clean)} samples after outlier removal")

    # 5. Stacking model
    X_train = np.column_stack([ac, den, cnl, gr, np.log10(rt + 1)])
    y_train = 2.0 + 0.01 * ac - 3.0 * den + 10 * cnl + rng.randn(n) * 0.5

    model = SimpleStackingTOC(n_folds=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    r2 = 1.0 - np.sum((y_train - y_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    mae = np.mean(np.abs(y_train - y_pred))
    rmse = np.sqrt(np.mean((y_train - y_pred) ** 2))
    print(f"\n5) Stacking model (train): R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

    # 6. Hierarchical evaluation
    metrics = {
        "ΔlogR": {"R2": 0.40, "MAE": 5.41, "RMSE": 7.08},
        "Stacking": {"R2": 0.66, "MAE": 1.89, "RMSE": 3.14},
        "DualShale": {"R2": 0.35, "MAE": 6.00, "RMSE": 8.00},
    }
    best = hierarchical_model_selection(metrics)
    print(f"\n6) Hierarchical model selection: best = {best}")
    assert best == "Stacking"

    print("\n✓ All toc_prediction tests passed.\n")


if __name__ == "__main__":
    test_all()
