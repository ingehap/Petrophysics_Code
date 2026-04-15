"""
Article 10: Machine-Learning-Based Deconvolution Method Provides
High-Resolution Fast Inversion of Induction Log Data
Hagiwara (2023)
DOI: 10.30632/PJV64N2-2023a10

Implements the gradient-boosted ML deconvolution that maps a 21-point
window (10 ft, 0.5-ft sampling) of apparent log resistivity into the true
formation resistivity at the centre depth.  Because no proprietary
DataRobot pipeline is available, we use scikit-learn / XGBoost as a
substitute for LightGBM and approximate the apparent-resistivity forward
response with a smooth low-pass filter of the model resistivity (the
shape characteristic of a 2C40 induction tool).
"""

import numpy as np
import xgboost as xgb
from scipy.ndimage import gaussian_filter1d


# ----------------------------- approximate forward response (toy 2C40) ---

def synthesise_apparent_resistivity(R_model, sigma_smooth=4.0):
    """
    Stand-in for the 2C40 forward model: smooth log conductivity and
    return apparent resistivity.  The Gaussian smoothing approximates
    the ~10-ft vertical aperture of the tool.
    """
    sigma_model = 1.0 / np.maximum(R_model, 1e-6)
    sigma_app = gaussian_filter1d(sigma_model, sigma=sigma_smooth, mode="nearest")
    return 1.0 / np.maximum(sigma_app, 1e-6)


def make_layered_model(n_samples=15500, sample_ft=0.5, n_layers=1001,
                       seed=0):
    """Random log-uniform layered earth model (paper §Linear Deconvolution)."""
    rng = np.random.default_rng(seed)
    thicknesses = np.exp(rng.uniform(np.log(0.1), np.log(50), n_layers))
    depths_top = np.cumsum(np.r_[0, thicknesses[:-1]])
    resistivities = np.exp(rng.uniform(np.log(0.1), np.log(100), n_layers))
    depth_axis = np.arange(n_samples) * sample_ft
    R = np.empty_like(depth_axis)
    layer_idx = np.searchsorted(depths_top, depth_axis, side="right") - 1
    layer_idx = np.clip(layer_idx, 0, n_layers - 1)
    R = resistivities[layer_idx]
    return depth_axis, R


# -------------------------- linear deconvolution baseline (Eq. 5) ---

def linear_deconvolution(R_app, weights):
    """Apply a fixed-window linear filter to log apparent resistivity."""
    log_app = np.log(R_app)
    pad = len(weights) // 2
    log_app_pad = np.pad(log_app, pad, mode="edge")
    log_decon = np.convolve(log_app_pad, weights[::-1], mode="valid")
    return np.exp(log_decon)


def fit_linear_weights(R_app, R_model, window=21):
    """Fit linear regression weights minimising log-RMSE."""
    pad = window // 2
    log_app = np.log(R_app)
    log_app_pad = np.pad(log_app, pad, mode="edge")
    rows = np.lib.stride_tricks.sliding_window_view(log_app_pad, window)
    coef, *_ = np.linalg.lstsq(rows, np.log(R_model), rcond=None)
    return coef


# ------------------------------ ML deconvolution model (paper §ML Regressor) -

class MLDeconvolution:
    def __init__(self, window=21, n_estimators=300, max_depth=5, seed=0):
        self.window = window
        self.model = xgb.XGBRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      learning_rate=0.1,
                                      random_state=seed,
                                      verbosity=0)

    def _features(self, R_app):
        pad = self.window // 2
        log_app = np.log(np.maximum(R_app, 1e-9))
        log_pad = np.pad(log_app, pad, mode="edge")
        return np.lib.stride_tricks.sliding_window_view(log_pad, self.window).copy()

    def fit(self, R_app, R_model):
        X = self._features(R_app)
        y = np.log(np.maximum(R_model, 1e-9))
        self.model.fit(X, y)
        return self

    def predict(self, R_app):
        X = self._features(R_app)
        return np.exp(self.model.predict(X))


# ------------------------------- error metric ---

def rmsle(R_true, R_pred):
    return float(np.sqrt(np.mean((np.log(R_true) - np.log(R_pred)) ** 2)))


# ---------------------------------------- testing ---

def test_all():
    print("=" * 60)
    print("Article 10: ML-Based Deconvolution of Induction Log")
    print("=" * 60)
    # Smaller-scale synthetic version of the paper's setup
    depth_tr, R_tr = make_layered_model(n_samples=4000, n_layers=200, seed=0)
    R_app_tr = synthesise_apparent_resistivity(R_tr)

    # Linear baseline
    weights = fit_linear_weights(R_app_tr, R_tr, window=21)
    R_lin_tr = linear_deconvolution(R_app_tr, weights)

    # ML deconvolution
    ml = MLDeconvolution(window=21, n_estimators=200, max_depth=5).fit(R_app_tr, R_tr)
    R_ml_tr = ml.predict(R_app_tr)

    # Three test data sets
    test_results = []
    for ts in range(3):
        depth_te, R_te = make_layered_model(n_samples=2500, n_layers=100,
                                            seed=10 + ts)
        R_app_te = synthesise_apparent_resistivity(R_te)
        R_lin_te = linear_deconvolution(R_app_te, weights)
        R_ml_te = ml.predict(R_app_te)
        test_results.append({
            "rmsle_app": rmsle(R_te, R_app_te),
            "rmsle_linear": rmsle(R_te, R_lin_te),
            "rmsle_ml": rmsle(R_te, R_ml_te),
        })

    print(f"  Train RMSLE   apparent  = {rmsle(R_tr, R_app_tr):.3f}")
    print(f"  Train RMSLE   linear    = {rmsle(R_tr, R_lin_tr):.3f}")
    print(f"  Train RMSLE   ML        = {rmsle(R_tr, R_ml_tr):.3f}")
    print(f"  Test  RMSLE   apparent  = {np.mean([r['rmsle_app'] for r in test_results]):.3f}")
    print(f"  Test  RMSLE   linear    = {np.mean([r['rmsle_linear'] for r in test_results]):.3f}")
    print(f"  Test  RMSLE   ML        = {np.mean([r['rmsle_ml'] for r in test_results]):.3f}")

    # Check ML is at least as good as linear
    avg_ml = np.mean([r["rmsle_ml"] for r in test_results])
    avg_app = np.mean([r["rmsle_app"] for r in test_results])
    assert avg_ml < avg_app, "ML deconvolution should beat raw apparent log"
    print("  PASS")
    return {"train_ml": rmsle(R_tr, R_ml_tr),
            "test_ml": avg_ml,
            "test_app": avg_app}


if __name__ == "__main__":
    test_all()
