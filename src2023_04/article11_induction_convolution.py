"""
Article 11: Machine-Learning-Based Convolution Method for Fast Forward
Modeling of Induction Log
Hagiwara (2023)
DOI: 10.30632/PJV64N2-2023a11

Companion to Article 10.  Implements the inverse direction: given a
layered earth model R_model, predict the apparent resistivity R_app at
each depth via an ML 'convolution' operating on a window of model
resistivities (101-point, 50-ft window for 60° deviated wells, after
the paper).
"""

import numpy as np
import xgboost as xgb

from article10_induction_deconvolution import (
    make_layered_model,
    synthesise_apparent_resistivity,
    rmsle,
)


# ----------------------- linear convolution baseline (Eq. 2) ---

def fit_linear_conv_weights(R_model, R_app, window=101):
    pad = window // 2
    log_model = np.log(np.maximum(R_model, 1e-9))
    log_pad = np.pad(log_model, pad, mode="edge")
    rows = np.lib.stride_tricks.sliding_window_view(log_pad, window)
    coef, *_ = np.linalg.lstsq(rows, np.log(np.maximum(R_app, 1e-9)), rcond=None)
    return coef


def linear_convolution(R_model, weights):
    pad = len(weights) // 2
    log_model = np.log(np.maximum(R_model, 1e-9))
    log_pad = np.pad(log_model, pad, mode="edge")
    log_app = np.convolve(log_pad, weights[::-1], mode="valid")
    return np.exp(log_app)


# ---------------------------------- ML convolution model ---

class MLConvolution:
    def __init__(self, window=101, n_estimators=300, max_depth=5, seed=0):
        self.window = window
        self.model = xgb.XGBRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      learning_rate=0.08,
                                      random_state=seed,
                                      verbosity=0)

    def _features(self, R_model):
        pad = self.window // 2
        log_m = np.log(np.maximum(R_model, 1e-9))
        log_pad = np.pad(log_m, pad, mode="edge")
        return np.lib.stride_tricks.sliding_window_view(log_pad, self.window).copy()

    def fit(self, R_model, R_app):
        X = self._features(R_model)
        y = np.log(np.maximum(R_app, 1e-9))
        self.model.fit(X, y)
        return self

    def predict(self, R_model):
        X = self._features(R_model)
        return np.exp(self.model.predict(X))


# ---------------------------------- window-size scan ---

def window_size_scan(R_model_train, R_app_train, R_model_test, R_app_test,
                     window_sizes=(11, 21, 51, 101, 151)):
    results = {}
    for w in window_sizes:
        ml = MLConvolution(window=w, n_estimators=150, max_depth=4).fit(
            R_model_train, R_app_train)
        pred_te = ml.predict(R_model_test)
        results[w] = rmsle(R_app_test, pred_te)
    return results


# --------------------------------------------------- testing ---

def test_all():
    print("=" * 60)
    print("Article 11: ML-Based Convolution for Forward Induction Modelling")
    print("=" * 60)
    # Training set
    _, R_tr = make_layered_model(n_samples=4000, n_layers=200, seed=0)
    R_app_tr = synthesise_apparent_resistivity(R_tr, sigma_smooth=8.0)  # 60° deviation -> wider response

    # Linear baseline
    weights = fit_linear_conv_weights(R_tr, R_app_tr, window=51)
    R_lin_tr = linear_convolution(R_tr, weights)

    # ML convolution
    ml = MLConvolution(window=51, n_estimators=200, max_depth=5).fit(R_tr, R_app_tr)
    R_ml_tr = ml.predict(R_tr)

    # 3 test models
    test_rmsle_lin = []
    test_rmsle_ml = []
    for ts in range(3):
        _, R_te = make_layered_model(n_samples=2500, n_layers=100, seed=20 + ts)
        R_app_te = synthesise_apparent_resistivity(R_te, sigma_smooth=8.0)
        test_rmsle_lin.append(rmsle(R_app_te, linear_convolution(R_te, weights)))
        test_rmsle_ml.append(rmsle(R_app_te, ml.predict(R_te)))

    print(f"  Train RMSLE   linear    = {rmsle(R_app_tr, R_lin_tr):.4f}")
    print(f"  Train RMSLE   ML        = {rmsle(R_app_tr, R_ml_tr):.4f}")
    print(f"  Test  RMSLE   linear    = {np.mean(test_rmsle_lin):.4f}")
    print(f"  Test  RMSLE   ML        = {np.mean(test_rmsle_ml):.4f}")

    # Window-size scan (smaller window list for speed)
    _, R_te = make_layered_model(n_samples=1500, n_layers=80, seed=99)
    R_app_te = synthesise_apparent_resistivity(R_te, sigma_smooth=8.0)
    scan = window_size_scan(R_tr, R_app_tr, R_te, R_app_te,
                            window_sizes=(11, 31, 51, 81))
    print(f"  Window-size scan RMSLE: {scan}")

    avg_ml = np.mean(test_rmsle_ml)
    avg_lin = np.mean(test_rmsle_lin)
    assert avg_ml <= avg_lin * 1.5, "ML should be competitive with linear"
    print("  PASS")
    return {"test_ml": avg_ml, "test_linear": avg_lin, "window_scan": scan}


if __name__ == "__main__":
    test_all()
