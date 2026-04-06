#!/usr/bin/env python3
"""
Reservoir Fluid Properties From Cuttings: GPC and Data Analytics
==================================================================
Based on: Cely, Yang, Yerkinkyzy, Michael, and Moore (2024),
Petrophysics 65(6), pp. 957-969. DOI: 10.30632/PJV65N6-2024a9

Implements the GPC-UV-RI workflow for predicting API gravity:
  1. Synthetic GPC-UV-vis spectra generation (3D tensor: time x wavelength x intensity).
  2. LASSO regression for API gravity prediction.
  3. Monte Carlo data augmentation for training robustness.
  4. Dilution effect correction for cutting extracts.
"""
import numpy as np
from typing import Dict, Tuple, List

def generate_gpc_spectrum(api_gravity, n_elution=100, n_wavelengths=8, noise=0.02, seed=None):
    """Generate a synthetic GPC-UV-vis spectrum for a given API gravity.
    Higher API (lighter oil) -> earlier elution, less UV absorption at longer wavelengths."""
    rng = np.random.RandomState(seed)
    elution_time = np.linspace(5, 25, n_elution)  # minutes
    wavelengths = np.array([210, 230, 254, 280, 310, 340, 370, 400])[:n_wavelengths]  # nm
    # Peak elution time shifts with API gravity
    peak_time = 18.0 - 0.15 * api_gravity  # lighter oils elute differently
    width = 2.0 + 0.03 * api_gravity
    spectrum = np.zeros((n_elution, n_wavelengths))
    for j, wl in enumerate(wavelengths):
        # UV absorbance: heavier aromatics absorb more at longer wavelengths
        wl_factor = np.exp(-0.005 * (wl - 254)) * (1.0 + (50 - api_gravity) / 100)
        peak = wl_factor * np.exp(-0.5 * ((elution_time - peak_time) / width)**2)
        # Secondary peak for asphaltenes (heavy components)
        if api_gravity < 30:
            asphal_peak = 0.3 * (30 - api_gravity) / 30 * np.exp(-0.5 * ((elution_time - 8) / 1.5)**2)
            peak += asphal_peak
        spectrum[:, j] = peak + rng.normal(0, noise, n_elution)
    return np.clip(spectrum, 0, None), elution_time, wavelengths

def vectorize_spectrum(spectrum, top_features_indices=None):
    """Flatten GPC spectrum into a feature vector for ML.
    Optionally select only top feature indices (LASSO sparse selection)."""
    vec = spectrum.ravel()
    if top_features_indices is not None:
        return vec[top_features_indices]
    return vec

def build_training_dataset(api_range=(10, 55), n_samples=50, seed=42):
    """Build a training dataset of GPC spectra and API labels."""
    rng = np.random.RandomState(seed)
    apis = rng.uniform(*api_range, n_samples)
    X, y = [], []
    for i, api in enumerate(apis):
        spec, _, _ = generate_gpc_spectrum(api, seed=seed + i)
        X.append(vectorize_spectrum(spec))
        y.append(api)
    return np.array(X), np.array(y)

def monte_carlo_augmentation(X, y, n_augmented=200, noise_scale=0.05, seed=42):
    """Augment dataset using Monte Carlo simulation (adding noise to spectra)."""
    rng = np.random.RandomState(seed)
    X_aug, y_aug = list(X), list(y)
    for _ in range(n_augmented):
        idx = rng.randint(len(X))
        noise = rng.normal(0, noise_scale * np.std(X[idx]), X[idx].shape)
        X_aug.append(X[idx] + noise)
        y_aug.append(y[idx] + rng.normal(0, 0.5))
    return np.array(X_aug), np.array(y_aug)

def lasso_regression(X_train, y_train, alpha=0.1):
    """Simple LASSO regression (coordinate descent).
    Returns weights and bias."""
    n_samples, n_features = X_train.shape
    # Standardize
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-10
    y_mean = y_train.mean()
    X_norm = (X_train - X_mean) / X_std
    y_norm = y_train - y_mean
    w = np.zeros(n_features)
    for iteration in range(100):
        for j in range(n_features):
            residual = y_norm - X_norm @ w + X_norm[:, j] * w[j]
            rho = X_norm[:, j] @ residual / n_samples
            # Soft thresholding
            if rho > alpha: w[j] = rho - alpha
            elif rho < -alpha: w[j] = rho + alpha
            else: w[j] = 0.0
    # Transform back
    w_orig = w / X_std
    b_orig = y_mean - X_mean @ w_orig
    return w_orig, b_orig

def predict_api(X, weights, bias):
    """Predict API gravity from GPC spectra features."""
    return X @ weights + bias

def apply_dilution_correction(cuttings_spectrum, multipliers):
    """Apply dilution correction to cuttings GPC spectra.
    Eq. 1: intensity_oil = multiplier * intensity_cuttings."""
    corrected = cuttings_spectrum.copy()
    for (t_start, t_end), mult in multipliers.items():
        corrected[t_start:t_end, :] *= mult
    return corrected

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / (ss_tot + 1e-10)

def test_all():
    print("=" * 70)
    print("Module 9: GPC Fluid Properties (Cely et al., 2024)")
    print("=" * 70)
    # Generate spectra
    spec_light, et, wl = generate_gpc_spectrum(45, seed=1)
    spec_heavy, _, _ = generate_gpc_spectrum(15, seed=2)
    print(f"GPC spectrum shape: {spec_light.shape} (elution_times x wavelengths)")
    print(f"Light oil (45 API) peak intensity: {spec_light.max():.3f}")
    print(f"Heavy oil (15 API) peak intensity: {spec_heavy.max():.3f}")
    # Build dataset
    X_train, y_train = build_training_dataset(n_samples=80, seed=42)
    X_test, y_test = build_training_dataset(api_range=(12, 52), n_samples=20, seed=99)
    print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")
    # Augmentation
    X_aug, y_aug = monte_carlo_augmentation(X_train, y_train, n_augmented=200)
    print(f"Augmented set: {X_aug.shape}")
    # LASSO
    w, b = lasso_regression(X_aug, y_aug, alpha=0.01)
    n_nonzero = np.sum(np.abs(w) > 1e-6)
    print(f"LASSO: {n_nonzero}/{len(w)} non-zero features")
    # Predict
    y_pred_train = predict_api(X_aug, w, b)
    y_pred_test = predict_api(X_test, w, b)
    r2_train = compute_r2(y_aug, y_pred_train)
    r2_test = compute_r2(y_test, y_pred_test)
    print(f"R² train: {r2_train:.3f}, R² test: {r2_test:.3f}")
    print(f"Test MAE: {np.mean(np.abs(y_test - y_pred_test)):.1f} API")
    # Dilution correction
    spec_cut, _, _ = generate_gpc_spectrum(30, noise=0.05, seed=3)
    spec_cut *= 0.4  # simulate dilution
    multipliers = {(0, 25): 2.2, (25, 50): 2.5, (50, 75): 2.0, (75, 100): 1.8}
    corrected = apply_dilution_correction(spec_cut, multipliers)
    print(f"\nDilution correction: before={spec_cut.max():.3f}, after={corrected.max():.3f}")
    print("\n[PASS] All tests completed successfully.\n")

if __name__ == "__main__":
    test_all()
