"""
article_03_okwoli_probe_screening.py
====================================
Implementation of ideas from:

    Okwoli, E. and Potter, D. K. (2023).  "Probe Screening Techniques for
    Rapid, High-Resolution Core Analysis and Their Potential Usefulness
    for Energy Transition Applications."  Petrophysics, 64(5), 640-655.
    DOI: 10.30632/PJV64N5-2023a3

The paper introduces four hand-held / bench-top probe measurements that
can be performed continuously along slabbed cores at lamina-scale
resolution (mm scale):

    * probe luminance        (optical reflectance, brightness)
    * probe magnetic susceptibility
    * probe acoustic velocity
    * probe permeability     (mini-permeameter)

The advantages they emphasise are: (1) very high vertical resolution,
(2) non-destructive, (3) better depth-matching to wireline logs, and
(4) the ability to flag thin cemented or fractured zones missed by
plug-based core analysis.

This module:

    * generates a synthetic high-resolution depth track of the four
      probe measurements with realistic correlations
    * upscales the high-resolution data to plug- and log-scale, showing
      how thin features are smeared out
    * cross-correlates probe permeability vs probe luminance / mag-sus
      (the two most useful proxies highlighted in the paper) and fits
      simple log-linear regressions that can be used as
      "permeability-from-image" predictors
    * implements a depth-shift function that aligns probe data to a
      coarser log measurement by maximising cross-correlation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic core
# ---------------------------------------------------------------------------
@dataclass
class CoreProbes:
    depth_m: np.ndarray         # high-resolution depth axis
    luminance: np.ndarray       # 0..1
    mag_sus: np.ndarray         # 1e-9 m3/kg (mass susceptibility)
    vp: np.ndarray              # m/s
    perm_mD: np.ndarray         # mD (mini-permeameter)


def make_synthetic_core(length_m: float = 5.0,
                        sample_dz_mm: float = 1.0,
                        n_thin_features: int = 4,
                        rng: np.random.Generator | None = None) -> CoreProbes:
    """Build a synthetic core with mm-scale heterogeneity, including a few
    thin cemented (low-perm, high-luminance) features such as the paper
    flags as 'easy to miss with plug-based analysis'."""
    if rng is None:
        rng = np.random.default_rng(0)
    dz = sample_dz_mm * 1e-3
    z = np.arange(0, length_m, dz)
    n = z.size

    # smooth bedding signal -- Markov-chain-like AR(1)
    base = np.zeros(n)
    for i in range(1, n):
        base[i] = 0.995 * base[i - 1] + 0.05 * rng.standard_normal()
    base = (base - base.min()) / (base.max() - base.min())

    luminance = 0.3 + 0.5 * base + 0.02 * rng.standard_normal(n)
    mag_sus = 5.0 + 20.0 * (1.0 - base) + 1.5 * rng.standard_normal(n)
    perm = 10.0 ** (1.5 + 1.5 * base + 0.15 * rng.standard_normal(n))   # 30 - 5000 mD
    vp = 3500.0 + 1500.0 * (1.0 - base) + 50.0 * rng.standard_normal(n)

    # Inject thin cemented streaks
    for _ in range(n_thin_features):
        i0 = rng.integers(20, n - 20)
        w = rng.integers(2, 8)        # 2-8 mm
        luminance[i0:i0 + w] += 0.3
        perm[i0:i0 + w] *= 0.01       # 100x perm reduction
        vp[i0:i0 + w] += 1500.0
        mag_sus[i0:i0 + w] *= 0.3

    luminance = np.clip(luminance, 0.0, 1.0)
    mag_sus = np.clip(mag_sus, 0.1, None)
    perm = np.clip(perm, 1e-3, None)
    return CoreProbes(z, luminance, mag_sus, vp, perm)


# ---------------------------------------------------------------------------
# Up-scaling: plug-scale and log-scale
# ---------------------------------------------------------------------------
def boxcar(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average."""
    if window <= 1:
        return x.copy()
    k = np.ones(window) / window
    return np.convolve(x, k, mode="same")


def upscale_to(probe: CoreProbes, sample_dz_mm: float,
               target_window_cm: float) -> CoreProbes:
    """Upscale all four probe channels to a coarser support length."""
    w = max(1, int(round((target_window_cm * 10.0) / sample_dz_mm)))
    return CoreProbes(probe.depth_m.copy(),
                      boxcar(probe.luminance, w),
                      boxcar(probe.mag_sus, w),
                      boxcar(probe.vp, w),
                      np.exp(boxcar(np.log(probe.perm_mD), w)))


# ---------------------------------------------------------------------------
# Probe permeability vs other channels - regression
# ---------------------------------------------------------------------------
def fit_perm_predictor(probe: CoreProbes) -> tuple[np.ndarray, float]:
    """Simple multivariate log-linear fit:
        log10(k) = a*lum + b*log10(MS) + c*Vp + d
    Returns the coefficients (a,b,c,d) and the R^2.
    """
    X = np.column_stack([probe.luminance,
                         np.log10(probe.mag_sus),
                         probe.vp,
                         np.ones_like(probe.vp)])
    y = np.log10(probe.perm_mD)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return coef, float(r2)


def predict_perm(coef: np.ndarray, lum: np.ndarray,
                 mag_sus: np.ndarray, vp: np.ndarray) -> np.ndarray:
    a, b, c, d = coef
    return 10.0 ** (a * lum + b * np.log10(mag_sus) + c * vp + d)


# ---------------------------------------------------------------------------
# Depth-shift between probe and log
# ---------------------------------------------------------------------------
def best_depth_shift(probe_signal: np.ndarray, log_signal: np.ndarray,
                     dz_m: float, max_shift_m: float = 0.5) -> float:
    """Return the depth shift (positive = move probe deeper) that maximises
    Pearson correlation with the (resampled) log signal."""
    n_max = int(max_shift_m / dz_m)
    best_corr = -np.inf
    best_shift = 0
    a = (probe_signal - probe_signal.mean()) / probe_signal.std()
    b = (log_signal - log_signal.mean()) / log_signal.std()
    for s in range(-n_max, n_max + 1):
        a_s = np.roll(a, s)
        c = float(np.mean(a_s * b))
        if c > best_corr:
            best_corr, best_shift = c, s
    return best_shift * dz_m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    rng = np.random.default_rng(7)
    core = make_synthetic_core(length_m=3.0, sample_dz_mm=1.0,
                               n_thin_features=5, rng=rng)
    assert core.depth_m.size > 2900
    # thin features should appear in mm-scale data:
    n_low = np.sum(core.perm_mD < 1.0)
    assert n_low >= 2, n_low

    # Up-scale to 30 cm (typical wireline log) - thin features should be muted
    coarse = upscale_to(core, 1.0, target_window_cm=30.0)
    n_low_coarse = np.sum(coarse.perm_mD < 1.0)
    assert n_low_coarse <= n_low, (n_low_coarse, n_low)

    # Permeability predictor should explain a substantial fraction of variance
    coef, r2 = fit_perm_predictor(core)
    assert r2 > 0.5, r2
    pred = predict_perm(coef, core.luminance, core.mag_sus, core.vp)
    assert pred.shape == core.perm_mD.shape

    # Depth-shift recovery (sign and magnitude). The blurred 'log' loses
    # high-frequency content so the recovered shift is approximate.
    shifted = np.roll(core.luminance, 50)        # +5 cm (deeper)
    log_resp = boxcar(core.luminance, 200)       # 'log' = blurred truth
    shift = best_depth_shift(shifted, log_resp, dz_m=1e-3, max_shift_m=0.2)
    # Should recover a negative shift of order -5 cm (probe needs to move
    # shallower to align with the log)
    assert -0.10 < shift < 0.0, shift

    print(f"article_03_okwoli_probe_screening: OK  (R^2={r2:.2f}, "
          f"shift={shift*100:.1f} cm)")


if __name__ == "__main__":
    test_all()
