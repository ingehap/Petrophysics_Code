"""
Article 1: Data Quality Considerations for Petrophysical Machine-Learning
           Models
McDonald (2021)
DOI: 10.30632/PJV62N6-2021a1

A review/tutorial on why data quality governs petrophysical ML outcomes:
data-quality dimensions, well-log-specific issues (outliers, missing data,
normalization, naming), remediation, feature selection, and ML evaluation.
Three Volve case studies show ML degradation from missing data, additive
noise, and poor feature selection.

Implements:

  - z-score outlier detection  z = (x - mean) / sigma          (Eq. 1)
  - interquartile-range (box-plot) outliers  IQR = Q3 - Q1     (Eq. 2)
  - simple normalization  V_norm = a * V + x                   (Eq. 3)
  - reference-percentile normalization (Shier 2004)            (Eq. 4)
  - precision = TP/(TP+FP)                                     (Eq. 5)
  - recall = TP/(TP+FN)                                        (Eq. 6)
  - mean absolute error                                        (Eq. 7)
  - root-mean-square error                                     (Eq. 8)
  - Gaussian noise generation / additive corruption           (Eqs. 9-10)
  - Pearson correlation coefficient                            (Eq. 11)

Note: the journal's typeset equations are image-rendered and were not in the
machine-readable text; the forms here are faithful standard reconstructions
of the methods described (every symbol is defined in the paper's prose).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

SENTINELS = (-999, -999.25, -9999)     # common LAS/DLIS no-data values


# ---------------------------------------------- Eq. 1: z-score ----------

def zscore(x):
    """z = (x - mean) / sigma  (Eq. 1)."""
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


def zscore_outliers(x, threshold=3.0):
    """Boolean outlier mask using |z| > threshold (default 3 -> 99.7%)."""
    return np.abs(zscore(x)) > threshold


# ---------------------------------------------- Eq. 2: IQR --------------

def iqr_bounds(x, k=1.5):
    """Box-plot whisker bounds (Q1 - k*IQR, Q3 + k*IQR), IQR = Q3 - Q1 (Eq. 2)."""
    q1, q3 = np.percentile(np.asarray(x, float), [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def iqr_outliers(x, k=1.5):
    lo, hi = iqr_bounds(x, k)
    x = np.asarray(x, float)
    return (x < lo) | (x > hi)


# ---------------------------------------------- Eqs. 3-4: normalize -----

def normalize_simple(v, a, x):
    """V_norm = a * V + x  (Eq. 3): multiplier a, offset x."""
    return a * np.asarray(v, float) + x


def normalize_reference(v, R_min, R_max, W_min, W_max):
    """Reference-percentile normalization (Eq. 4, Shier 2004).

    V_norm = R_min + (R_max - R_min) * (V - W_min)/(W_max - W_min).
    R_* are reference-well 5th/95th percentiles, W_* the target well's.
    """
    v = np.asarray(v, float)
    return R_min + (R_max - R_min) * (v - W_min) / (W_max - W_min)


# ---------------------------------------------- Eqs. 5-6: P / R ---------

def precision(tp, fp):
    """Precision = TP / (TP + FP)  (Eq. 5)."""
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(tp, fn):
    """Recall = TP / (TP + FN)  (Eq. 6)."""
    return tp / (tp + fn) if (tp + fn) else 0.0


# ---------------------------------------------- Eqs. 7-8: errors --------

def mae(y_true, y_pred):
    """Mean absolute error (Eq. 7)."""
    return petrolib.ml_stats.mae(y_true, y_pred)


def rmse(y_true, y_pred):
    """Root-mean-square error (Eq. 8)."""
    return petrolib.ml_stats.rmse(y_true, y_pred)


# ---------------------------------------------- Eqs. 9-10: noise --------

def add_gaussian_noise(x, sigma_fraction, rng=None):
    """Add N(0, sigma^2) noise with sigma = fraction * mean(|x|) (Eqs. 9-10)."""
    x = np.asarray(x, float)
    rng = rng or np.random.default_rng(0)
    sigma = sigma_fraction * np.abs(x).mean()
    return x + rng.normal(0.0, sigma, size=x.shape)


# ---------------------------------------------- Eq. 11: Pearson ---------

def pearson(x, y):
    """Pearson correlation coefficient r (Eq. 11)."""
    r = petrolib.ml_stats.pearson_r(x, y)
    return r if np.isfinite(r) else 0.0  # historical zero-variance fallback


# ---------------------------------------------- missing data ------------

def sentinels_to_nan(x, sentinels=SENTINELS):
    """Replace LAS/DLIS no-data sentinels with NaN."""
    x = np.asarray(x, float).copy()
    for s in sentinels:
        x[np.isclose(x, s)] = np.nan
    return x


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Data Quality Considerations for ML Models")
    print("=" * 60)

    # Confusion matrix from Table 3 (positive = sandstone)
    TP, FN, FP, TN = 100, 10, 42, 200
    p = precision(TP, FP)
    r = recall(TP, FN)
    print(f"  precision (Eq. 5)      = {p:.4f}")
    print(f"  recall    (Eq. 6)      = {r:.4f}")
    assert abs(p - 0.7042) < 1e-3 and abs(r - 0.9091) < 1e-3

    # z-score + IQR flag a planted spike
    x = np.r_[np.random.default_rng(0).normal(50, 5, 200), [500.0]]
    assert zscore_outliers(x)[-1]
    assert iqr_outliers(x)[-1]
    print(f"  z-score / IQR caught planted outlier = True")

    # sentinels -> NaN
    log = np.array([60.0, -999.25, 62.0, -9999.0, 64.0])
    clean = sentinels_to_nan(log)
    assert np.isnan(clean).sum() == 2

    # reference-percentile normalization maps W-range onto R-range
    v = np.linspace(20, 120, 50)
    vn = normalize_reference(v, R_min=30, R_max=110, W_min=v.min(), W_max=v.max())
    assert abs(vn.min() - 30) < 1e-9 and abs(vn.max() - 110) < 1e-9

    # additive noise increases RMSE vs the clean target
    rng = np.random.default_rng(1)
    truth = np.sin(np.linspace(0, 6, 300)) * 10 + 50
    noisy = add_gaussian_noise(truth, 0.5, rng)
    e0 = rmse(truth, truth)
    e1 = rmse(truth, noisy)
    print(f"  RMSE clean / noisy     = {e0:.3f} / {e1:.3f}")
    assert e1 > e0 and e0 == 0.0

    # Pearson: feature strongly correlated with target
    feat = truth + rng.normal(0, 1, truth.size)
    print(f"  Pearson(feat, target)  = {pearson(feat, truth):.3f}")
    assert pearson(feat, truth) > 0.9

    # MAE sanity
    assert abs(mae([1, 2, 3], [1, 2, 5]) - (2 / 3)) < 1e-9
    print("  PASS")
    return {"precision": p, "recall": r, "rmse_noisy": e1}


if __name__ == "__main__":
    test_all()
