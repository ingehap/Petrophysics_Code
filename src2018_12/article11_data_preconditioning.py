"""
Article 11: Data Preconditioning for Predictive and Interpretive Algorithms:
            Importance in Data-Driven Analytics and Methods for Application
Frost, Quinn (2018)
DOI: 10.30632/PJV59N6Y2018a10

Data-driven petrophysics depends on the quality of the input data: raw well logs
contain outliers, gaps and differing scales that degrade predictive and
interpretive algorithms.  Preconditioning - outlier removal, gap imputation and
feature scaling - measurably improves downstream model performance.

Implements:

  - Z-score and min-max feature scaling
  - Outlier detection (z-score and IQR rules) and removal
  - Gap imputation (interpolation)
  - Downstream-model improvement from preconditioning (RMSE)

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard data-preconditioning steps the paper recommends.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- scaling -----------------

def zscore(x):
    return petrolib.ml_stats.zscore(x, eps=1e-12)


def minmax(x):
    return petrolib.ml_stats.minmax(x, eps=1e-12)


# ---------------------------------------------- cleaning ----------------

def zscore_outliers(x, thresh=3.0):
    """Boolean mask of z-score outliers (|z| > thresh)."""
    return np.abs(zscore(x)) > thresh


def iqr_outliers(x, k=1.5):
    """Boolean mask of IQR outliers (outside Q1 - k*IQR, Q3 + k*IQR)."""
    x = np.asarray(x, float)
    q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def impute_gaps(x):
    """Fill NaN gaps by linear interpolation over the index."""
    x = np.asarray(x, float).copy()
    idx = np.arange(len(x)); good = ~np.isnan(x)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 11: Data Preconditioning for Data-Driven Analytics")
    print("=" * 60)

    # Scaling: z-score -> ~0 mean/unit std; min-max -> [0,1]
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert abs(zscore(x).mean()) < 1e-9 and abs(zscore(x).std() - 1.0) < 1e-9
    mm = minmax(x); assert abs(mm.min()) < 1e-12 and abs(mm.max() - 1.0) < 1e-12

    # Outlier detection flags a planted spike
    data = np.r_[np.full(50, 5.0) + 0.1 * np.random.default_rng(0).standard_normal(50), [50.0]]
    assert zscore_outliers(data)[-1] and iqr_outliers(data)[-1]
    assert zscore_outliers(data)[:-1].sum() == 0

    # Imputation fills NaN gaps by interpolation
    g = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    filled = impute_gaps(g)
    print(f"  imputed gaps           = {np.array2string(filled, precision=1)}")
    assert not np.any(np.isnan(filled)) and abs(filled[1] - 2.0) < 1e-9

    # Downstream improvement: a sentinel value (-999) wrecks a fit until cleaned
    rng = np.random.default_rng(11)
    n = 200
    feat = rng.uniform(0, 10, n)
    target = 2.0 * feat + 1.0 + rng.normal(0, 0.5, n)
    raw = feat.copy(); raw[::25] = -999.0             # sentinel "no-data" spikes

    def fit_rmse(xf):
        A = np.vstack([xf, np.ones_like(xf)]).T
        c, *_ = np.linalg.lstsq(A, target, rcond=None)
        return float(np.sqrt(np.mean((A @ c - target) ** 2)))

    rmse_raw = fit_rmse(raw)
    clean = raw.copy(); clean[zscore_outliers(raw)] = np.nan
    clean = impute_gaps(np.where(clean == -999.0, np.nan, clean))
    rmse_clean = fit_rmse(clean)
    print(f"  RMSE raw / preconditioned = {rmse_raw:.2f} / {rmse_clean:.2f}")
    assert rmse_clean < rmse_raw
    print("  PASS")
    return {"rmse_raw": rmse_raw, "rmse_clean": rmse_clean}


if __name__ == "__main__":
    test_all()
