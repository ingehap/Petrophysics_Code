"""
Article 5: Synthetic Sonic Log Generation With Machine Learning - A Contest
           Summary From Five Methods
Yu, Xu, Misra, Li, Ashby, et al. (2021)
DOI: 10.30632/PJV62N4-2021a4

Summary of the SPWLA PDDA 2020 contest: predict compressional and shear sonic
logs (DTC, DTS) from seven easy-to-acquire logs (CALI, NPHI, GR, RDEP, RMED,
PEF, RHOB).  Submissions were ranked by RMSE against a blind well; a Random
Forest benchmark scored RMSE = 17.93, and the top five methods improved on it
by ~27%.

Implements:

  - Contest scoring metric: pooled DTC+DTS RMSE                   (Eq. 1)
  - Per-log RMSE and coefficient of determination R^2
  - z-score and min-max feature normalization
  - log10 resistivity transform
  - A simple multivariate linear-regression baseline predictor

Note: the contest equations were image-rendered and not in the text; the
pooled-RMSE form here is the reconstruction consistent with the single
benchmark scalar (17.93).  The five ML models are summarized in the README;
this module implements the scoring + a numpy baseline.  Slowness in us/ft.
"""

import numpy as np

FEATURES = ["CALI", "NPHI", "GR", "RDEP", "RMED", "PEF", "RHOB"]
BENCHMARK_RMSE = 17.93


# ---------------------------------------------- metrics -----------------

def rmse(y_true, y_pred):
    """Single-log root-mean-square error."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d ** 2)))


def pooled_rmse(dtc_true, dtc_pred, dts_true, dts_pred):
    """Contest metric: RMSE over DTC and DTS pooled together (Eq. 1)."""
    dtc_true = np.asarray(dtc_true, float); dtc_pred = np.asarray(dtc_pred, float)
    dts_true = np.asarray(dts_true, float); dts_pred = np.asarray(dts_pred, float)
    se = np.concatenate([(dtc_pred - dtc_true) ** 2, (dts_pred - dts_true) ** 2])
    return float(np.sqrt(np.mean(se)))


def r2_score(y_true, y_pred):
    """Coefficient of determination R^2."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------- preprocessing -----------

def zscore_normalize(x):
    """Standardize to zero mean, unit variance."""
    x = np.asarray(x, float)
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)


def minmax_normalize(x):
    """Scale each column to [0, 1]."""
    x = np.asarray(x, float)
    lo, hi = x.min(axis=0), x.max(axis=0)
    return (x - lo) / (hi - lo + 1e-12)


def log_resistivity(R):
    """log10 transform applied to resistivity features."""
    return np.log10(np.asarray(R, float))


# ---------------------------------------------- baseline model ----------

def fit_linear(X, y):
    """Ordinary least squares with an intercept; returns coefficient vector."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def predict_linear(X, beta):
    return np.column_stack([np.ones(len(X)), X]) @ beta


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Synthetic Sonic Log Generation (ML Contest)")
    print("=" * 60)

    # Synthetic Volve-like dataset with a known feature -> sonic relationship
    rng = np.random.default_rng(0)
    n = 2000
    CALI = rng.uniform(8, 13, n)
    NPHI = rng.uniform(0.05, 0.45, n)
    GR = rng.uniform(20, 150, n)
    RDEP = 10 ** rng.uniform(-0.3, 2.3, n)
    RMED = RDEP * rng.uniform(0.7, 1.0, n)
    PEF = rng.uniform(1.8, 5.5, n)
    RHOB = rng.uniform(2.0, 2.75, n)
    DTC = 50 + 120 * NPHI + 0.05 * GR + rng.normal(0, 2, n)
    DTS = 1.9 * DTC - 10 + rng.normal(0, 2, n)

    X = np.column_stack([CALI, NPHI, GR, log_resistivity(RDEP),
                         log_resistivity(RMED), PEF, RHOB])
    Xn = zscore_normalize(X)
    assert abs(Xn.mean()) < 1e-9 and abs(Xn.std() - 1.0) < 1e-6
    assert minmax_normalize(X).min() >= 0.0 and minmax_normalize(X).max() <= 1.0

    # Train/test split
    ntr = 1500
    b_dtc = fit_linear(Xn[:ntr], DTC[:ntr])
    b_dts = fit_linear(Xn[:ntr], DTS[:ntr])
    dtc_hat = predict_linear(Xn[ntr:], b_dtc)
    dts_hat = predict_linear(Xn[ntr:], b_dts)

    rmse_dtc = rmse(DTC[ntr:], dtc_hat)
    rmse_dts = rmse(DTS[ntr:], dts_hat)
    pooled = pooled_rmse(DTC[ntr:], dtc_hat, DTS[ntr:], dts_hat)
    r2_dtc = r2_score(DTC[ntr:], dtc_hat)
    print(f"  RMSE DTC / DTS         = {rmse_dtc:.2f} / {rmse_dts:.2f} us/ft")
    print(f"  pooled RMSE (Eq. 1)    = {pooled:.2f} us/ft")
    print(f"  R^2 (DTC)              = {r2_dtc:.3f}")
    assert rmse_dtc < 4.0 and r2_dtc > 0.95     # model recovers the relation

    # A constant-mean baseline scores roughly the population std (worse)
    base = pooled_rmse(DTC[ntr:], np.full(n - ntr, DTC[:ntr].mean()),
                       DTS[ntr:], np.full(n - ntr, DTS[:ntr].mean()))
    print(f"  constant-mean baseline = {base:.2f} us/ft")
    assert base > pooled
    print("  PASS")
    return {"rmse_dtc": rmse_dtc, "pooled_rmse": pooled, "r2_dtc": r2_dtc}


if __name__ == "__main__":
    test_all()
