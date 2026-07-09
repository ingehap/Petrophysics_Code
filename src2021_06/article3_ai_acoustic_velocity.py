"""
Article 3: Real-Time Prediction of Acoustic Velocities While Drilling Vertical
           Complex Lithology Using AI Technique
Alsaihati, Elkatatny (2021)
DOI: 10.30632/PJV62N3-2021a2

AI models (ANN / Random Forest) predict compressional and shear velocities
(Vp, Vs) in real time from six surface drilling parameters (flow rate, standpipe
pressure, rotary speed, weight on bit, ROP, torque).  Features are screened with
Spearman correlation and the data cleaned with DBSCAN; models are scored with
AAPE and the correlation coefficient R.

Implements:

  - Spearman rank-order correlation  r_s = 1 - 6*sum(D^2)/(n^3-n)  (Eq. 1)
  - Average absolute percentage error AAPE                         (Eq. 2)
  - Correlation coefficient R                                      (Eq. 3)
  - Min-max feature normalization
  - Empirical Vs-from-Vp correlations (Appendix 1)                 (Eqs. A1.1-A1.9)
  - A numpy linear-regression surrogate for the ANN/RF predictor

Equations transcribed from the rendered article.  Velocities in km/s
(empirical correlations), AAPE in percent.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Eq. 1: Spearman ---------

def _rankdata(a):
    """Average ranks (1-based), ties averaged."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(x, y):
    """Spearman rank-order correlation  r_s = 1 - 6*sum(D^2)/(n^3-n)  (Eq. 1)."""
    rx, ry = _rankdata(x), _rankdata(y)
    D = rx - ry
    n = len(x)
    return 1.0 - 6.0 * np.sum(D ** 2) / (n ** 3 - n)


# ---------------------------------------------- Eqs. 2-3: metrics -------

def aape(y_true, y_pred):
    """Average absolute percentage error (Eq. 2), in percent."""
    return petrolib.ml_stats.mape(y_true, y_pred)


def correlation_coefficient(y_true, y_pred):
    """Correlation coefficient R (Eq. 3)."""
    # The historical body used the textbook n*Sxy form; the canonical
    # centered-sums path agrees to <1e-13 at this article's data scales
    # and is better conditioned against cancellation.
    return petrolib.ml_stats.pearson_r(y_true, y_pred)


# ---------------------------------------------- normalization -----------

def minmax(x, lo=-1.0, hi=1.0):
    """Min-max scale each column into [lo, hi]."""
    return petrolib.ml_stats.minmax(x, axis=0, lo=lo, hi=hi, eps=1e-12)


# ---------------------------------------------- Appendix 1: Vs from Vp --

def vs_pickett_limestone(vp):
    """Pickett (1963) limestone  Vs = Vp/1.9  (Eq. A1.1)."""
    return petrolib.acoustic_geomech.vs_from_vp(vp, method="pickett_ls")


def vs_pickett_dolomite(vp):
    """Pickett (1963) dolomite  Vs = Vp/1.8  (Eq. A1.2)."""
    return petrolib.acoustic_geomech.vs_from_vp(vp, method="pickett_dol")


def vs_carroll(vp):
    """Carroll (1969)  Vs = 0.756090 * Vp^0.81846  (Eq. A1.3).  km/s."""
    return petrolib.acoustic_geomech.vs_from_vp(vp, method="carroll")


def vs_castagna_limestone(vp):
    """Castagna et al. (1985) limestone (Eq. A1.4).  km/s."""
    return petrolib.acoustic_geomech.vs_from_vp(vp, method="castagna_ls")


def vs_brocher(vp):
    """Brocher (2005) regression (Eq. A1.8).  km/s."""
    return petrolib.acoustic_geomech.vs_from_vp(vp, method="brocher")


# ---------------------------------------------- linear surrogate --------

def fit_linear(X, y):
    """Least-squares linear predictor with intercept (ANN/RF surrogate).

    HAZARD (LIBRARY_MERGE_PLAN.md section 9): this article's beta puts the
    INTERCEPT FIRST (the historical design matrix prepended the ones
    column); the canonical ols returns (coef, intercept) separately, so
    the layout is rebuilt explicitly here.  The column reorder changes the
    lstsq float path by <1e-13 relative at this article's scales.
    """
    coef, intercept = petrolib.ml_stats.ols(X, y)
    return np.concatenate([[intercept], coef])


def predict_linear(X, beta):
    beta = np.asarray(beta, float)
    return petrolib.ml_stats.predict_linear(X, beta[1:], float(beta[0]))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Real-Time Acoustic-Velocity Prediction (AI)")
    print("=" * 60)

    # Spearman: perfect monotonic relations give +/-1
    x = np.arange(1, 11.0)
    assert abs(spearman(x, 2 * x + 3) - 1.0) < 1e-9
    assert abs(spearman(x, -x) + 1.0) < 1e-9
    print(f"  Spearman monotonic     = {spearman(x, 2*x+3):.3f}")

    # AAPE worked example: actual [100,200,50], pred [110,180,55] -> 10%
    a = aape([100, 200, 50], [110, 180, 55])
    print(f"  AAPE example           = {a:.2f}%  (expect 10.0)")
    assert abs(a - 10.0) < 1e-9

    # R: perfect linear fit -> 1
    assert abs(correlation_coefficient(x, 3 * x - 1) - 1.0) < 1e-9

    # Empirical Vs correlations at Vp = 5.9 km/s
    print(f"  Pickett ls Vs(5.9)     = {vs_pickett_limestone(5.9):.3f} km/s")
    print(f"  Carroll Vs(5.9)        = {vs_carroll(5.9):.3f} km/s")
    assert abs(vs_pickett_limestone(5.9) - 5.9 / 1.9) < 1e-9
    assert abs(vs_carroll(5.9) - 3.23) < 0.03
    assert vs_brocher(5.9) > 0 and vs_castagna_limestone(5.9) > 0

    # Min-max scaling lands in [-1, 1]
    Xn = minmax(np.random.default_rng(0).uniform(0, 100, (50, 3)))
    assert Xn.min() >= -1.0 - 1e-9 and Xn.max() <= 1.0 + 1e-9

    # Surrogate predictor: recover Vp/Vs from synthetic drilling features
    rng = np.random.default_rng(1)
    n = 800
    Q = rng.uniform(268, 309, n); SPP = rng.uniform(1463, 3344, n)
    RS = rng.uniform(48, 152, n); WOB = rng.uniform(5, 19, n)
    ROP = rng.uniform(3, 72, n); T = rng.uniform(0.6, 8.9, n)
    Vp = 4.2 + 0.01 * RS + 0.1 * T + 0.0003 * SPP + rng.normal(0, 0.05, n)
    X = minmax(np.column_stack([Q, SPP, RS, WOB, ROP, T]))
    beta = fit_linear(X[:600], Vp[:600])
    vp_hat = predict_linear(X[600:], beta)
    err = aape(Vp[600:], vp_hat)
    R = correlation_coefficient(Vp[600:], vp_hat)
    print(f"  surrogate Vp AAPE / R  = {err:.2f}% / {R:.3f}")
    assert err < 5.0 and R > 0.9
    print("  PASS")
    return {"aape_example": a, "vp_aape": err, "vp_R": R}


if __name__ == "__main__":
    test_all()
