"""
Article 8: Comparative Study of Shallow Learning Models for Generating
           Compressional and Shear Traveltime Logs
He, Misra, Li (2018)
DOI: 10.30632/PJV59N6Y2018a7

Where sonic logs are missing, shallow machine-learning models predict the
compressional (DTC) and shear (DTS) traveltime logs from conventional logs (GR,
density, neutron, resistivity).  This module compares two shallow models - a
linear (ordinary least squares) regressor and a k-nearest-neighbor regressor -
scored by the correlation coefficient and RMSE.

Implements:

  - Ordinary-least-squares multivariate regressor
  - k-nearest-neighbor regressor
  - Correlation coefficient R and RMSE
  - Comparison of the two shallow models on DTC / DTS

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the shallow-learning sonic-prediction comparison the paper performs.
"""

import numpy as np


# ---------------------------------------------- models ------------------

def ols_fit(X, y):
    """Ordinary least squares with an intercept."""
    Xb = np.hstack([np.asarray(X, float), np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(Xb, np.asarray(y, float), rcond=None)
    return coef


def ols_predict(coef, X):
    Xb = np.hstack([np.asarray(X, float), np.ones((len(X), 1))])
    return Xb @ coef


def knn_predict(X_train, y_train, X_test, k=8):
    Xtr = np.asarray(X_train, float); ytr = np.asarray(y_train, float)
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr_n = (Xtr - mean) / std
    Xte_n = (np.asarray(X_test, float) - mean) / std
    out = np.zeros(len(Xte_n))
    for i, x in enumerate(Xte_n):
        d = np.linalg.norm(Xtr_n - x, axis=1)
        out[i] = ytr[np.argsort(d)[:k]].mean()
    return out


def correlation(y, yhat):
    return float(np.corrcoef(np.asarray(y, float), np.asarray(yhat, float))[0, 1])


def rmse(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Shallow Learning for Sonic Traveltime Logs")
    print("=" * 60)

    rng = np.random.default_rng(7)
    n = 600
    GR = rng.uniform(20, 150, n); RHOB = rng.uniform(2.0, 2.7, n)
    NPHI = rng.uniform(0.05, 0.40, n); RT = rng.uniform(1, 100, n)
    X = np.column_stack([GR, RHOB, NPHI, np.log10(RT)])
    Xs = (X - X.mean(0)) / X.std(0)
    # DTC (us/ft) ~ rises with porosity, falls with density; mild nonlinearity
    dtc = 70 + 18 * Xs[:, 2] + 12 * (-Xs[:, 1]) + 4 * np.tanh(Xs[:, 0]) \
        + rng.normal(0, 2.0, n)

    ntr = 450
    coef = ols_fit(X[:ntr], dtc[:ntr])
    pred_ols = ols_predict(coef, X[ntr:])
    pred_knn = knn_predict(X[:ntr], dtc[:ntr], X[ntr:], k=8)

    R_ols, e_ols = correlation(dtc[ntr:], pred_ols), rmse(dtc[ntr:], pred_ols)
    R_knn, e_knn = correlation(dtc[ntr:], pred_knn), rmse(dtc[ntr:], pred_knn)
    print(f"  DTC  OLS R/RMSE        = {R_ols:.3f} / {e_ols:.2f}")
    print(f"  DTC  kNN R/RMSE        = {R_knn:.3f} / {e_knn:.2f}")
    assert R_ols > 0.9 and R_knn > 0.85
    assert e_ols < 4.0 and e_knn < 6.0

    # Metric sanity
    assert abs(correlation([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9
    print("  PASS")
    return {"R_ols": R_ols, "R_knn": R_knn, "RMSE_ols": e_ols}


if __name__ == "__main__":
    test_all()
