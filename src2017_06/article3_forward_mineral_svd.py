"""
Article 3: Forward Mineral Modeling Using Regularized Least-Squares Regression
           With Singular Value Decomposition: Case Study From Qusaiba Shale
Xu, McCormick, Herron, Cheshire, Al-Salim, Almarzouq (2017)
Reference: Petrophysics Vol. 58, No. 3 (June 2017), pp. 242-269
DOI: none assigned (this issue predates SPWLA DOI assignment)

Mineral abundances are predicted as a linear combination of organic-free major
elements by a calibrated forward model M = E*x + N.  The coefficient matrix x is
solved by least squares, stabilized with a truncated singular-value
decomposition (drop the smallest singular values) and L2 (ridge) regularization
to suppress noise from near-collinear elements.

Implements:

  - Organic-free elemental correction  E_of = E/(1 - 1.2*TOC/100)
  - Forward model fit  M = E*x  by ordinary least squares
  - Truncated-SVD pseudoinverse solution
  - Ridge (L2) solution  x = (E^T E + lambda I)^-1 E^T M  and condition number

Note: this issue's PDF has a text layer; Eq. 1 survived, while the SVD/ridge
relations lost their glyphs and are faithful standard-form reconstructions.  The
reported optimal ridge lambda = 0.90 is the default.  Elements/minerals in wt%.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- preprocessing --------------

def organic_free_correction(element, toc_pct):
    """Organic-free elemental concentration  E_of = E/(1 - 1.2*TOC/100)  (Eq. 1)."""
    return np.asarray(element, float) / (1.0 - 1.2 * np.asarray(toc_pct, float) / 100.0)


# ---------------------------------------------- regression --------------

def fit_ols(elements, minerals):
    """Ordinary-least-squares coefficient matrix x for  M = E*x  (Eq. forward model)."""
    x, *_ = np.linalg.lstsq(np.asarray(elements, float), np.asarray(minerals, float), rcond=None)
    return x


def fit_svd(elements, minerals, n_drop=0):
    """Truncated-SVD solution: drop the n_drop smallest singular values (Eqs. 2-3)."""
    e = np.asarray(elements, float)
    u, s, vt = np.linalg.svd(e, full_matrices=False)
    s_inv = 1.0 / s
    if n_drop > 0:
        s_inv[-n_drop:] = 0.0
    pinv = vt.T @ np.diag(s_inv) @ u.T
    return pinv @ np.asarray(minerals, float)


def fit_ridge(elements, minerals, lam=0.90):
    """Ridge (L2) solution  x = (E^T E + lambda I)^-1 E^T M  (Eqs. 4-5)."""
    return petrolib.inversion_numerics.linear.tikhonov_solve(
        elements, minerals, lam)


def condition_number(elements):
    """Condition number of the elemental matrix  = sigma_max/sigma_min."""
    return petrolib.inversion_numerics.linear.condition_number(elements)


def predict(elements, x):
    """Predict mineral abundances  M = E*x."""
    return np.asarray(elements, float) @ x


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Forward Mineral Modeling (SVD/ridge)")
    print("=" * 60)

    # Organic-free correction raises a measured element
    assert organic_free_correction(20.0, 5.0) > 20.0

    # Synthetic calibration: M = E*x_true exactly, OLS recovers the fit
    rng = np.random.default_rng(6)
    E = rng.uniform(0, 50, (60, 8))
    x_true = rng.uniform(-0.5, 1.0, (8, 9))
    M = E @ x_true
    x_ols = fit_ols(E, M)
    print(f"  OLS prediction error   = {np.abs(predict(E, x_ols) - M).mean():.2e}")
    assert np.allclose(predict(E, x_ols), M, atol=1e-6)

    # SVD and ridge both predict well; ridge shrinks the coefficient norm
    x_svd = fit_svd(E, M, n_drop=0)
    assert np.allclose(predict(E, x_svd), M, atol=1e-6)
    x_ridge = fit_ridge(E, M, lam=0.90)
    assert np.linalg.norm(x_ridge) < np.linalg.norm(x_ols)

    # Condition number is finite and > 1
    cn = condition_number(E)
    print(f"  condition number       = {cn:.1f}")
    assert cn > 1.0
    print("  PASS")
    return {"condition_number": cn}


if __name__ == "__main__":
    test_all()
