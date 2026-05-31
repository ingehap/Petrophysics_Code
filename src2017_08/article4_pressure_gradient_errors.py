"""
Article 4: The Impact of Depth and Pressure Measurement Errors on the Estimation
           of Pressure Gradients
Bowers, Schnacke, Hermance (2017)
Reference: Petrophysics Vol. 58, No. 4 (August 2017), pp. 376-396
DOI: none assigned (this issue predates SPWLA DOI assignment)

Fitting a pressure gradient to pretest pressure-depth pairs is biased because
both depth and pressure carry measurement error.  Ordinary least squares of
pressure-on-depth underestimates the gradient while the reciprocal of
depth-on-pressure overestimates it, so the two bracket the truth; orthogonal
regression and a method-of-moments estimator (which subtracts the known
depth-error variance) recover an unbiased gradient.

Implements:

  - Pressure-depth relation  p = p0 + gamma*(d - d0)
  - OLS slope, and the pressure-on-depth / depth-on-pressure bracket
  - Orthogonal (total-least-squares) regression with an error-variance ratio
  - Method-of-moments gradient corrected for the depth-error variance

Note: this issue's PDF has a text layer; the early relations survived in ASCII
while the estimator equations lost their glyphs and are faithful standard-form
reconstructions.  Pressure in psi, depth in ft, gradient in psi/ft.
"""

import numpy as np


# ---------------------------------------------- model --------------

def pressure_depth(p0, gamma, d, d0=0.0):
    """Pressure-depth relation  p = p0 + gamma*(d - d0)  (Eq. 1)."""
    return p0 + gamma * (np.asarray(d, float) - d0)


def ols_slope(x, y):
    """Ordinary-least-squares slope of y on x  (Eqs. 2-3)."""
    return float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0])


def gradient_bracket(depth, pressure):
    """Bracket the gradient: (pressure-on-depth slope, 1/(depth-on-pressure slope))."""
    m_pod = ols_slope(depth, pressure)
    m_dop = ols_slope(pressure, depth)
    return m_pod, 1.0 / m_dop


def orthogonal_regression_slope(depth, pressure, delta=1.0):
    """Orthogonal (total-least-squares) slope with error-variance ratio delta = s2p/s2d (Eq. 4)."""
    x = np.asarray(depth, float)
    y = np.asarray(pressure, float)
    sxx = x.var(ddof=1)
    syy = y.var(ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    return (syy - delta * sxx + np.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy ** 2)) / (2 * sxy)


def method_of_moments_slope(depth, pressure, depth_error_variance):
    """Method-of-moments gradient corrected for the depth-error variance (Eq. 7)

        m = cov(d, p)/(var(d) - sigma_d^2).
    """
    x = np.asarray(depth, float)
    y = np.asarray(pressure, float)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    return sxy / (x.var(ddof=1) - depth_error_variance)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Pressure-Gradient Measurement Errors")
    print("=" * 60)

    # Exact pressure-depth model
    assert np.isclose(pressure_depth(3500.0, 0.30, 8020.0, 8000.0), 3506.0)

    # Synthetic pretests with error in BOTH depth and pressure
    rng = np.random.default_rng(4)
    n = 400
    grad = 0.30
    depth_true = rng.normal(8000.0, 1.0, n)
    press_true = pressure_depth(3500.0, grad, depth_true, 8000.0)
    sigma_d, sigma_p = 0.6, 0.1
    depth = depth_true + rng.normal(0, sigma_d, n)
    press = press_true + rng.normal(0, sigma_p, n)

    # The two OLS slopes bracket the true gradient
    m_pod, m_recip = gradient_bracket(depth, press)
    print(f"  bracket                = {m_pod:.4f} < {grad} < {m_recip:.4f}")
    assert m_pod < grad < m_recip

    # Orthogonal regression lands inside the bracket
    m_or = orthogonal_regression_slope(depth, press, delta=(sigma_p / sigma_d) ** 2)
    assert m_pod < m_or < m_recip

    # Method of moments (knowing the depth-error variance) recovers the gradient
    m_mm = method_of_moments_slope(depth, press, sigma_d ** 2)
    print(f"  method-of-moments      = {m_mm:.4f}")
    assert abs(m_mm - grad) < abs(m_pod - grad) and abs(m_mm - grad) < 0.03
    print("  PASS")
    return {"m_pod": m_pod, "m_recip": m_recip, "m_mm": m_mm}


if __name__ == "__main__":
    test_all()
