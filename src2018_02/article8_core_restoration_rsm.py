"""
Article 8: Application of an Optimization Method for the Restoration of Core
           Samples for SCAL Experiments
Sripal, James (2018)
DOI: 10.30632/petro_059_1_a7

Wettability restoration of core for special core analysis is optimized with a
Box-Behnken response-surface design over three factors (brine salinity,
restoration temperature, aging time) against wettability responses (contact
angle, USBM index).  This module implements the USBM wettability index, the
Box-Behnken design, a second-order response-surface least-squares fit, and the
optimization of the fitted surface.

Implements:

  - USBM wettability index  W = log10(A1/A2)
  - Box-Behnken design for three factors
  - Second-order (quadratic) response-surface fit by least squares
  - Optimization of the fitted response over the factor box

Note: this issue's PDF has a text layer; the USBM index is literally present,
while the response-surface polynomial coefficients were in dropped display
equations / truncated tables and are not recoverable - so the RSM fit here is
the standard Box-Behnken quadratic least-squares procedure the paper applies.
Factors are coded to [-1, 1].
"""

import numpy as np


# ---------------------------------------------- USBM --------------

def usbm_index(area_drainage, area_imbibition):
    """USBM wettability index  W = log10(A1/A2).

    A1 = area under the secondary-drainage Pc curve, A2 = area under the primary-
    imbibition curve.  W > 0 water-wet, W < 0 oil-wet, ~0 neutral.
    """
    return np.log10(area_drainage / area_imbibition)


# ---------------------------------------------- response surface --------------

def box_behnken_3():
    """Box-Behnken design for 3 coded factors: 12 edge points + center."""
    pts = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for a in (-1, 1):
            for b in (-1, 1):
                row = [0, 0, 0]
                row[i], row[j] = a, b
                pts.append(row)
    pts.append([0, 0, 0])
    return np.array(pts, float)


def _design_matrix(x):
    """Full quadratic terms: 1, x1, x2, x3, x1x2, x1x3, x2x3, x1^2, x2^2, x3^2."""
    x = np.atleast_2d(np.asarray(x, float))
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    return np.column_stack([np.ones_like(x1), x1, x2, x3,
                            x1 * x2, x1 * x3, x2 * x3,
                            x1 ** 2, x2 ** 2, x3 ** 2])


def fit_response(factors, response):
    """Least-squares fit of the quadratic response surface; returns coefficients."""
    beta, *_ = np.linalg.lstsq(_design_matrix(factors), np.asarray(response, float), rcond=None)
    return beta


def predict_response(beta, factors):
    """Predict the response from fitted coefficients."""
    return _design_matrix(factors) @ beta


def optimize_response(beta, n=21, maximize=True):
    """Grid-search the coded factor box [-1,1]^3 for the optimum response."""
    g = np.linspace(-1, 1, n)
    grid = np.array([[a, b, c] for a in g for b in g for c in g])
    vals = predict_response(beta, grid)
    idx = int(np.argmax(vals) if maximize else np.argmin(vals))
    return grid[idx], float(vals[idx])


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Core Restoration RSM")
    print("=" * 60)

    # USBM index sign: more drainage area -> water-wet (W > 0)
    assert usbm_index(2.0, 1.0) > 0 and usbm_index(1.0, 2.0) < 0

    # Box-Behnken design has 13 runs for 3 factors, all within the coded box
    design = box_behnken_3()
    print(f"  Box-Behnken runs       = {len(design)}")
    assert design.shape == (13, 3) and np.all(np.abs(design) <= 1)

    # Fit a known concave quadratic and check the surface is reproduced
    true_beta = np.array([5.0, 2.0, -3.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
    y = predict_response(true_beta, design)
    beta = fit_response(design, y)
    assert np.allclose(predict_response(beta, design), y, atol=1e-8)

    # Optimization moves toward the analytic optimum (x1->+1, x2->-1, x3->+0.5)
    xopt, fopt = optimize_response(beta, n=21, maximize=True)
    print(f"  optimum factors        = {np.array2string(xopt, precision=2)}")
    assert xopt[0] > 0.5 and xopt[1] < -0.5 and abs(xopt[2] - 0.5) < 0.2
    print("  PASS")
    return {"opt_factors": xopt.tolist(), "opt_value": fopt}


if __name__ == "__main__":
    test_all()
