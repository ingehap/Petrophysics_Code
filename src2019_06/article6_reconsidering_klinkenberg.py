"""
Article 6: Reconsidering Klinkenberg's Permeability Data
Ruth, Arabjamaloei (2019)
DOI: 10.30632/PJV60N3-2019a5

A re-examination of Klinkenberg's classic gas-slippage permeability data.  The
first-order Klinkenberg model k_app = k_l*(1 + b/Pm) fits most data, but a
second-order term k_app = k_l*(1 + b/Pm + c/Pm^2) better captures the
high-Knudsen (low-pressure) curvature; both are obtained by regressing apparent
permeability against the reciprocal mean pressure.

Implements:

  - First-order Klinkenberg  k_app = k_l*(1 + b/Pm)
  - Second-order model  k_app = k_l*(1 + b/Pm + c/Pm^2)
  - Linear / quadratic regression for k_l, b (and c) vs 1/Pm
  - Residual comparison of the two models

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the journal metadata and these are faithful
standard-form reconstructions of the Klinkenberg slip-flow analysis the paper
reconsiders.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- models ------------------

def klinkenberg(k_l, b, p_mean):
    """First-order Klinkenberg apparent permeability  k_app = k_l*(1 + b/Pm)."""
    return petrolib.flow_transport.klinkenberg_apparent(k_l, b=b, p_mean=p_mean)


def klinkenberg_second_order(k_l, b, c, p_mean):
    """Second-order apparent permeability  k_app = k_l*(1 + b/Pm + c/Pm^2)."""
    return petrolib.flow_transport.klinkenberg_apparent(k_l, b=b, p_mean=p_mean, c2=c)


def fit_first_order(p_mean, k_app):
    """Regress k_app on 1/Pm: intercept = k_l, slope/intercept = b."""
    return petrolib.flow_transport.fit_klinkenberg(p_mean, k_app)


def fit_second_order(p_mean, k_app):
    """Quadratic regression in 1/Pm: returns (k_l, b, c)."""
    x = 1.0 / np.asarray(p_mean, float)
    a2, a1, a0 = np.polyfit(x, np.asarray(k_app, float), 2)
    k_l = a0
    return float(k_l), float(a1 / k_l), float(a2 / k_l)


def rss(y, yhat):
    """Residual sum of squares."""
    return float(np.sum((np.asarray(y, float) - np.asarray(yhat, float)) ** 2))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Reconsidering Klinkenberg's Permeability Data")
    print("=" * 60)

    # First-order: apparent permeability decreases with mean pressure toward k_l
    k_l, b = 5.0, 6.0
    pm = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    k_app = klinkenberg(k_l, b, pm)
    assert np.all(np.diff(k_app) < 0) and np.all(k_app > k_l)

    # Linear regression recovers k_l and b exactly (first-order data)
    kl_fit, b_fit = fit_first_order(pm, k_app)
    print(f"  recovered k_l / b      = {kl_fit:.3f} / {b_fit:.3f}")
    assert abs(kl_fit - k_l) < 1e-9 and abs(b_fit - b) < 1e-9

    # Second-order data: low-pressure curvature that first-order cannot capture
    c = 4.0
    k2 = klinkenberg_second_order(k_l, b, c, pm)
    kl2, b2, c2 = fit_second_order(pm, k2)
    print(f"  2nd-order k_l/b/c      = {kl2:.3f} / {b2:.3f} / {c2:.3f}")
    assert abs(kl2 - k_l) < 1e-6 and abs(c2 - c) < 1e-6

    # The second-order model fits the curved data far better than first-order
    kl_lin, b_lin = fit_first_order(pm, k2)
    rss_first = rss(k2, klinkenberg(kl_lin, b_lin, pm))
    rss_second = rss(k2, klinkenberg_second_order(kl2, b2, c2, pm))
    print(f"  RSS first / second     = {rss_first:.4f} / {rss_second:.2e}")
    assert rss_second < rss_first
    print("  PASS")
    return {"k_l": kl_fit, "b": b_fit, "c": c2,
            "rss_first": rss_first, "rss_second": rss_second}


if __name__ == "__main__":
    test_all()
