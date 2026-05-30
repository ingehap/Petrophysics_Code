"""
Article 10: Feasibility and Design of Hydraulic Fracturing Stress Tests Using a
            Quantitative Risk Assessment and Control Approach
Berard, Chugunov, Desroches, Prioul (2019)
DOI: 10.30632/PJV60N1Y2019a9

A hydraulic-fracturing stress test (micro-frac / mini-frac) measures the minimum
in-situ stress from the fracture closure pressure.  The G-function analysis
linearizes the pressure decline after shut-in so the closure pressure is the
departure from the straight line; the design uses a quantitative risk assessment
(probability of a successful, interpretable test).

Implements:

  - Instantaneous shut-in pressure (ISIP) and net pressure
  - G-function pressure-decline model and closure-pressure pick
  - Minimum-stress gradient from the closure pressure
  - Quantitative risk: probability of a successful test

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard mini-frac / G-function stress-test relations the
paper's title describes.
"""

import numpy as np


# ---------------------------------------------- stress test -------------

def net_pressure(isip, min_stress):
    """Net (fracture) pressure  P_net = ISIP - sigma_min."""
    return isip - min_stress


def g_function_decline(G, p0, slope, p_closure, closure_G):
    """Pressure vs G-function: linear leak-off then a steeper post-closure slope.

        P(G) = p0 - slope*G            for G <= closure_G  (before closure)
        P(G) = p_closure - 0.4*slope*(G - closure_G)   after closure
    Returns the modeled pressure (used to test the closure pick).
    """
    G = np.asarray(G, float)
    before = p0 - slope * G
    after = p_closure - 0.4 * slope * (G - closure_G)
    return np.where(G <= closure_G, before, after)


def pick_closure(G, P):
    """Pick closure as the G where the dP/dG slope magnitude drops (curvature).

    The closure is the point of maximum downward change in slope of P vs G.
    """
    G = np.asarray(G, float); P = np.asarray(P, float)
    dPdG = np.gradient(P, G)
    curv = np.gradient(dPdG, G)
    i = int(np.argmax(curv))                      # slope becomes less steep
    return float(G[i]), float(P[i])


def min_stress_gradient(closure_pressure, tvd):
    """Minimum-stress gradient  sigma_min/TVD  (e.g. psi/ft)."""
    return closure_pressure / tvd


def test_success_probability(snr, data_quality, base=0.5):
    """Logistic probability of a successful (interpretable) stress test."""
    z = 1.5 * (snr - 1.0) + 2.0 * (data_quality - 0.5)
    return float(1.0 / (1.0 + np.exp(-z)))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 10: Hydraulic-Fracturing Stress Test (QRA)")
    print("=" * 60)

    # Net pressure is the excess of ISIP over the minimum stress
    assert abs(net_pressure(5200.0, 5000.0) - 200.0) < 1e-9

    # G-function: closure pick recovers the planted closure G/pressure
    G = np.linspace(0, 6, 200)
    p_closure_true, closure_G_true = 5000.0, 2.5
    P = g_function_decline(G, p0=5600.0, slope=240.0,
                           p_closure=p_closure_true, closure_G=closure_G_true)
    Gc, Pc = pick_closure(G, P)
    print(f"  closure G / P          = {Gc:.2f} / {Pc:.0f}  (true {closure_G_true} / {p_closure_true:.0f})")
    assert abs(Gc - closure_G_true) < 0.2 and abs(Pc - p_closure_true) < 80.0

    # Minimum-stress gradient from the closure pressure
    grad = min_stress_gradient(Pc, 8000.0)
    print(f"  min-stress gradient    = {grad:.3f} psi/ft")
    assert 0.5 < grad < 1.0

    # QRA: a high-SNR, high-quality test is very likely to succeed
    p_good = test_success_probability(2.0, 0.9)
    p_poor = test_success_probability(0.8, 0.3)
    print(f"  success prob good/poor = {p_good:.2f} / {p_poor:.2f}")
    assert p_good > 0.8 and p_poor < 0.5
    print("  PASS")
    return {"closure_G": Gc, "min_stress_grad": float(grad), "p_success": p_good}


if __name__ == "__main__":
    test_all()
