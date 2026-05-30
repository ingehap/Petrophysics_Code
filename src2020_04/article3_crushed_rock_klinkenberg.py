"""
Article 3: Low-Permeability Measurement on Crushed Rock: Insights
Profice, Lenormand (2020)
DOI: 10.30632/PJV61N2-2020a3

Crushed-rock (GRI) gas permeability and core-plug pressure-decay methods are
compared on tight rocks (1 to 50 nD).  Apparent gas permeability is corrected
for Klinkenberg gas slippage and extrapolated to the liquid-equivalent
(Klinkenberg) permeability by plotting apparent permeability against the inverse
mean pressure.

Implements:

  - Klinkenberg apparent permeability  k_app = k_l*(1 + b/Pm)    (Eq. 3)
  - Klinkenberg extrapolation (linear fit vs 1/Pm) -> k_l, b
  - Slip-factor ratio between two gases  b ~ mu*sqrt(1/M)        (Eqs. 1-2)
  - Mean pore pressure  Pm = (P1 + P2)/2                         (Eq. 4)

Note: this issue's PDF text layer kept the equation numbers and definitions but
dropped the typeset glyphs, so these are the standard Klinkenberg forms anchored
to those definitions.  Paper anchor reproduced: theoretical He/N2 slip-factor
ratio = 2.9; absolute permeability range 1-50 nD.
"""

import numpy as np

# Gas properties: dynamic viscosity (Pa.s) and molar mass (g/mol) at lab T
GAS = {
    "He": {"mu": 1.96e-5, "M": 4.003},
    "N2": {"mu": 1.78e-5, "M": 28.013},
}


# ---------------------------------------------- Klinkenberg ------------

def klinkenberg_kapp(k_l, b, p_mean):
    """Apparent gas permeability  k_app = k_l*(1 + b/Pm)  (Eq. 3)."""
    return k_l * (1.0 + b / np.asarray(p_mean, float))


def mean_pressure(p1, p2):
    """Mean pore pressure  Pm = (P1 + P2)/2  (Eq. 4)."""
    return 0.5 * (np.asarray(p1, float) + np.asarray(p2, float))


def klinkenberg_extrapolate(p_mean, k_app):
    """Fit k_app = k_l + (k_l*b)*(1/Pm); return (k_l, b).

    k_l is the intercept (1/Pm -> 0, liquid-equivalent permeability), and the
    slope equals k_l*b so b = slope/intercept.
    """
    x = 1.0 / np.asarray(p_mean, float)
    slope, intercept = np.polyfit(x, np.asarray(k_app, float), 1)
    k_l = intercept
    b = slope / intercept
    return float(k_l), float(b)


def slip_factor_ratio(gas1="He", gas2="N2"):
    """Theoretical slip-factor ratio  b1/b2 = (mu1/mu2)*sqrt(M2/M1)  (Eqs. 1-2).

    The slip factor scales with the mean free path (proportional to viscosity
    and inversely to sqrt(molar mass)).
    """
    g1, g2 = GAS[gas1], GAS[gas2]
    return (g1["mu"] / g2["mu"]) * np.sqrt(g2["M"] / g1["M"])


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Klinkenberg Permeability on Crushed Rock")
    print("=" * 60)

    # Apparent permeability exceeds liquid permeability and falls with pressure
    k_l_true, b_true = 10e-9, 8.0      # 10 nD, slip factor 8 (units of pressure)
    pm = np.array([2.0, 4.0, 8.0, 16.0])
    k_app = klinkenberg_kapp(k_l_true, b_true, pm)
    print(f"  k_app @Pm=2/16         = {k_app[0]*1e9:.1f} / {k_app[-1]*1e9:.1f} nD")
    assert np.all(np.diff(k_app) < 0)         # decreasing with mean pressure
    assert np.all(k_app > k_l_true)           # always above the liquid value

    # Extrapolation recovers planted k_l and b from the (1/Pm, k_app) line
    k_l, b = klinkenberg_extrapolate(pm, k_app)
    print(f"  recovered k_l / b      = {k_l*1e9:.2f} nD / {b:.2f}")
    assert abs(k_l - k_l_true) < 1e-12 and abs(b - b_true) < 1e-9

    # Mean pressure helper
    assert abs(mean_pressure(3.0, 5.0) - 4.0) < 1e-12

    # Theoretical He/N2 slip-factor ratio reproduces the paper's 2.9
    ratio = slip_factor_ratio("He", "N2")
    print(f"  slip-factor ratio He/N2 = {ratio:.2f}  (expect ~2.9)")
    assert abs(ratio - 2.9) < 0.15
    print("  PASS")
    return {"k_l_nD": k_l * 1e9, "b": b, "He_N2_ratio": float(ratio)}


if __name__ == "__main__":
    test_all()
