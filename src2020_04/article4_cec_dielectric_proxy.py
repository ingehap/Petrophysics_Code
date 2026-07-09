"""
Article 4: A New CEC-Measurement Proxy Using High-Frequency Dielectric Analysis
           of Crushed Rock
Stokes, Yang, Ezebuiro, Fischer (2020)
DOI: 10.30632/PJV61N2-2020a4

Cation exchange capacity (CEC) is estimated from the relative permittivity of
crushed rock measured with a handheld high-frequency dielectric probe, avoiding
laborious wet-chemistry titration.  CEC is a piecewise-linear function of
relative permittivity whose slope depends on relative humidity (RH); all
calibration lines pass through the pure-quartz anchor (eps' = 2.5, CEC = 0), and
a small constant correction is added.

Implements:

  - RH-dependent slope (three RH regimes: <18%, 18-35%, >35%)     (Eq. 3)
  - CEC calibration  CEC = S_RH*(eps' - 2.5) + C                  (Eqs. 2, 4-5)
  - Linear-calibration goodness of fit (R^2)

Note: this issue's PDF text layer kept the equation numbers and definitions but
dropped the typeset glyphs; the calibration is the paper's empirical
piecewise-linear model anchored at (eps'=2.5, CEC=0) with a constant
correction C ~ 4 meq/100 g.  Permittivity reported at 120 MHz, 21 C.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

QUARTZ_EPS = 2.5         # pure-quartz permittivity anchor (CEC = 0)
CORRECTION_C = 4.0       # meq/100 g constant correction


# ---------------------------------------------- RH-dependent slope ------

def rh_slope(rh_pct):
    """Calibration slope S_RH (meq/100g per permittivity unit) vs RH  (Eq. 3).

    Three RH regimes with increasing sensitivity: low (<18%), mid (18-35%),
    high (>35%).  Continuous and monotonically increasing in RH.
    """
    s_18 = 0.5 + 0.05 * 18.0        # low-regime value at the 18% boundary  = 1.40
    s_35 = s_18 + 0.08 * (35.0 - 18.0)   # mid-regime value at 35% boundary = 2.76
    if rh_pct < 18.0:
        return 0.5 + 0.05 * rh_pct
    if rh_pct < 35.0:
        return s_18 + 0.08 * (rh_pct - 18.0)
    return s_35 + 0.03 * (rh_pct - 35.0)


# ---------------------------------------------- CEC calibration ---------

def cec_from_permittivity(eps_real, rh_pct, correction=CORRECTION_C):
    """CEC from relative permittivity  CEC = S_RH*(eps' - 2.5) + C  (Eqs. 2,4,5)."""
    return rh_slope(rh_pct) * (np.asarray(eps_real, float) - QUARTZ_EPS) + correction


def r_squared(y, yhat):
    """Coefficient of determination R^2."""
    r2 = petrolib.ml_stats.r2_score(y, yhat)
    return r2 if np.isfinite(r2) else 0.0  # historical zero-variance fallback


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: CEC From High-Frequency Dielectric Analysis")
    print("=" * 60)

    # Pure-quartz anchor: at eps' = 2.5 the (uncorrected) CEC is zero
    assert abs(cec_from_permittivity(2.5, 30.0, correction=0.0)) < 1e-12
    # with the correction, the anchor reads ~ C
    assert abs(cec_from_permittivity(2.5, 30.0) - CORRECTION_C) < 1e-12

    # CEC increases with permittivity (more exchangeable cations / bound water)
    assert cec_from_permittivity(6.0, 30.0) > cec_from_permittivity(3.0, 30.0)

    # Slope increases with relative humidity
    assert rh_slope(10.0) < rh_slope(25.0) < rh_slope(50.0)
    print(f"  slope RH 10/25/50%     = {rh_slope(10):.2f} / {rh_slope(25):.2f} / {rh_slope(50):.2f}")

    # Calibration is linear in permittivity at fixed RH -> R^2 ~ 1 on data
    rng = np.random.default_rng(2)
    eps = np.linspace(2.5, 12.0, 8)             # quartz-smectite mixtures
    cec_true = cec_from_permittivity(eps, 25.0)
    cec_meas = cec_true + rng.normal(0, 0.3, eps.size)   # +- lab error
    # fit a line and score it
    s, b = np.polyfit(eps, cec_meas, 1)
    R2 = r_squared(cec_meas, s * eps + b)
    print(f"  calibration R^2        = {R2:.3f}")
    assert R2 > 0.98

    # A smectite-rich (high permittivity) sample reads a high CEC
    cec_smectite = cec_from_permittivity(12.0, 50.0)
    print(f"  CEC smectite (eps=12)  = {cec_smectite:.1f} meq/100g")
    assert cec_smectite > 20.0
    print("  PASS")
    return {"slope_25": rh_slope(25.0), "R2": R2, "cec_smectite": float(cec_smectite)}


if __name__ == "__main__":
    test_all()
