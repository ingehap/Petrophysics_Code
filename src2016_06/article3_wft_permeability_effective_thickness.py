"""
Article 3: Permeability Interpretation from Wireline Formation Testing
           Measurements with Consideration of Effective Thickness
Yang, Yang (2016)
Reference: Petrophysics Vol. 57, No. 3 (June 2016), pp. 251-269
DOI: none assigned (this issue predates SPWLA DOI assignment)

A near-wellbore numerical model simulates the wireline-formation-tester (WFT)
pumpout and buildup to study how "effective thickness" (the formation actually
sensed within tool resolution) controls the interpreted permeability.  A
Brooks-Corey relative-permeability / capillary-pressure model drives the
two-phase flow; permeability is read from the probe drawdown (spherical flow)
and from the pressure-derivative flow regimes (spherical -1/2 slope, then radial
when bounded by both boundaries).

Implements:

  - Brooks-Corey relative permeability and capillary pressure (Eqs. 1-3)
  - Single-probe spherical drawdown permeability (Moran-Finklea form)
  - Spherical permeability  ks = (kh^2 * kv)^(1/3)
  - Pressure-derivative flow-regime slope (spherical vs radial)

Note: this issue is largely a numerical-simulation/sensitivity study; the
Brooks-Corey relations (Eqs. 1-3) are transcribed from the body, while the WFT
drawdown/flow-regime relations are the standard formation-tester forms (Moran &
Finklea, 1962) the paper relies on.  Permeability in mD, pressures in psi, rates
in cm^3/s, viscosity in cP.
"""

import numpy as np


# ---------------------------------------------- Brooks-Corey --------------

def normalized_water_saturation(sw, swi, sor):
    """Normalized (effective) water saturation (Eq. 1)

        Sw* = (Sw - Swi)/(1 - Swi - Sor).
    """
    return (sw - swi) / (1.0 - swi - sor)


def krw(sw_star, krw0, ew):
    """Brooks-Corey water relative permeability  krw = krw0 * Sw*^ew  (Eq. 2a)."""
    return krw0 * np.asarray(sw_star, float) ** ew


def kro(sw_star, kro0, eo):
    """Brooks-Corey oil relative permeability  kro = kro0 * (1 - Sw*)^eo  (Eq. 2b)."""
    return kro0 * (1.0 - np.asarray(sw_star, float)) ** eo


def capillary_pressure(sw_star, pc0, ep):
    """Brooks-Corey capillary pressure  Pc = Pc0 * Sw*^(-ep)  (Eq. 3)."""
    return pc0 * np.asarray(sw_star, float) ** (-ep)


# ---------------------------------------------- WFT permeability --------------

def drawdown_permeability(flow_rate, viscosity, dp, probe_radius, geom_factor=1.0):
    """Single-probe spherical drawdown permeability (Moran & Finklea, 1962)

        k = C * q * mu / (r_probe * dp),

    with C a probe geometric factor; the steady drawdown relates flow rate,
    fluid viscosity and the probe pressure drop to permeability.
    """
    return geom_factor * flow_rate * viscosity / (probe_radius * dp)


def spherical_permeability(kh, kv):
    """Spherical permeability  ks = (kh^2 * kv)^(1/3),

    the geometric mean sensed by an early-time spherical-flow WFT response in an
    anisotropic formation.
    """
    return (kh ** 2 * kv) ** (1.0 / 3.0)


def pressure_derivative_slope(times, pressures):
    """Log-log pressure-derivative slope used for flow-regime diagnosis.

    Spherical flow gives a slope near -1/2; radial flow (bounded by both
    boundaries) flattens toward 0.  Returns the slope of log10(dP/dlnt) vs
    log10(t) over the supplied (late-time) window.
    """
    t = np.asarray(times, float)
    p = np.asarray(pressures, float)
    dpdlnt = np.gradient(p, np.log(t))
    good = dpdlnt > 0
    slope, _ = np.polyfit(np.log10(t[good]), np.log10(dpdlnt[good]), 1)
    return float(slope)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: WFT Permeability & Effective Thickness")
    print("=" * 60)

    # Normalized saturation maps [Swi, 1-Sor] -> [0, 1]
    assert np.isclose(normalized_water_saturation(0.2, 0.2, 0.2), 0.0)
    assert np.isclose(normalized_water_saturation(0.8, 0.2, 0.2), 1.0)

    # Brooks-Corey endpoints: krw rises, kro falls with Sw*
    assert np.isclose(krw(1.0, 0.4, 3.0), 0.4) and np.isclose(krw(0.0, 0.4, 3.0), 0.0)
    assert np.isclose(kro(0.0, 0.9, 2.0), 0.9) and np.isclose(kro(1.0, 0.9, 2.0), 0.0)
    # Capillary pressure diverges as Sw* -> 0 (entry) and is finite at Sw* = 1
    assert capillary_pressure(0.1, 5.0, 0.5) > capillary_pressure(1.0, 5.0, 0.5)

    # Drawdown permeability scales with rate*viscosity / (probe radius * dp)
    k = drawdown_permeability(flow_rate=10.0, viscosity=0.5, dp=20.0,
                              probe_radius=0.5, geom_factor=1.0)
    print(f"  drawdown k             = {k:.3f} (consistent units)")
    assert k > 0 and drawdown_permeability(20.0, 0.5, 20.0, 0.5) > k

    # Spherical permeability is the geometric mean of kh, kv
    ks = spherical_permeability(kh=100.0, kv=25.0)
    print(f"  spherical k            = {ks:.1f} mD")
    assert np.isclose(ks, (100.0 ** 2 * 25.0) ** (1.0 / 3.0))
    assert np.isclose(spherical_permeability(50.0, 50.0), 50.0)

    # Pressure-derivative slope is ~ -1/2 for a synthetic spherical-flow response
    t = np.logspace(-2, 0, 40)
    p = 100.0 - 5.0 * t ** (-0.5)              # dP/dlnt ~ t^(-1/2)
    slope = pressure_derivative_slope(t, p)
    print(f"  derivative slope       = {slope:.2f}")
    assert -0.7 < slope < -0.3
    print("  PASS")
    return {"ks": float(ks), "slope": slope}


if __name__ == "__main__":
    test_all()
