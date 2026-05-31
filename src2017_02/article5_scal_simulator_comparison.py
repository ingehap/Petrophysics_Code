"""
Article 5: Comparison of Four Numerical Simulators for SCAL Experiments
Lenormand, Lorentzen, Maas, Ruth (2017)
Reference: Petrophysics Vol. 58, No. 1 (February 2017), pp. 48-56
DOI: none assigned (this issue predates SPWLA DOI assignment)

Four 1D two-phase core-flood SCAL simulators are benchmarked on synthetic
steady-state, unsteady-state (Buckley-Leverett), and centrifuge cases.  The
agreement hinges on consistently treating the capillary pressure (Pc = P_oil -
P_water) and the end-piece boundary condition.  This module implements the shared
physics: Corey relative permeabilities, the fractional-flow function and its
Buckley-Leverett front, the Darcy pressure drop, and the capillary convention.

Implements:

  - Capillary pressure convention  Pc = P_oil - P_water
  - Corey relative permeabilities and the water fractional flow
  - Buckley-Leverett fractional-flow derivative (front saturation)
  - Darcy pressure drop across the plug

Note: this issue's PDF has a text layer; only Pc = P_oil - P_water survived as a
display equation, so the rest are standard-form reconstructions of the two-phase
flow physics the simulators share.  SI units.
"""

import numpy as np


# ---------------------------------------------- capillary / kr --------------

def capillary_pressure(p_oil, p_water):
    """Capillary pressure convention  Pc = P_oil - P_water."""
    return p_oil - p_water


def corey_kr(sw, swc, sor, krw_max, kro_max, nw, no):
    """Corey water/oil relative permeabilities, returns (krw, kro)."""
    swn = np.clip((np.asarray(sw, float) - swc) / (1.0 - swc - sor), 0.0, 1.0)
    return krw_max * swn ** nw, kro_max * (1.0 - swn) ** no


def fractional_flow(sw, swc, sor, krw_max, kro_max, nw, no, mu_w=1e-3, mu_o=2e-3):
    """Water fractional flow  fw = (krw/mu_w)/(krw/mu_w + kro/mu_o)."""
    krw, kro = corey_kr(sw, swc, sor, krw_max, kro_max, nw, no)
    lam_w = krw / mu_w
    lam_o = kro / mu_o
    return np.where(lam_w + lam_o > 0, lam_w / (lam_w + lam_o), 0.0)


def fractional_flow_derivative(sw, swc, sor, krw_max, kro_max, nw, no, mu_w=1e-3, mu_o=2e-3):
    """Numerical dfw/dSw (the Buckley-Leverett wave speed)."""
    sw = np.asarray(sw, float)
    fw = fractional_flow(sw, swc, sor, krw_max, kro_max, nw, no, mu_w, mu_o)
    return np.gradient(fw, sw)


def darcy_pressure_drop(q, mu, kr, k, area, length):
    """Darcy pressure drop across the plug  dP = q*mu*L/(k*kr*A)."""
    return q * mu * length / (k * kr * area)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: SCAL Simulator Comparison")
    print("=" * 60)

    # Capillary-pressure convention
    assert np.isclose(capillary_pressure(1.5e5, 1.0e5), 5.0e4)

    swc, sor = 0.15, 0.25
    sw = np.linspace(swc, 1 - sor, 50)
    krw, kro = corey_kr(sw, swc, sor, 0.4, 1.0, 2.0, 3.0)
    assert np.all(np.diff(krw) >= 0) and np.all(np.diff(kro) <= 0)

    # Fractional flow runs 0 -> 1 and is monotonic in water saturation
    fw = fractional_flow(sw, swc, sor, 0.4, 1.0, 2.0, 3.0)
    print(f"  fw range               = {fw[0]:.3f} -> {fw[-1]:.3f}")
    assert np.isclose(fw[0], 0.0) and np.isclose(fw[-1], 1.0) and np.all(np.diff(fw) >= 0)

    # Buckley-Leverett wave speed peaks at an intermediate saturation
    dfw = fractional_flow_derivative(sw, swc, sor, 0.4, 1.0, 2.0, 3.0)
    print(f"  max dfw/dSw at Sw       = {sw[int(np.argmax(dfw))]:.3f}")
    assert dfw.max() > 0 and swc < sw[int(np.argmax(dfw))] < 1 - sor

    # Darcy pressure drop is positive and falls as permeability rises
    dp = darcy_pressure_drop(1e-7, 1e-3, 0.4, 1e-13, 1e-4, 0.05)
    assert dp > 0 and darcy_pressure_drop(1e-7, 1e-3, 0.4, 1e-12, 1e-4, 0.05) < dp
    print("  PASS")
    return {"fw_mid": float(fw[len(fw) // 2]), "dP": float(dp)}


if __name__ == "__main__":
    test_all()
