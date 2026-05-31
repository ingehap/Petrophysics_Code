"""
Article 3: An Analytical Model for Analysis of Capillary Pressure Measurements
           by Centrifuge
Andersen, Skjaeveland, Standnes (2017)
Reference: Petrophysics Vol. 58, No. 4 (August 2017), pp. 366-375
DOI: none assigned (this issue predates SPWLA DOI assignment)

An analytical model for primary-drainage centrifuge capillary-pressure
experiments: the equilibrium capillary pressure at the inner face follows the
Hassler-Brunner relation, a minimum (critical) rotation speed is needed to start
flow against the threshold pressure, and the average saturation approaches its
equilibrium value exponentially with a characteristic time set by the rock
mobility.  Corey relative permeabilities and a capillary-pressure correlation
close the model.

Implements:

  - Hassler-Brunner inner-face capillary pressure  Pc = 0.5*drho*omega^2*(r2^2 - r1^2)
  - Critical rotation speed to overcome the threshold pressure
  - Exponential saturation history  Sw(t) = Sw_eq + (Sw_init - Sw_eq)*exp(-t/tau)
  - Corey relative permeabilities and a capillary-pressure correlation

Note: this issue's PDF has a text layer but the typeset display equations were
dropped, so the relations are faithful standard-form reconstructions from the
surviving remarks (the Hassler-Brunner form and the exponential Sw(t) are named
explicitly).  SI: omega in rad/s, radii in m, density in kg/m^3, Pc in Pa.
"""

import numpy as np


# ---------------------------------------------- equilibrium --------------

def hassler_brunner_pc(delta_rho, omega, r1, r2):
    """Inner-face equilibrium capillary pressure  Pc = 0.5*drho*omega^2*(r2^2 - r1^2)."""
    return 0.5 * delta_rho * omega ** 2 * (r2 ** 2 - r1 ** 2)


def critical_speed(p_threshold, delta_rho, r1, r2):
    """Minimum rotation speed to overcome the threshold pressure

        omega_c = sqrt(2*Pth/(drho*(r2^2 - r1^2))).
    """
    return np.sqrt(2.0 * p_threshold / (delta_rho * (r2 ** 2 - r1 ** 2)))


# ---------------------------------------------- dynamics --------------

def saturation_history(t, sw_init, sw_eq, tau):
    """Average saturation vs time  Sw(t) = Sw_eq + (Sw_init - Sw_eq)*exp(-t/tau)."""
    return sw_eq + (sw_init - sw_eq) * np.exp(-np.asarray(t, float) / tau)


def corey_kr(sw, swc, krw_max=0.3, kro_max=1.0, nw=2.0, no=2.0):
    """Corey water/oil relative permeabilities (primary drainage), returns (krw, kro)."""
    swn = np.clip((np.asarray(sw, float) - swc) / (1.0 - swc), 0.0, 1.0)
    return krw_max * swn ** nw, kro_max * (1.0 - swn) ** no


def pc_correlation(sw, swc, pc_max, pth, b=2.0):
    """Capillary-pressure correlation between Pc(swc)=Pc_max and Pc(1)=Pth."""
    swn = np.clip((np.asarray(sw, float) - swc) / (1.0 - swc), 0.0, 1.0)
    return pth + (pc_max - pth) * (1.0 - swn) ** b


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Centrifuge Capillary Pressure")
    print("=" * 60)

    drho, r1, r2 = 200.0, 0.05, 0.10
    # Hassler-Brunner Pc rises with rotation speed
    omega = 2 * np.pi * 2000 / 60.0                  # 2000 rpm -> rad/s
    pc = hassler_brunner_pc(drho, omega, r1, r2)
    print(f"  Pc at 2000 rpm         = {pc:.0f} Pa")
    assert pc > 0 and hassler_brunner_pc(drho, 2 * omega, r1, r2) > pc

    # Critical speed reproduces the threshold pressure exactly
    pth = 1.0e4
    wc = critical_speed(pth, drho, r1, r2)
    print(f"  critical speed         = {wc * 60 / (2 * np.pi):.0f} rpm")
    assert np.isclose(hassler_brunner_pc(drho, wc, r1, r2), pth)

    # Saturation decays exponentially from initial to equilibrium
    t = np.linspace(0, 5 * 3600, 50)
    sw = saturation_history(t, sw_init=1.0, sw_eq=0.3, tau=3600.0)
    print(f"  Sw(0) / Sw(end)        = {sw[0]:.3f} / {sw[-1]:.3f}")
    assert np.isclose(sw[0], 1.0) and abs(sw[-1] - 0.3) < 0.02 and np.all(np.diff(sw) < 0)

    # Corey: krw rises, kro falls; Pc decreases with saturation
    krw, kro = corey_kr(np.array([0.3, 0.6, 0.9]), swc=0.2)
    assert np.all(np.diff(krw) > 0) and np.all(np.diff(kro) < 0)
    assert pc_correlation(0.3, 0.2, 1e5, 1e4) > pc_correlation(0.9, 0.2, 1e5, 1e4)
    print("  PASS")
    return {"Pc_2000rpm": float(pc), "critical_rpm": float(wc * 60 / (2 * np.pi))}


if __name__ == "__main__":
    test_all()
