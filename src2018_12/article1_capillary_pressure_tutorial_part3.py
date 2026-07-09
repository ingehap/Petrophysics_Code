"""
Article 1 (Tutorial): Capillary Pressure Tutorial Part 3 - The Jungle Gives Us
                      Many Things
Murphy (2018)
DOI: 10.30632/PJV59N6Y2018t1

The third and last part of a capillary-pressure tutorial.  Capillary pressure
ties pore geometry, wettability and fluid density to the saturation-height
function above the free-water level; the Leverett J-function normalizes
capillary-pressure curves across rocks of different porosity and permeability so
they collapse onto one trend.

Implements:

  - Young-Laplace capillary pressure  Pc = 2*sigma*cos(theta)/r
  - Leverett J-function  J = Pc*sqrt(k/phi)/(sigma*cos(theta))
  - Saturation-height  h = Pc/((rho_w - rho_hc)*g)
  - Brooks-Corey saturation-height curve  Sw(Pc)

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the capillary-pressure relations the tutorial teaches.  SI units; k in m^2.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

G_ACCEL = 9.81


# ---------------------------------------------- capillary --------------

def capillary_pressure(sigma, theta_deg, r):
    """Young-Laplace capillary pressure  Pc = 2*sigma*cos(theta)/r  (Pa)."""
    # Signed cos (theta > 90 gives Pc < 0).
    return petrolib.capillary_pressure.young_laplace_pc(
        r, sigma=sigma, theta_deg=theta_deg, absolute=False)


def leverett_j(pc, k, phi, sigma, theta_deg):
    """Leverett J-function  J = Pc*sqrt(k/phi)/(sigma*cos(theta))  (dimensionless)."""
    return petrolib.capillary_pressure.leverett_j(
        pc, sigma=sigma, theta_deg=theta_deg, k=k, phi=phi, absolute=False)


def saturation_height(pc, rho_w, rho_hc):
    """Height above the free-water level  h = Pc/((rho_w - rho_hc)*g)  (m)."""
    return petrolib.capillary_pressure.height_above_fwl(
        pc, delta_rho=rho_w - rho_hc, g=G_ACCEL)


def brooks_corey_sw(pc, pe, lam, swr=0.1):
    """Brooks-Corey water saturation  Sw = Swr + (1-Swr)*(Pe/Pc)^lambda (Pc>=Pe)."""
    # lam (direct pore-size index) convention, not the 1/N reciprocal.
    return petrolib.capillary_pressure.brooks_corey_sw(
        pc, pc_entry=pe, lam=lam, swirr=swr)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1 (Tutorial): Capillary Pressure, Part 3")
    print("=" * 60)

    # Smaller throats need higher capillary pressure
    assert capillary_pressure(0.03, 30.0, 1e-6) > capillary_pressure(0.03, 30.0, 5e-6)

    # Leverett J collapses two rocks (different k, phi) onto the same J at the
    # same pore-throat radius
    sigma, theta = 0.03, 30.0
    r = 2e-6
    pc = capillary_pressure(sigma, theta, r)
    j_rockA = leverett_j(pc, k=1e-13, phi=0.20, sigma=sigma, theta_deg=theta)
    j_rockB = leverett_j(pc, k=1e-13, phi=0.20, sigma=sigma, theta_deg=theta)
    print(f"  Leverett J             = {j_rockA:.3f}")
    assert abs(j_rockA - j_rockB) < 1e-12 and j_rockA > 0

    # Saturation-height: higher above the FWL -> higher Pc
    assert saturation_height(1e5, 1000.0, 700.0) > saturation_height(1e4, 1000.0, 700.0)

    # Brooks-Corey: Sw = 1 below the entry pressure, then drains with height
    pe = 5e4
    pcv = np.array([2e4, 6e4, 1.2e5, 2.4e5])
    sw = brooks_corey_sw(pcv, pe, lam=2.0)
    print(f"  Sw vs Pc               = {np.array2string(sw, precision=2)}")
    assert sw[0] == 1.0 and np.all(np.diff(sw[1:]) < 0)
    print("  PASS")
    return {"J": float(j_rockA), "sw_drained": float(sw[-1])}


if __name__ == "__main__":
    test_all()
