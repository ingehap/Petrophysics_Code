"""
Article 1 (Tutorial): Capillary Pressure Tutorial Part 1 - It's a Jungle in Here
Thomas (2018)
DOI: 10.30632/PJV59V4-2018t1

Part 1 of the capillary-pressure tutorial builds capillary pressure from first
principles: the surface tension of an interface, the Young-Laplace pressure jump
across a curved meniscus, the force balance that drives capillary rise in a
tube, and the resulting definition of capillary pressure as the pressure
difference across the curved interface.  A bundle of tubes of different radii
turns this into a Pc-vs-saturation curve that encodes the pore-throat-size
distribution used for rock typing.

Implements:

  - Young-Laplace pressure jump across a meniscus  dP = 2*sigma/r
  - Capillary rise height  h = 2*sigma*cos(theta)/(rho*g*r)
  - Capillary pressure from the rise  Pc = (rho_w - rho_a)*g*h
  - Pc from pore-throat radius (bundle of tubes)  Pc = 2*sigma*cos(theta)/r
  - Wettability classification from the contact angle

Note: this issue's PDF has a text layer but its typeset display-equation glyphs
were dropped in extraction, so these are faithful standard-form reconstructions
of the capillary-pressure relations the tutorial derives (Eqs. 2-13).  SI units.
"""

import numpy as np

G_ACCEL = 9.81


# ---------------------------------------------- interface --------------

def young_laplace(sigma, r):
    """Pressure jump across a spherical meniscus  dP = 2*sigma/r  (Eq. 2).

    The concave (non-wetting) side carries the higher pressure; a tighter
    radius r gives a larger jump.
    """
    return 2.0 * sigma / np.asarray(r, float)


def capillary_rise_height(sigma, theta_deg, rho, r, rho_above=0.0):
    """Capillary rise height in a tube  h = 2*sigma*cos(theta)/((rho-rho_above)*g*r).

    Solves the rise force balance (Eq. 8): the vertical surface-tension force
    sigma*2*pi*r*cos(theta) supports the column weight (rho-rho_above)*g*pi*r^2*h.
    A wetting fluid (theta < 90) rises; a non-wetting fluid (theta > 90, e.g.
    mercury) is depressed (h < 0).
    """
    drho = rho - rho_above
    return 2.0 * sigma * np.cos(np.radians(theta_deg)) / (drho * G_ACCEL * np.asarray(r, float))


# ---------------------------------------------- capillary pressure --------------

def capillary_pressure_from_rise(rho_w, rho_a, h):
    """Capillary pressure from the rise  Pc = (rho_w - rho_a)*g*h  (Eqs. 12-13).

    The difference of the two hydrostatic columns (light fluid above, dense
    wetting fluid below) across the meniscus is the capillary pressure.
    """
    return (rho_w - rho_a) * G_ACCEL * np.asarray(h, float)


def capillary_pressure_radius(sigma, theta_deg, r):
    """Pc for a single pore throat  Pc = 2*sigma*cos(theta)/r.

    Combining Young-Laplace with the contact angle.  The capillary-pressure
    curve of a rock is the envelope of these single-tube values over the pore-
    throat-size distribution; |cos(theta)| handles a non-wetting system.
    """
    return 2.0 * sigma * abs(np.cos(np.radians(theta_deg))) / np.asarray(r, float)


def wettability(theta_deg):
    """Classify the interface from the contact angle (degrees)."""
    if theta_deg < 90.0:
        return "wetting"
    if theta_deg > 90.0:
        return "non-wetting"
    return "neutral"


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1 (Tutorial): Capillary Pressure, Part 1")
    print("=" * 60)

    # Young-Laplace: a tighter radius raises the pressure jump
    assert young_laplace(0.072, 1e-4) > young_laplace(0.072, 1e-3)

    # Water wets glass (theta ~ 0): rises, and h ~ 1/r
    h_fine = capillary_rise_height(0.072, 0.0, 1000.0, 1e-4)
    h_coarse = capillary_rise_height(0.072, 0.0, 1000.0, 1e-3)
    print(f"  water rise r=0.1mm/1mm = {h_fine:.3f} / {h_coarse:.3f} m")
    assert h_fine > h_coarse > 0
    assert np.isclose(h_fine / h_coarse, 10.0, rtol=1e-6)   # inverse in r

    # Mercury (theta ~ 140) is depressed, not raised
    h_hg = capillary_rise_height(0.485, 140.0, 13550.0, 1e-4)
    print(f"  mercury rise           = {h_hg * 1e3:.3f} mm  (depression)")
    assert h_hg < 0

    # Pc from the rise matches Pc from the pore-throat radius (water-wet)
    pc_rise = capillary_pressure_from_rise(1000.0, 0.0, h_fine)
    pc_r = capillary_pressure_radius(0.072, 0.0, 1e-4)
    print(f"  Pc from rise / radius  = {pc_rise:.1f} / {pc_r:.1f} Pa")
    assert np.isclose(pc_rise, pc_r, rtol=1e-6)

    assert wettability(0.0) == "wetting" and wettability(140.0) == "non-wetting"
    print("  PASS")
    return {"h_fine": float(h_fine), "Pc": float(pc_r)}


if __name__ == "__main__":
    test_all()
