"""
Article 6: A New Waterflood Initialization Protocol With Wettability Alteration
           for Pore-Scale Multiphase Flow Experiments
Lin, Bijeljic, Krevor, Blunt, Rucker, Berg, Coorn, van der Linde, Georgiadis,
Wilson (2019)
DOI: 10.30632/PJV60N2-2019a4

A pore-scale waterflood experiment must first be initialized to a representative
initial water saturation and wettability state: the sample is drained to the
connate water saturation along a primary-drainage capillary-pressure curve, then
aged so crude oil alters the contact angle from water-wet toward mixed/oil-wet.
This module implements the capillary-pressure / contact-angle relations the
initialization protocol controls.

Implements:

  - Young-Laplace capillary pressure  Pc = 2*sigma*cos(theta)/r
  - Primary-drainage initial water saturation from a threshold Pc
  - Aging: contact-angle alteration and its effect on Pc
  - Amott-style wettability shift from the contact angle

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the capillary-pressure / wettability-alteration relations the protocol uses.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- capillary pressure ------

def capillary_pressure(sigma, theta_deg, r):
    """Young-Laplace capillary pressure  Pc = 2*sigma*cos(theta)/r  (Pa)."""
    # Signed cos (theta > 90 gives Pc < 0).
    return petrolib.capillary_pressure.young_laplace_pc(
        r, sigma=sigma, theta_deg=theta_deg, absolute=False)


def drainage_swi(pore_radii, volumes, pc_applied, sigma, theta_deg):
    """Initial water saturation after primary drainage at an applied Pc.

    Pores with entry pressure below the applied Pc are invaded by oil (drained);
    the remaining (smallest) water-filled pores set Swi.
    """
    r = np.asarray(pore_radii, float); v = np.asarray(volumes, float)
    pc_entry = capillary_pressure(sigma, theta_deg, r)
    water_filled = pc_entry > pc_applied              # too tight for oil to enter
    return float(v[water_filled].sum() / v.sum())


# ---------------------------------------------- aging -------------------

def age_contact_angle(theta_initial, aging_strength):
    """Aging alters the contact angle from water-wet toward oil-wet.

        theta_aged = theta_initial + aging_strength*(180 - theta_initial)
    aging_strength in [0, 1].
    """
    return theta_initial + aging_strength * (180.0 - theta_initial)


def amott_wettability(theta_deg):
    """Amott-style wettability index from contact angle  WI = cos(theta)."""
    return np.cos(np.radians(theta_deg))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Waterflood Initialization With Wettability Alteration")
    print("=" * 60)

    # Capillary pressure: smaller throats need higher Pc; oil-wet (theta>90) < 0
    assert capillary_pressure(0.03, 30.0, 1e-6) > capillary_pressure(0.03, 30.0, 5e-6)
    assert capillary_pressure(0.03, 120.0, 1e-6) < 0

    # Primary drainage: higher applied Pc drains more oil in -> lower Swi
    radii = np.array([0.2, 0.5, 1.0, 2.0, 5.0]) * 1e-6
    vols = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    swi_lo = drainage_swi(radii, vols, 5e4, 0.03, 30.0)
    swi_hi = drainage_swi(radii, vols, 2e5, 0.03, 30.0)
    print(f"  Swi at Pc 5e4 / 2e5 Pa = {swi_lo:.2f} / {swi_hi:.2f}")
    assert swi_hi < swi_lo

    # Aging shifts the contact angle toward oil-wet and lowers Amott WI
    th0 = 30.0
    th_aged = age_contact_angle(th0, 0.7)
    print(f"  contact angle 0/aged   = {th0} / {th_aged:.0f} deg")
    assert th_aged > th0
    assert amott_wettability(th_aged) < amott_wettability(th0)
    # strong aging can drive the surface oil-wet (WI < 0)
    assert amott_wettability(age_contact_angle(30.0, 1.0)) < 0
    print("  PASS")
    return {"swi_lo": swi_lo, "swi_hi": swi_hi, "theta_aged": float(th_aged)}


if __name__ == "__main__":
    test_all()
