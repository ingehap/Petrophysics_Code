"""
Article 8: Spontaneous Gas-Water Imbibition in Mixed-Wet Pores
Wang, He, Xiao, Wang, Ma (2020)
DOI: 10.30632/PJV61N2-2020a8

Spontaneous water imbibition in mixed-wet pores is governed by the capillary
driving force, which is positive (imbibing) in water-wet pores and negative
(resisting) in oil-wet pores.  The Lucas-Washburn equation describes the
square-root-of-time imbibition front, and the net mixed-wet imbibition follows
from the water-wet / oil-wet pore fractions.

Implements:

  - Lucas-Washburn imbibition length  L = sqrt(sigma*r*cos(theta)*t/(2*mu))
  - Young-Laplace capillary driving pressure  Pc = 2*sigma*cos(theta)/r
  - Net mixed-wet capillary force from water-wet / oil-wet pore fractions
  - Imbibition rate dL/dt

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard Lucas-Washburn / Young-Laplace relations the
paper's title describes.  SI units; angles in degrees.
"""

import numpy as np


# ---------------------------------------------- Lucas-Washburn ----------

def washburn_length(sigma, r, theta_deg, mu, t):
    """Lucas-Washburn imbibition length  L = sqrt(sigma*r*cos(theta)*t/(2*mu)).

    Only defined (real imbibition) for water-wet pores (cos(theta) > 0);
    returns 0 where the pore is oil-wet (no spontaneous water uptake).
    """
    c = np.cos(np.radians(theta_deg))
    drive = sigma * r * c * np.asarray(t, float) / (2.0 * mu)
    return np.sqrt(np.clip(drive, 0.0, None))


def imbibition_rate(sigma, r, theta_deg, mu, t):
    """Imbibition rate  dL/dt = 0.5*sqrt(sigma*r*cos(theta)/(2*mu*t))."""
    c = np.cos(np.radians(theta_deg))
    t = np.asarray(t, float)
    return 0.5 * np.sqrt(np.clip(sigma * r * c / (2.0 * mu * t), 0.0, None))


# ---------------------------------------------- Young-Laplace -----------

def capillary_pressure(sigma, r, theta_deg):
    """Young-Laplace capillary pressure  Pc = 2*sigma*cos(theta)/r.

    Positive for water-wet (theta < 90), negative for oil-wet (theta > 90).
    """
    return 2.0 * sigma * np.cos(np.radians(theta_deg)) / r


# ---------------------------------------------- mixed-wet ---------------

def net_capillary_force(sigma, r, f_waterwet, theta_ww=40.0, theta_ow=120.0):
    """Net capillary driving pressure for a mixed-wet pore population.

    Volume-weighted sum of the (positive) water-wet and (negative) oil-wet
    Young-Laplace pressures; f_waterwet is the water-wet pore fraction.
    """
    pc_ww = capillary_pressure(sigma, r, theta_ww)
    pc_ow = capillary_pressure(sigma, r, theta_ow)
    return f_waterwet * pc_ww + (1.0 - f_waterwet) * pc_ow


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Spontaneous Gas-Water Imbibition (Mixed-Wet)")
    print("=" * 60)

    sigma, r, mu = 0.05, 1e-6, 1e-3        # N/m, 1 um pore, water viscosity

    # Lucas-Washburn: imbibition length grows as sqrt(t) in a water-wet pore
    L1 = washburn_length(sigma, r, 40.0, mu, 1.0)
    L4 = washburn_length(sigma, r, 40.0, mu, 4.0)
    print(f"  L(t=1) / L(t=4)        = {L1*1e3:.3f} / {L4*1e3:.3f} mm")
    assert abs(L4 / L1 - 2.0) < 1e-9       # doubles when time quadruples
    # an oil-wet pore does not spontaneously imbibe water
    assert washburn_length(sigma, r, 120.0, mu, 1.0) == 0.0

    # Imbibition rate decreases with time (slows as the front advances)
    assert imbibition_rate(sigma, r, 40.0, mu, 4.0) < imbibition_rate(sigma, r, 40.0, mu, 1.0)

    # Young-Laplace: water-wet drives imbibition (+), oil-wet resists (-)
    assert capillary_pressure(sigma, r, 40.0) > 0
    assert capillary_pressure(sigma, r, 120.0) < 0

    # Mixed-wet net force turns positive once enough pores are water-wet
    nf_low = net_capillary_force(sigma, r, f_waterwet=0.2)
    nf_high = net_capillary_force(sigma, r, f_waterwet=0.8)
    print(f"  net Pc  20%/80% water-wet = {nf_low:.0f} / {nf_high:.0f} Pa")
    assert nf_high > nf_low and nf_high > 0 and nf_low < nf_high
    print("  PASS")
    return {"L_t4_mm": float(L4 * 1e3), "net_pc_80": float(nf_high)}


if __name__ == "__main__":
    test_all()
