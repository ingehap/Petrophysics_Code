"""
Article 2: A Blind Study of Four Digital Rock Physics Vendor Laboratories on
           Porosity, Absolute Permeability, and Primary Drainage Capillary
           Pressure Data on Tight Outcrops
Chhatre, Sahoo, Leonardi, Vidal, Rainey, Braun, Patel (2018)
DOI: 10.30632/petro_059_1_a1

A blind benchmark of four digital-rock-physics vendors against in-house physical
porosity, absolute permeability, and primary-drainage capillary pressure on six
tight outcrops.  The implementable core is the petrophysical machinery used to
compare the curves: converting capillary pressure to a pore-throat radius with
the Young-Laplace relation, normalizing water saturation, and fitting a
power-law saturation-height (drainage Pc) curve.

Implements:

  - Young-Laplace pore-throat radius  r = 2*sigma*cos(theta)/Pc
  - Normalized water saturation  Swn = (Sw - Swir)/(1 - Swir)
  - Power-law drainage capillary pressure  Pc = Pcth + A*Swn^(-B)
  - Interfacial-tension rescaling between fluid systems

Note: this issue's PDF has a text layer but its typeset display-equation glyphs
were dropped in extraction, so the numbered relations (Eqs. 1-2) are faithful
standard-form reconstructions from the surrounding prose and nomenclature.  SI:
Pc in Pa, sigma in N/m, r in m.
"""

import numpy as np


# ---------------------------------------------- capillary pressure --------------

def pore_throat_radius(pc, sigma=0.072, theta_deg=0.0):
    """Pore-throat radius from capillary pressure  r = 2*sigma*cos(theta)/Pc.

    Young-Laplace for a cylindrical throat; primary drainage uses theta ~ 0.
    """
    return 2.0 * sigma * np.cos(np.radians(theta_deg)) / np.asarray(pc, float)


def normalized_sw(sw, swir):
    """Normalized water saturation  Swn = (Sw - Swir)/(1 - Swir)."""
    return (np.asarray(sw, float) - swir) / (1.0 - swir)


def drainage_pc(swn, pc_threshold, a, b):
    """Power-law primary-drainage capillary pressure  Pc = Pcth + A*Swn^(-B).

    Pc rises sharply as the normalized saturation approaches its irreducible
    value; Pcth is the threshold (entry) capillary pressure.
    """
    return pc_threshold + a * np.asarray(swn, float) ** (-b)


def rescale_ift(pc, sigma_from, sigma_to, theta_from=0.0, theta_to=0.0):
    """Rescale Pc between fluid systems by the |sigma*cos(theta)| ratio."""
    return (np.asarray(pc, float) * abs(sigma_to * np.cos(np.radians(theta_to)))
            / abs(sigma_from * np.cos(np.radians(theta_from))))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: DRP Blind Study (capillary pressure)")
    print("=" * 60)

    # Higher capillary pressure probes smaller pore throats
    r_hi = pore_throat_radius(1e6)
    r_lo = pore_throat_radius(1e5)
    print(f"  r at Pc 1e6 / 1e5      = {r_hi * 1e9:.1f} / {r_lo * 1e9:.1f} nm")
    assert r_lo > r_hi > 0

    # Drainage Pc increases as saturation drops toward irreducible
    swn = normalized_sw(np.array([0.8, 0.5, 0.25]), swir=0.15)
    pc = drainage_pc(swn, pc_threshold=2e5, a=1e5, b=0.5)
    print(f"  drainage Pc            = {np.array2string(pc, precision=0)}")
    assert pc[0] < pc[1] < pc[2] and np.all(pc > 2e5)

    # Gas-brine (72 mN/m) to oil-water (28 mN/m) lowers Pc
    pc_ow = rescale_ift(1e6, sigma_from=0.072, sigma_to=0.028)
    assert pc_ow < 1e6
    print("  PASS")
    return {"r_hi_nm": float(r_hi * 1e9), "pc_irr": float(pc[-1])}


if __name__ == "__main__":
    test_all()
