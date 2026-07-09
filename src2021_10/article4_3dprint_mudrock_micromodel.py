"""
Article 4: 3D Printing Mudrocks: Experiments in Validating Clay as a Build
           Material for 3D Printing Porous Micromodels
Hasiuk, Harding (2021)
DOI: 10.30632/PJV62N5-2021a4

Paste-extrusion (liquid deposition modeling) 3D printing of wet clay to make
core-plug-sized mudrock analogs.  Porosity and pore-throat-size distributions
of printed cylinders (desiccated and fired) are measured by helium pycnometry
(Boyle's law) and mercury intrusion porosimetry (Washburn) and compared to
hand-cast controls and natural-rock ranges.

Implements:

  - Washburn pore-throat diameter  D = -4*gamma*cos(theta) / P    (Eq. 1)
  - Boyle's-law grain-volume helper  P1*V1 = P2*V2
  - Porosity  phi = (V_bulk - V_grain) / V_bulk
  - Cylinder bulk volume
  - Dimensional / mass loss percent on firing

Note: the journal's Eq. 1 glyph was image-rendered; the standard Washburn
form here matches the paper's two anchor points (a few psi -> tens of microns;
33,000 psi -> single-digit nm).  Mercury defaults: gamma = 0.480 N/m,
theta = 140 deg.  Pressures in Pa, lengths in metres unless noted.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

HG_SURFACE_TENSION = 0.480      # N/m (480 erg/cm^2)
HG_CONTACT_ANGLE = 140.0        # degrees
PSI_TO_PA = 6894.76


# ---------------------------------------------- Eq. 1: Washburn ---------

def washburn_diameter(pressure_pa, gamma=HG_SURFACE_TENSION,
                      theta_deg=HG_CONTACT_ANGLE):
    """Pore-throat diameter  D = -4*gamma*cos(theta)/P  (Eq. 1).  Metres.

    theta > 90 deg makes cos negative, so D is positive.
    """
    theta = np.radians(theta_deg)
    return -4.0 * gamma * np.cos(theta) / np.asarray(pressure_pa, float)


# ---------------------------------------------- Boyle's law -------------

def boyle_grain_volume(v_cell, v_expansion, p1, p2):
    """Grain volume from helium expansion  V_grain = V_cell - V_exp/(P1/P2 - 1)."""
    return petrolib.porosity_lithology.boyle_grain_volume(v_cell, v_expansion, p1, p2)


# ---------------------------------------------- porosity ----------------

def porosity(v_bulk, v_grain):
    """Porosity  phi = (V_bulk - V_grain) / V_bulk."""
    return petrolib.porosity_lithology.porosity_from_volumes(v_bulk, v_grain)


def cylinder_volume(diameter, height):
    """Cylinder bulk volume  V = pi*(d/2)^2*h."""
    return np.pi * (diameter / 2.0) ** 2 * height


# ---------------------------------------------- loss percent ------------

def loss_percent(before, after):
    """Dimensional or mass loss  100*(before-after)/before."""
    return 100.0 * (before - after) / before


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: 3D-Printed Mudrock Micromodels")
    print("=" * 60)

    # Washburn anchors: low pressure -> tens of microns; high -> a few nm
    d_lo = washburn_diameter(3.0 * PSI_TO_PA)        # ~ a few psi
    d_hi = washburn_diameter(33000.0 * PSI_TO_PA)    # max pressure
    print(f"  D @ 3 psi              = {d_lo*1e6:.1f} um")
    print(f"  D @ 33,000 psi         = {d_hi*1e9:.1f} nm")
    assert 40e-6 < d_lo < 90e-6          # tens of microns
    assert 4e-9 < d_hi < 10e-9           # single-digit nm
    # inverse pressure dependence
    assert d_lo > d_hi
    assert abs(washburn_diameter(2 * 3.0 * PSI_TO_PA) - d_lo / 2.0) < 1e-12

    # Cylinder bulk volume (25 mm dia x 25 mm tall -> 2.5 cm)
    v_bulk = cylinder_volume(2.5, 2.5)
    print(f"  cylinder bulk volume   = {v_bulk:.2f} cm^3")
    assert abs(v_bulk - 12.27) < 0.05

    # Porosity from bulk & grain volumes (desiccated clay ~ 36%)
    phi = porosity(v_bulk, v_grain=v_bulk * 0.64)
    print(f"  porosity               = {phi:.3f}")
    assert abs(phi - 0.36) < 1e-9

    # Firing reduces porosity and shrinks dimensions
    fired_phi = porosity(v_bulk, v_grain=v_bulk * 0.93)
    assert fired_phi < phi
    shrink = loss_percent(25.0, 22.0)
    print(f"  height loss on firing  = {shrink:.0f}%")
    assert 0 < shrink < 30
    print("  PASS")
    return {"D_3psi_um": d_lo * 1e6, "D_max_nm": d_hi * 1e9,
            "porosity": phi}


if __name__ == "__main__":
    test_all()
