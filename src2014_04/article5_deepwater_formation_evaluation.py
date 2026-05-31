"""
Article 5: Formation-Evaluation Challenges and Opportunities in Deepwater
Roland Chemali, Wade Samec, Ron Balliet, Paul Cooper, David Torres, Chris Jones
           (2014)
Reference: Petrophysics Vol. 55, No. 2 (April 2014), pp. 124-135
DOI: none assigned (this issue predates SPWLA DOI assignment)

Special Issue on Deepwater.  An applied-technology review of deepwater formation
evaluation: the narrow pore-pressure / fracture-pressure (ECD) drilling window,
resistivity anisotropy (Rh, Rv) for laminated-sand saturation, and NMR fluid
typing.  This module implements the standard relations the review references.

Implements:

  - Equivalent-circulating-density (ECD) drilling window check
  - Resistivity anisotropy ratio  lambda = sqrt(Rv/Rh)
  - Thomas-Stieber laminated-sand horizontal/vertical resistivity
  - Laminated-sand water saturation from Rh
  - NMR hydrogen index and a T1 fluid-typing contrast

Note: this is an applied review; the models it cites (Eaton, Thomas-Stieber,
Mollison, Coates) are referenced but not written, so the standard forms are
reconstructed here.  A field case reports phi = 30 p.u., k = 1,000 md, hydrogen
index 0.33 (gas).  Resistivities in Ohm*m, pressures in psi.
"""

import numpy as np


# ---------------------------------------------- ECD window --------------

def within_ecd_window(ecd_pressure, pore_pressure, fracture_pressure):
    """Check that the equivalent-circulating-density pressure stays inside the
    deepwater drilling window

        pore_pressure < ECD < fracture_pressure,

    i.e. above pore pressure (no influx) and below fracture pressure (no losses).
    """
    return bool(pore_pressure < ecd_pressure < fracture_pressure)


def ecd_margin(pore_pressure, fracture_pressure):
    """Width of the drilling window  frac - pore  (psi), the deepwater margin."""
    return fracture_pressure - pore_pressure


# ---------------------------------------------- anisotropy --------------

def anisotropy_ratio(r_v, r_h):
    """Resistivity anisotropy coefficient  lambda = sqrt(Rv/Rh)."""
    return np.sqrt(r_v / r_h)


def laminated_resistivity(r_sand, r_shale, sand_fraction):
    """Thomas-Stieber laminated-sand resistivities

        1/Rh = Vsd/Rsand + (1-Vsd)/Rshale     (parallel / horizontal),
        Rv  = Vsd*Rsand + (1-Vsd)*Rshale      (series / vertical),

    with the laminar sand fraction Vsd.  Returns (Rh, Rv).
    """
    vsd = sand_fraction
    rh = 1.0 / (vsd / r_sand + (1.0 - vsd) / r_shale)
    rv = vsd * r_sand + (1.0 - vsd) * r_shale
    return rh, rv


def laminated_water_saturation(r_sand, rw, phi_sand, a=1.0, m=2.0, n=2.0):
    """Archie water saturation of the laminar sand from its (anisotropy-resolved)
    sand resistivity

        Sw = (a*Rw/(phi_sand^m*Rsand))^(1/n).
    """
    return (a * rw / (phi_sand ** m * r_sand)) ** (1.0 / n)


# ---------------------------------------------- NMR --------------

def hydrogen_index(phi_apparent, phi_true):
    """Hydrogen index from the apparent (NMR) and true porosity

        HI = phi_apparent/phi_true,

    HI < 1 for gas (low hydrogen density), diagnostic in fluid typing.
    """
    return phi_apparent / phi_true


def t1_fluid_contrast(t1_hydrocarbon, t1_water):
    """T1 fluid-typing contrast  T1_hc/T1_water (> 1 for light hydrocarbons)."""
    return t1_hydrocarbon / t1_water


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Deepwater Formation-Evaluation")
    print("=" * 60)

    # ECD window: in-window vs out-of-window
    assert within_ecd_window(9500, 9000, 10000)
    assert not within_ecd_window(10500, 9000, 10000)
    print(f"  ECD margin = {ecd_margin(9000, 10000):.0f} psi")
    assert ecd_margin(9000, 10000) == 1000

    # Anisotropy: laminated sand/shale gives Rv > Rh, lambda > 1
    rh, rv = laminated_resistivity(r_sand=20.0, r_shale=1.0, sand_fraction=0.5)
    lam = anisotropy_ratio(rv, rh)
    print(f"  Rh={rh:.3f}  Rv={rv:.3f}  lambda={lam:.3f}")
    assert rv > rh and lam > 1.0

    # Resolving the true sand resistivity gives a lower (more correct) Sw than
    # using the bulk horizontal resistivity
    sw_sand = laminated_water_saturation(r_sand=20.0, rw=0.05, phi_sand=0.30)
    sw_bulk = laminated_water_saturation(r_sand=rh, rw=0.05, phi_sand=0.30)
    print(f"  Sw(sand)={sw_sand:.3f}  Sw(bulk Rh)={sw_bulk:.3f}")
    assert sw_sand < sw_bulk

    # NMR: hydrogen index 0.33 flags gas; light HC has longer T1 than water
    hi = hydrogen_index(0.10, 0.30)
    print(f"  hydrogen index = {hi:.2f}")
    assert np.isclose(hi, 0.333, atol=0.01) and hi < 1.0
    assert t1_fluid_contrast(4.0, 0.5) > 1.0
    print("  PASS")
    return {"ECD_margin": 1000.0, "lambda": float(lam), "HI": float(hi)}


if __name__ == "__main__":
    test_all()
