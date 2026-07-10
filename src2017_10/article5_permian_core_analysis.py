"""
Article 5: Lessons Learned in Permian Core Analysis: Comparison Between Retort,
           GRI, and Routine Methodologies
Blount, Croft, Driskill, Tepper (2017)
Reference: Petrophysics Vol. 58, No. 5 (October 2017), pp. 517-527
DOI: none assigned (this issue predates SPWLA DOI assignment)

A round-robin comparison of retort, GRI (crushed-rock), and routine core-analysis
(RCA) methods on twin Permian (Bone Spring) samples, motivated by porosity and
hydrocarbon-pore-volume discrepancies between methods/vendors.  This *methodology
proxy* implements the standard core-analysis mass-balance relations the three
methods rely on: porosity from grain/bulk volume, retort and Dean-Stark
saturations, hydrocarbon pore volume, and the relative discrepancy used to
compare methods.

Implements:

  - Porosity from grain and bulk volume  phi = (BV - GV)/BV
  - Retort saturations  So = Vo/Vp,  Sw = Vw/Vp,  Sg = 1 - So - Sw
  - Dean-Stark water saturation  Sw = Vw/Vp
  - Hydrocarbon pore volume  HCPV = phi*(1 - Sw)
  - Relative discrepancy between two method results

Note: this article's body was beyond this issue's machine extraction (only the
abstract/intro were captured), so - consistent with the methodology proxies
elsewhere in this repository - the relations below are the standard
retort/GRI/Dean-Stark core-analysis formulas the paper compares, not formulas
transcribed from it.  Volumes consistent (cm^3), fractions dimensionless.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- porosity / saturation --------------

def porosity(grain_volume, bulk_volume):
    """Porosity  phi = (BV - GV)/BV  (pore volume over bulk volume)."""
    return petrolib.porosity_lithology.porosity_from_volumes(bulk_volume, grain_volume)


def retort_saturations(oil_volume, water_volume, pore_volume):
    """Retort fluid saturations  So, Sw, Sg (gas by difference), returned as a tuple."""
    so = oil_volume / pore_volume
    sw = water_volume / pore_volume
    sg = 1.0 - so - sw
    return so, sw, max(sg, 0.0)


def dean_stark_sw(water_volume, pore_volume):
    """Dean-Stark water saturation  Sw = Vw/Vp (extracted-water volume)."""
    return water_volume / pore_volume


def hydrocarbon_pore_volume(phi, sw):
    """Hydrocarbon pore volume fraction  HCPV = phi*(1 - Sw)."""
    return petrolib.porosity_lithology.hydrocarbon_pore_volume(phi, sw)


def relative_discrepancy(value_a, value_b):
    """Relative discrepancy between two method results  |a - b|/mean(a, b)."""
    return float(petrolib.data_qc_io.clean.relative_discrepancy(value_a, value_b))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Permian Core Analysis (proxy)")
    print("=" * 60)

    # Porosity from grain/bulk volume
    phi = porosity(grain_volume=8.0, bulk_volume=10.0)
    print(f"  porosity               = {phi:.3f}")
    assert np.isclose(phi, 0.2)

    # Retort saturations sum to 1 (gas by difference)
    pore_vol = 2.0
    so, sw, sg = retort_saturations(oil_volume=0.6, water_volume=1.0, pore_volume=pore_vol)
    print(f"  So / Sw / Sg           = {so:.2f} / {sw:.2f} / {sg:.2f}")
    assert np.isclose(so + sw + sg, 1.0)

    # Dean-Stark Sw matches the retort water saturation here
    assert np.isclose(dean_stark_sw(1.0, pore_vol), sw)

    # Hydrocarbon pore volume
    hcpv = hydrocarbon_pore_volume(phi, sw)
    print(f"  HCPV                   = {hcpv:.4f}")
    assert np.isclose(hcpv, 0.2 * (1 - 0.5))

    # Method discrepancy (the ~2-3 p.u. / ~25% HCPV gaps the paper reports)
    disc = relative_discrepancy(0.20, 0.17)
    print(f"  porosity discrepancy   = {disc * 100:.1f} %")
    assert 0.0 < disc < 0.3
    print("  PASS")
    return {"phi": float(phi), "HCPV": float(hcpv), "discrepancy": float(disc)}


if __name__ == "__main__":
    test_all()
