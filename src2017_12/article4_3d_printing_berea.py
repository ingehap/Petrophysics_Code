"""
Article 4: 3D Printing Berea Sandstone: Testing a New Tool for Petrophysical
           Analysis of Reservoirs
Ishutov, Hasiuk (2017)
Reference: Petrophysics Vol. 58, No. 6 (December 2017), pp. 592-602
DOI: none assigned (this issue predates SPWLA DOI assignment)

A digital pore-network model extracted from CT images of Berea sandstone is
magnified so its smallest throats exceed the 3D printer's resolution, then
printed and characterized to test 3D printing as a petrophysical tool.  This
module implements the quantitative pieces: porosity from segmented voxels, the
model magnification, the printer's gap-test resolution calibration (a 160-micron
design printed as ~132 microns), the printability check, and the porosity /
pore-throat offsets of the proxy versus the natural sample.

Implements:

  - Porosity from segmented voxels  phi = pore_voxels/total_voxels
  - Model magnification of a feature size
  - Gap-test printed-size calibration (linear design->printed fit)
  - Printability check against the minimum printable design gap
  - Proxy-vs-natural porosity / throat offsets

Note: this article is an experimental/workflow study (no governing equations),
so the relations below are the calibration and geometry computations it reports;
the gap-test datum (160-micron design -> ~132-micron printed; gaps < 160 microns
do not print) is reproduced.  Sizes in micrometres unless noted.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

MIN_PRINTABLE_DESIGN_UM = 160.0     # smallest design gap that prints


# ---------------------------------------------- geometry --------------

def porosity_from_voxels(pore_voxels, total_voxels):
    """Porosity from a segmented CT volume  phi = pore_voxels/total_voxels."""
    return np.asarray(pore_voxels, float) / total_voxels


def magnify(feature_size, factor):
    """Magnify a model feature size by the print magnification factor."""
    return np.asarray(feature_size, float) * factor


def calibrate_gap_test(design_gaps, printed_gaps):
    """Fit the printer's design->printed size calibration (slope, intercept)."""
    lf = petrolib.inversion_numerics.fitting.fit_line(design_gaps, printed_gaps)
    return lf.slope, lf.intercept


def printed_size(design_gap, slope, intercept):
    """Predicted printed feature size from the gap-test calibration."""
    return slope * np.asarray(design_gap, float) + intercept


def is_printable(design_gap, min_design=MIN_PRINTABLE_DESIGN_UM):
    """True if a design gap is at or above the minimum printable size (~160 um)."""
    return np.asarray(design_gap, float) >= min_design


def offset(proxy_value, natural_value):
    """Signed proxy-minus-natural offset (porosity p.u. or throat micrometres)."""
    return proxy_value - natural_value


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: 3D Printing Berea Sandstone")
    print("=" * 60)

    # Porosity from voxel counts
    phi = porosity_from_voxels(213, 1000)
    assert np.isclose(phi, 0.213)

    # A sub-resolution 18-micron throat must be magnified ~10x to print
    assert not is_printable(18.0)
    assert is_printable(magnify(18.0, 10.0))

    # Gap-test calibration: a 160-micron design prints near 132 microns
    design = np.array([160.0, 320.0, 640.0, 1280.0])
    printed = np.array([132.0, 290.0, 600.0, 1240.0])
    slope, intercept = calibrate_gap_test(design, printed)
    p160 = printed_size(160.0, slope, intercept)
    print(f"  printed size @160um    = {p160:.0f} um")
    assert abs(p160 - 132.0) < 15.0

    # The proxy reads lower porosity and smaller throats than the natural sample
    assert offset(0.193, 0.213) < 0          # ~ -2 p.u. porosity
    assert offset(132.0, 188.0) < 0          # smaller pore throat
    print("  PASS")
    return {"phi": float(phi), "printed_160": float(p160)}


if __name__ == "__main__":
    test_all()
