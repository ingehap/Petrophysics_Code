"""
Article 10: Improved Interpretation of Electrical Resistivity Measurements in
            Mixed-Wet Rocks: An Experimental Core-Scale Application and Model
            Verification
Newgord, Garcia, Rostami, Heidari (2018)
DOI: 10.30632/PJV59N5-2018a9

In mixed-wet rocks the Archie saturation exponent n is not constant: as oil wets
part of the pore surface, the conductive water becomes less connected, so the
resistivity index departs from the single-exponent Archie line and n increases
with the oil-wet fraction.  A wettability-dependent saturation exponent improves
the water-saturation interpretation.

Implements:

  - Archie resistivity index  I = Sw^(-n)
  - Wettability-dependent saturation exponent  n(oil-wet fraction)
  - Water saturation with the corrected n
  - Saturation-error from assuming a fixed n

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard wettability-dependent Archie relations the
paper's title describes.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Archie ------------------

def resistivity_index(sw, n=2.0):
    """Archie resistivity index  I = Rt/R0 = Sw^(-n)."""
    # NOTE: despite the name this is the power law in Sw, not the Rt/Ro
    # definition — it maps to resistivity_index_from_sw.
    return petrolib.saturation_resistivity.resistivity_index_from_sw(sw, n=n)


def saturation_exponent(oil_wet_fraction, n_ww=1.8, dn=2.0):
    """Saturation exponent rising with the oil-wet fraction  n = n_ww + dn*f_ow."""
    return n_ww + dn * np.asarray(oil_wet_fraction, float)


def water_saturation(I, n):
    """Water saturation from the resistivity index  Sw = I^(-1/n)."""
    return np.asarray(I, float) ** (-1.0 / n)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 10: Resistivity in Mixed-Wet Rocks")
    print("=" * 60)

    # Resistivity index rises as water saturation falls
    assert resistivity_index(0.4, 2.0) > resistivity_index(0.8, 2.0)

    # Saturation exponent increases with the oil-wet fraction
    n_ww = saturation_exponent(0.0)
    n_ow = saturation_exponent(1.0)
    print(f"  n water-wet / oil-wet  = {n_ww:.2f} / {n_ow:.2f}")
    assert n_ow > n_ww

    # Using the (too-low) water-wet n on a mixed-wet rock underestimates Sw
    # (and so overestimates the hydrocarbon volume)
    sw_true = 0.35
    f_ow = 0.6
    n_true = saturation_exponent(f_ow)
    I_meas = resistivity_index(sw_true, n_true)        # measured RI of the rock
    sw_fixed = water_saturation(I_meas, n_ww)          # interpret with fixed n
    sw_correct = water_saturation(I_meas, n_true)      # interpret with correct n
    print(f"  Sw correct / fixed-n   = {sw_correct:.3f} / {sw_fixed:.3f}")
    assert abs(sw_correct - sw_true) < 1e-9
    assert sw_fixed < sw_true                          # too-low n -> Sw underestimate
    print("  PASS")
    return {"n_oilwet": float(n_ow), "sw_fixed": float(sw_fixed),
            "sw_correct": float(sw_correct)}


if __name__ == "__main__":
    test_all()
