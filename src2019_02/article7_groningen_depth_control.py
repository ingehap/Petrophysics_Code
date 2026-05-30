"""
Article 7 (Technical Note): Connecting the Dots - The Relevance of Proper Depth
                            Control in the Discovery of the Groningen Field
Fokkema, Visser (2019)
DOI: 10.30632/PJV60N1Y2019a6

A historical technical note on how proper depth control was decisive in
recognizing the Groningen gas field: a depth mismatch between wells (or between
logging runs) shifts the apparent top and thickness of a reservoir and biases
the volumetric (gas-in-place) estimate, so tying logs to a common datum is
essential.  This module quantifies the impact of a depth mismatch.

Implements:

  - Depth shift between two logs from a marker tie
  - Net-pay-thickness error from a depth mismatch
  - Gas-in-place error from a depth/thickness mismatch
  - Datum correction (shift to a common reference)

Note: this issue's PDF has a text layer but the piece is a short technical note;
this module implements the standard depth-tie / volumetric-impact relations it
discusses (typeset glyphs were dropped in extraction).
"""

import numpy as np


# ---------------------------------------------- depth tie ---------------

def marker_shift(marker_depth_a, marker_depth_b):
    """Depth shift between two wells/runs from a common marker  (b - a)."""
    return marker_depth_b - marker_depth_a


def apply_datum_correction(depths, shift):
    """Shift a depth track to a common datum by subtracting the mismatch."""
    return np.asarray(depths, float) - shift


# ---------------------------------------------- volumetric impact -------

def net_pay_error(true_top, true_base, shift):
    """Net-pay thickness error if a top pick is mis-depthed by `shift`.

    A shift that moves the top into/out of the reservoir changes the counted
    thickness by up to |shift|.
    """
    true_thickness = true_base - true_top
    shifted_thickness = max(true_base - (true_top + shift), 0.0)
    return shifted_thickness - true_thickness


def gas_in_place(area, thickness, phi, sg, Bg):
    """Gas-in-place  GIIP = area*thickness*phi*Sg/Bg."""
    return area * thickness * phi * sg / Bg


def giip_error_fraction(true_thickness, thickness_error):
    """Fractional GIIP error from a thickness error (GIIP ~ thickness)."""
    return thickness_error / true_thickness


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Depth Control in the Groningen Discovery")
    print("=" * 60)

    # A marker mismatch quantifies the depth shift between two runs
    shift = marker_shift(2000.0, 2003.5)
    print(f"  marker depth shift     = {shift:.1f} m")
    assert abs(shift - 3.5) < 1e-9

    # Datum correction removes the shift (markers align)
    da = np.array([2000.0, 2050.0, 2100.0])
    db = da + shift
    assert np.allclose(apply_datum_correction(db, shift), da)

    # A mis-depthed top eats into the counted net pay
    err = net_pay_error(true_top=2010.0, true_base=2040.0, shift=6.0)
    print(f"  net-pay error (6 m top shift) = {err:.1f} m")
    assert abs(err + 6.0) < 1e-9                   # 6 m of pay lost

    # The thickness error propagates directly into GIIP
    giip_true = gas_in_place(1e6, 30.0, 0.18, 0.7, 0.005)
    giip_shifted = gas_in_place(1e6, 24.0, 0.18, 0.7, 0.005)
    frac = giip_error_fraction(30.0, -6.0)
    print(f"  GIIP error fraction    = {frac*100:.0f} %")
    assert abs((giip_shifted - giip_true) / giip_true - frac) < 1e-9
    assert abs(frac + 0.2) < 1e-9                  # 6/30 = 20% loss
    print("  PASS")
    return {"shift": float(shift), "giip_error_frac": float(frac)}


if __name__ == "__main__":
    test_all()
