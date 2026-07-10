"""
Article 4: Microresistivity Curve Extraction from Borehole Microimager Data
Roslin (2015)
Reference: Petrophysics Vol. 56, No. 2 (April 2015), pp. 140-146
DOI: none assigned (this issue predates SPWLA DOI assignment)

An automatic workflow extracts a microresistivity curve from electrical
borehole-image (button-electrode) data: the button array is histogram-scaled,
a median curve is computed across the buttons at each depth, and that median
curve is calibrated to a wireline deep-resistivity log by a log-log
(power-law) crossplot fit.  Where no wireline resistivity is available, the
median curve is still usable as a relative resistivity.

Implements:

  - Histogram (min-max) scaling of image button data to a reference range
  - Median image curve across the button electrodes at each depth
  - Log-log microresistivity calibration  Rmicro = 10^(A + B*log10(Rmedian))  (Eq. 1)
  - Calibration fit of (A, B) from a crossplot against wireline resistivity

Note: this issue's PDF has a text layer; the calibration relation (Eq. 1) is
transcribed from the body, while the typeset glyphs were dropped and
reconstructed in standard log-log form.  Resistivity in ohm-m.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- image processing --------------

def histogram_scale(image, ref_min, ref_max):
    """Linear min-max histogram scaling of an image array onto [ref_min, ref_max]."""
    return petrolib.data_qc_io.scale.normalize_to_reference(image, ref_min, ref_max, pct=None)


def median_curve(image_array):
    """Median image curve across the button electrodes at each depth.

    `image_array` is (n_depths, n_buttons); returns the per-depth median, a
    robust single-curve summary of the microimage.
    """
    return np.median(np.asarray(image_array, float), axis=1)


# ---------------------------------------------- calibration --------------

def microresistivity_calibration(r_median, a, b):
    """Log-log microresistivity calibration (Eq. 1)

        Rmicro = 10^(A + B*log10(Rmedian)) = 10^A * Rmedian^B,

    mapping the (relative) image median curve to a calibrated resistivity using
    the crossplot intercept A and gradient B.
    """
    return 10.0 ** (a + b * np.log10(np.asarray(r_median, float)))


def fit_calibration(r_median, r_wireline):
    """Fit the calibration coefficients (A, B) from a log-log crossplot of the
    median image curve against the wireline deep resistivity

        log10(Rwireline) = A + B*log10(Rmedian).

    Returns (A, B).
    """
    lf = petrolib.inversion_numerics.fitting.fit_line(
        r_median, r_wireline, xform="log10", yform="log10")
    return lf.intercept, lf.slope


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Microresistivity Curve Extraction")
    print("=" * 60)

    # Histogram scaling maps the data range onto the reference range
    img = np.array([[2.0, 4.0, 6.0], [1.0, 5.0, 9.0]])
    scaled = histogram_scale(img, 0.0, 100.0)
    assert np.isclose(scaled.min(), 0.0) and np.isclose(scaled.max(), 100.0)

    # Median curve summarizes the buttons per depth
    med = median_curve(img)
    assert np.allclose(med, [4.0, 5.0])

    # Calibration fit recovers the (A, B) used to synthesize a wireline curve
    rmed = np.array([2.0, 5.0, 10.0, 20.0, 50.0])
    a_true, b_true = 0.3, 0.9
    rwl = 10.0 ** (a_true + b_true * np.log10(rmed))
    a_fit, b_fit = fit_calibration(rmed, rwl)
    print(f"  fitted A / B           = {a_fit:.3f} / {b_fit:.3f}")
    assert np.isclose(a_fit, a_true) and np.isclose(b_fit, b_true)

    # The calibrated microresistivity overlays the wireline curve
    r_micro = microresistivity_calibration(rmed, a_fit, b_fit)
    print(f"  Rmicro / Rwireline[0]  = {r_micro[0]:.3f} / {rwl[0]:.3f}")
    assert np.allclose(r_micro, rwl)
    print("  PASS")
    return {"A": float(a_fit), "B": float(b_fit), "Rmicro0": float(r_micro[0])}


if __name__ == "__main__":
    test_all()
