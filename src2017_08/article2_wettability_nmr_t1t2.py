"""
Article 2: Laboratory and Downhole Wettability from NMR T1/T2 Ratio
Valori, Hursan, Ma (2017)
Reference: Petrophysics Vol. 58, No. 4 (August 2017), pp. 352-365
DOI: none assigned (this issue predates SPWLA DOI assignment)

The NMR T1/T2 ratio of the oil phase correlates with rock wettability (it
approaches 1 for water-wet rock and rises as the rock becomes oil-wet), giving
an in-situ wettability indicator.  The observed relaxation combines bulk and
surface terms; a pore-volume-weighted mean T1/T2 of the oil phase is mapped to a
renormalized USBM wettability index by a linear calibration.

Implements:

  - Observed relaxation time  1/T_obs = 1/T_bulk + 1/T_surface
  - Pore-volume-weighted mean T1/T2 over an NMR distribution
  - Linear T1/T2 -> USBM* wettability calibration and its application
  - Surface affinity index  A = tau_s/tau_m

Note: this issue's PDF has a text layer; the mean-ratio and inversion equations
lost their typeset glyphs, so they are faithful standard-form reconstructions.
The calibration points (oil-phase T1/T2 vs USBM*) are transcribed from the paper:
(3.72, -0.975), (1.68, 0.237), (1.16, 0.5).  USBM* runs -1 (oil-wet) to +1.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# Reported calibration points: oil-phase T1/T2 -> renormalized USBM index
CALIB_T1T2 = np.array([3.72, 1.68, 1.16])
CALIB_USBM = np.array([-0.975, 0.237, 0.5])


# ---------------------------------------------- relaxation --------------

def observed_relaxation_time(t_bulk, t_surface):
    """Observed relaxation time  1/T_obs = 1/T_bulk + 1/T_surface  (Eq. 1)."""
    return petrolib.nmr.combine_relaxation_times(t_bulk, t_surface)


def mean_t1t2(ratios, weights):
    """Pore-volume-weighted mean T1/T2 over an NMR distribution  (Eq. 5)."""
    r = np.asarray(ratios, float)
    w = np.asarray(weights, float)
    return float(np.sum(r * w) / np.sum(w))


# ---------------------------------------------- wettability --------------

def calibrate_wettability(t1t2=CALIB_T1T2, usbm=CALIB_USBM):
    """Fit the linear oil-phase T1/T2 -> USBM* calibration (slope, intercept)."""
    slope, intercept = np.polyfit(t1t2, usbm, 1)
    return slope, intercept


def wettability_index(t1t2_oil, slope=None, intercept=None):
    """Renormalized USBM wettability index from the oil-phase T1/T2 (Eq. 6).

    Clipped to [-1, 1]; +1 water-wet, -1 oil-wet.
    """
    if slope is None or intercept is None:
        slope, intercept = calibrate_wettability()
    return float(np.clip(slope * t1t2_oil + intercept, -1.0, 1.0))


def affinity_index(tau_surface, tau_motion):
    """Surface affinity index  A = tau_s/tau_m (surface steps before bulk return)."""
    return tau_surface / tau_motion


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Wettability from NMR T1/T2")
    print("=" * 60)

    # Observed relaxation is faster (shorter) than either contribution
    assert observed_relaxation_time(1.9, 0.1) < 0.1

    # Pore-volume-weighted mean ratio
    assert np.isclose(mean_t1t2([1.0, 2.0, 4.0], [0.5, 0.3, 0.2]), 0.5 + 0.6 + 0.8)

    # Calibration: oil-wet (high T1/T2) -> negative USBM*, water-wet -> positive
    ww = wettability_index(1.16)
    ow = wettability_index(3.72)
    print(f"  USBM* water-wet/oil-wet = {ww:+.3f} / {ow:+.3f}")
    assert ww > 0 > ow

    # Worked example <T1/T2>oil = 1.488 -> mildly water-wet, between the calibrants
    wi = wettability_index(1.488)
    print(f"  USBM* at T1/T2=1.488   = {wi:+.3f}")
    assert 0.0 < wi < 0.5

    # Affinity index
    assert affinity_index(2.0, 0.5) == 4.0
    print("  PASS")
    return {"USBM_1488": wi, "USBM_oilwet": ow}


if __name__ == "__main__":
    test_all()
