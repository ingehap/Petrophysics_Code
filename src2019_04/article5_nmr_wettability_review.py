"""
Article 5: A Review of 60 Years of NMR Wettability
Valori, Nicot (2019)
DOI: 10.30632/PJV60N2-2019a3

A review of NMR-based wettability evaluation.  NMR senses wettability through
surface relaxation: the wetting fluid contacts the pore surface and relaxes
faster (shorter T2) than its bulk value.  An NMR wettability index is built by
placing the measured fluid relaxation between fully water-wet and fully oil-wet
reference states (an Amott-style index in the relaxation-rate domain).

Implements:

  - Surface relaxation  1/T2 = 1/T2_bulk + rho*(S/V)
  - Effective surface relaxivity vs contact angle
  - NMR wettability index from the water relaxation rate between end states
  - T2 log-mean of a distribution

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the NMR-wettability relations the review surveys.  T2 in ms; rho in um/ms.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- surface relaxation ------

def surface_relaxation_t2(rho, s_over_v, t2_bulk=3000.0):
    """Total T2 from bulk + surface relaxation  1/T2 = 1/T2_bulk + rho*(S/V) (ms).

    rho in um/ms, S/V in 1/um -> rho*(S/V) in 1/ms.
    """
    return petrolib.nmr.t2_apparent(t2_bulk=t2_bulk, rho=rho, s_over_v=s_over_v)


def effective_relaxivity(rho_strong, contact_angle_deg):
    """Effective surface relaxivity scaled by wetting strength  rho_eff = rho*cos(theta).

    A water-wet surface (theta -> 0) gives the full relaxivity; as the surface
    becomes oil-wet (theta -> 180) the water relaxivity vanishes.
    """
    return rho_strong * max(np.cos(np.radians(contact_angle_deg)), 0.0)


def t2_logmean(t2, amplitude):
    """Log-mean T2 of a distribution."""
    return petrolib.nmr.t2_logmean(t2, amplitude)


# ---------------------------------------------- wettability index -------

def nmr_wettability_index(t2_measured, t2_water_wet, t2_oil_wet):
    """Amott-style NMR wettability index in [-1, +1] from water T2 end states.

    Works in the relaxation-rate domain:
        WI = 2*(R_meas - R_ow)/(R_ww - R_ow) - 1,   R = 1/T2
    +1 when the water relaxes like the fully water-wet reference (fast),
    -1 when it relaxes like the fully oil-wet reference (slow, near bulk).
    """
    R_meas, R_ww, R_ow = 1.0 / t2_measured, 1.0 / t2_water_wet, 1.0 / t2_oil_wet
    return float(2.0 * (R_meas - R_ow) / (R_ww - R_ow) - 1.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: A Review of 60 Years of NMR Wettability")
    print("=" * 60)

    # Surface relaxation: smaller pores (higher S/V) relax faster
    t2_big = surface_relaxation_t2(0.005, 1.0 / 50.0)     # 50 um pore
    t2_small = surface_relaxation_t2(0.005, 1.0 / 5.0)    # 5 um pore
    print(f"  T2 big/small pore      = {t2_big:.0f} / {t2_small:.0f} ms")
    assert t2_big > t2_small

    # Effective relaxivity: full when water-wet, ~0 when oil-wet
    assert effective_relaxivity(0.01, 0.0) > effective_relaxivity(0.01, 80.0)
    assert effective_relaxivity(0.01, 180.0) == 0.0

    # T2 log-mean of a bimodal distribution lies between the modes
    lm = t2_logmean([10.0, 100.0, 1000.0], [0.3, 0.4, 0.3])
    assert 10.0 < lm < 1000.0

    # NMR wettability index: end states map to +1 (water-wet) and -1 (oil-wet)
    t2_ww, t2_ow = 30.0, 2000.0
    assert abs(nmr_wettability_index(t2_ww, t2_ww, t2_ow) - 1.0) < 1e-9
    assert abs(nmr_wettability_index(t2_ow, t2_ww, t2_ow) + 1.0) < 1e-9
    # an intermediate measured T2 gives a mixed-wet index near 0
    t2_mid = 2.0 / (1.0 / t2_ww + 1.0 / t2_ow)            # rate-domain midpoint
    wi_mid = nmr_wettability_index(t2_mid, t2_ww, t2_ow)
    print(f"  WI mixed-wet           = {wi_mid:.2f}")
    assert abs(wi_mid) < 0.1
    print("  PASS")
    return {"t2_small": float(t2_small), "WI_mixed": wi_mid}


if __name__ == "__main__":
    test_all()
