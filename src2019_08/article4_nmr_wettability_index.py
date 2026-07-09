"""
Article 4: Practical Approach to Derive Wettability Index by NMR in Core
           Analysis Experiments
Looyestijn (2019)
DOI: 10.30632/PJV60N4-2019a4

NMR sees wettability through surface relaxation: the fluid that wets the pore
surface relaxes faster (shorter T2) than its bulk value, while the non-wetting
fluid relaxes near its bulk T2.  Comparing the surface-relaxation contribution
of the water and oil signals in a partially saturated core gives a wettability
index in [-1, +1] (+1 water-wet, -1 oil-wet, 0 neutral).

Implements:

  - Surface-relaxation rate of a phase  1/T2_surf = 1/T2_obs - 1/T2_bulk
  - NMR wettability index  WI = (Rw - Ro)/(Rw + Ro)
  - Bound/free partition helper (T2 cutoff)

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the issue's table of contents and these are
faithful standard-form reconstructions of the NMR surface-relaxation
wettability-index method the paper describes.  T2 in ms.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- surface relaxation ------

def surface_relaxation_rate(t2_obs, t2_bulk):
    """Surface-relaxation contribution  1/T2_surf = 1/T2_obs - 1/T2_bulk (1/ms).

    Larger when the phase is in contact with (wets) the pore surface; ~0 when
    the phase relaxes at its bulk value (non-wetting).
    """
    return max(1.0 / t2_obs - 1.0 / t2_bulk, 0.0)


def wettability_index(t2w_obs, t2w_bulk, t2o_obs, t2o_bulk):
    """NMR wettability index  WI = (Rw - Ro)/(Rw + Ro)  in [-1, +1].

    Rw, Ro = surface-relaxation rates of the water and oil signals.
    """
    Rw = surface_relaxation_rate(t2w_obs, t2w_bulk)
    Ro = surface_relaxation_rate(t2o_obs, t2o_bulk)
    if Rw + Ro == 0:
        return 0.0
    return float(petrolib.relperm_wettability.nmr_wettability_index(Rw, Ro))


def bound_free_fraction(t2_ms, amplitude, cutoff_ms=33.0):
    """Bound (<= cutoff) and free (> cutoff) fractions of a T2 distribution."""
    t2 = np.asarray(t2_ms, float); a = np.asarray(amplitude, float)
    tot = a.sum()
    bound = a[t2 <= cutoff_ms].sum() / tot
    return float(bound), float(1.0 - bound)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: NMR Wettability Index (Looyestijn)")
    print("=" * 60)

    # Water-wet rock: water strongly surface-relaxed (T2 << bulk), oil ~ bulk
    wi_ww = wettability_index(t2w_obs=30.0, t2w_bulk=3000.0,
                              t2o_obs=950.0, t2o_bulk=1000.0)
    print(f"  WI water-wet           = {wi_ww:.2f}")
    assert wi_ww > 0.5

    # Oil-wet rock: oil strongly surface-relaxed, water ~ bulk
    wi_ow = wettability_index(t2w_obs=2900.0, t2w_bulk=3000.0,
                              t2o_obs=40.0, t2o_bulk=1000.0)
    print(f"  WI oil-wet             = {wi_ow:.2f}")
    assert wi_ow < -0.5

    # Neutral / mixed: both phases share the surface -> WI near 0
    wi_mixed = wettability_index(t2w_obs=80.0, t2w_bulk=3000.0,
                                 t2o_obs=70.0, t2o_bulk=1000.0)
    print(f"  WI mixed-wet           = {wi_mixed:.2f}")
    assert abs(wi_mixed) < 0.4

    # Surface relaxation rate is zero when a phase relaxes at its bulk value
    assert surface_relaxation_rate(1000.0, 1000.0) == 0.0
    assert surface_relaxation_rate(50.0, 1000.0) > 0.0

    # Bound/free partition sums to 1
    bound, free = bound_free_fraction([5, 20, 60, 200], [0.2, 0.3, 0.3, 0.2])
    print(f"  bound / free           = {bound:.2f} / {free:.2f}")
    assert abs(bound + free - 1.0) < 1e-9 and abs(bound - 0.5) < 1e-9
    print("  PASS")
    return {"WI_ww": wi_ww, "WI_ow": wi_ow, "WI_mixed": wi_mixed}


if __name__ == "__main__":
    test_all()
