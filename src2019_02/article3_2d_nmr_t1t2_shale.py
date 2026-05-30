"""
Article 3: Frequency and Temperature Dependence of 2D NMR T1-T2 Maps of Shale
Kausik, Freed, Fellah, Feng, Ling, Simpson (2019)
DOI: 10.30632/PJV60N1Y2019a2

Two-dimensional NMR T1-T2 maps are used to type fluids in shale: the T1/T2 ratio
separates bitumen / clay-bound water (high ratio) from movable water and light
hydrocarbon (low ratio).  The relaxation times - and hence the map - depend on
the measurement frequency (Larmor field) and temperature, which must be
accounted for when comparing lab and downhole conditions.

Implements:

  - T1/T2 ratio and fluid typing from a (T2, T1/T2) point
  - Temperature dependence  T2 ~ T/viscosity (faster relaxation when colder)
  - Frequency (field) dependence of the T1/T2 ratio
  - 2D map population fractions by fluid class

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the 2D NMR relaxation relations the paper analyzes.  T1, T2 in ms.
"""

import numpy as np


# ---------------------------------------------- T1/T2 typing ------------

def t1_t2_ratio(t1, t2):
    """T1/T2 ratio (dimensionless)."""
    return np.asarray(t1, float) / np.asarray(t2, float)


def fluid_type(t2_ms, ratio):
    """Classify a (T2, T1/T2) point into a shale NMR fluid class.

      ratio > 4                 -> bitumen / clay-bound (strongly restricted)
      2 <= ratio <= 4, T2 small -> capillary / structural water
      ratio < 2,  T2 large      -> movable water / light hydrocarbon
    """
    if ratio > 4.0:
        return "bitumen/clay-bound"
    if t2_ms < 10.0:
        return "capillary water"
    return "movable water/hydrocarbon"


# ---------------------------------------------- environment dependence --

def temperature_scaled_t2(t2_ref, T_ref_K, T_K, visc_ratio):
    """Temperature dependence of bulk T2  T2 ~ (T/viscosity).

        T2 = T2_ref * (T/T_ref) * (visc_ref/visc) = T2_ref*(T/T_ref)/visc_ratio
    visc_ratio = visc(T)/visc(T_ref) (< 1 when warmer, raising T2).
    """
    return t2_ref * (T_K / T_ref_K) / visc_ratio


def frequency_t1t2(ratio_ref, f_ref_MHz, f_MHz, alpha=0.3):
    """Frequency (field) dependence of the T1/T2 ratio  ratio ~ (f/f_ref)^alpha.

    Higher Larmor frequency increases T1 (and the T1/T2 contrast) for restricted
    fluids.
    """
    return ratio_ref * (f_MHz / f_ref_MHz) ** alpha


# ---------------------------------------------- map populations ---------

def map_fractions(t2_list, t1_list, amplitudes):
    """Fractional populations of each fluid class on a T1-T2 map."""
    classes = {}
    a = np.asarray(amplitudes, float); tot = a.sum()
    for t2, t1, amp in zip(t2_list, t1_list, a):
        c = fluid_type(t2, t1_t2_ratio(t1, t2))
        classes[c] = classes.get(c, 0.0) + amp / tot
    return classes


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Frequency/Temperature Dependence of 2D NMR T1-T2")
    print("=" * 60)

    # Fluid typing across the T1/T2 map
    assert fluid_type(2.0, t1_t2_ratio(12.0, 2.0)) == "bitumen/clay-bound"   # ratio 6
    assert fluid_type(5.0, t1_t2_ratio(15.0, 5.0)) == "capillary water"      # ratio 3
    assert fluid_type(200.0, t1_t2_ratio(300.0, 200.0)) == "movable water/hydrocarbon"

    # Temperature: warming (lower viscosity) lengthens T2
    t2_warm = temperature_scaled_t2(50.0, 300.0, 350.0, visc_ratio=0.6)
    print(f"  T2 ref/warm            = 50 / {t2_warm:.1f} ms")
    assert t2_warm > 50.0

    # Frequency: higher Larmor field raises the T1/T2 ratio of restricted fluids
    r_hi = frequency_t1t2(3.0, 2.0, 20.0)
    print(f"  T1/T2 at 2/20 MHz      = 3.0 / {r_hi:.2f}")
    assert r_hi > 3.0

    # Map populations sum to 1 and split between bound and movable fluids
    frac = map_fractions([2.0, 5.0, 300.0], [12.0, 15.0, 300.0], [0.4, 0.3, 0.3])
    print(f"  map fractions          = { {k: round(v,2) for k,v in frac.items()} }")
    assert abs(sum(frac.values()) - 1.0) < 1e-9
    assert "bitumen/clay-bound" in frac
    print("  PASS")
    return {"t2_warm": float(t2_warm), "ratio_20MHz": float(r_hi)}


if __name__ == "__main__":
    test_all()
