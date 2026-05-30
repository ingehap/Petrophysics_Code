"""
Article 6: NanoTags for Improved Cuttings Depth Correlation
Poitzsch, Zhu, Antoniv, Aljabri, Marsala (2021)
DOI: 10.30632/PJV62N6-2021a6

Mud loggers assign a depth of origin to drill cuttings via a lag time for
cuttings to travel up the annulus; this is uncertain because annular volume
varies with hole enlargement.  Engineered, barcoded NanoTags are injected
before the mud enters the drillstring; they travel DOWN the dimensionally-
known interior to tag freshly cut rock at the bit.  Because the downward lag
depends only on accurately-known interior volume, depth uncertainty drops from
+/-10-20 ft to about +/-2 ft.

Implements:

  - Upward (conventional) lag time   t_l = v_a / f             (Eq. 1)
  - Conventional generation time     t_g = t_c - t_l           (Eq. 2)
  - Downward (NanoTags) lag time     t_d = v_d / f             (Eq. 3)
  - NanoTag generation time          t_g = t + t_d             (Eq. 4)
  - Annular volume from hole/pipe diameters (capacity factor)
  - Depth error  = ROP * delta_t

Note: the journal's Eqs. 1-4 are image-rendered and were not in the text; the
forms here are faithful reconstructions of the volumetric lag-time algebra the
paper (and its nomenclature) describes.  Volumes in gallons, flow in gal/min,
lag in minutes, depths in feet.
"""

import numpy as np

GAL_PER_BBL = 42.0
# Annular/internal capacity factor: bbl per foot = (D_out^2 - D_in^2)/1029.4
_CAPACITY_DIVISOR = 1029.4


# ---------------------------------------------- Eqs. 1, 3: lag time -----

def lag_time(volume_gal, flow_gpm):
    """Volumetric lag time  t = V / f  (Eqs. 1 and 3).  Minutes."""
    return volume_gal / flow_gpm


# ---------------------------------------------- Eqs. 2, 4: gen. time ----

def generation_time_conventional(t_collect, lag_up):
    """Conventional cuttings generation time  t_g = t_c - t_l  (Eq. 2)."""
    return t_collect - lag_up


def generation_time_nanotag(t_inject, lag_down):
    """NanoTag generation time  t_g = t + t_d  (Eq. 4)."""
    return t_inject + lag_down


# ---------------------------------------------- volumes -----------------

def annular_capacity_bbl_per_ft(d_hole_in, d_pipe_in):
    """Annular capacity factor (bbl/ft) from hole & pipe diameters."""
    return (d_hole_in ** 2 - d_pipe_in ** 2) / _CAPACITY_DIVISOR


def annular_volume_gal(d_hole_in, d_pipe_in, length_ft):
    """Annular volume (gallons) over a length of hole."""
    return annular_capacity_bbl_per_ft(d_hole_in, d_pipe_in) * length_ft * GAL_PER_BBL


# ---------------------------------------------- depth error -------------

def depth_error_ft(rop_ft_per_hr, dt_min):
    """Depth error from a timing error  = ROP * dt."""
    return rop_ft_per_hr * (dt_min / 60.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: NanoTags for Cuttings Depth Correlation")
    print("=" * 60)

    f = 600.0                  # gal/min (driller nominal)

    # Downward lag: choose interior volume so t_d ~ 17 min (reservoir example)
    v_d = 10200.0              # gal of drillstring interior
    t_d = lag_time(v_d, f)
    print(f"  downward lag t_d       = {t_d:.1f} min (expect ~17)")
    assert abs(t_d - 17.0) < 0.1

    # Upward lag grows with hole enlargement: 8.375 in -> ~9 in
    pipe = 5.0
    L = 12000.0                # ft of annulus (vertical well depth)
    t_l_gauge = lag_time(annular_volume_gal(8.375, pipe, L), f)
    t_l_washed = lag_time(annular_volume_gal(9.0, pipe, L), f)
    print(f"  upward lag 8.375/9.0in = {t_l_gauge:.1f} / {t_l_washed:.1f} min")
    assert t_l_washed > t_l_gauge, "enlarged hole must lengthen the upward lag"

    # Round trip (down + up) and the 2-3x rule (t_l is 2-3x t_d)
    round_trip = t_d + t_l_gauge
    print(f"  round trip (down+up)   = {round_trip:.1f} min")
    assert 2.0 <= (t_l_gauge / t_d) <= 3.5

    # Generation-time conventions
    t_g_conv = generation_time_conventional(t_collect=54.0, lag_up=t_l_gauge)
    t_g_tag = generation_time_nanotag(t_inject=0.0, lag_down=t_d)
    print(f"  t_g conventional/tag   = {t_g_conv:.1f} / {t_g_tag:.1f} min")
    assert abs(t_g_tag - t_d) < 1e-9

    # A 2-min timing slip costs ~2 ft at 60 ft/hr (the caprock observation)
    err = depth_error_ft(60.0, 2.0)
    print(f"  depth error @60 ft/hr  = {err:.1f} ft for 2-min slip")
    assert abs(err - 2.0) < 1e-9
    assert depth_error_ft(20.0, 2.0) < err   # slower ROP -> smaller error
    print("  PASS")
    return {"t_d": t_d, "t_l_gauge": t_l_gauge, "round_trip": round_trip,
            "depth_error": err}


if __name__ == "__main__":
    test_all()
