"""
Article 7: Effect of Injection Pressure on the Imbibition Relative Permeability
           and Capillary Pressure Curves of Shale Gas Matrix
Al-Ameri, Mazeel (2020)
DOI: 10.30632/PJV61N2-2020a7

Water imbibition into a shale-gas matrix is characterized by capillary pressure
and relative permeability curves, and how they shift with injection pressure:
higher injection pressure forces more water into the matrix (raising the
trapped-water saturation), suppressing gas relative permeability.  This module
implements the standard Brooks-Corey capillary-pressure and relative-permeability
relations the workflow relies on.

Implements:

  - Brooks-Corey capillary pressure  Pc = Pe*Se^(-1/lambda)
  - Brooks-Corey wetting / nonwetting (gas) relative permeability
  - Injection-pressure effect: trapped-water saturation vs injection pressure

Note: this issue's source-PDF text extract ended before this article (it was
present only as a table-of-contents entry), so this module is a faithful
methodology proxy implementing the standard Brooks-Corey imbibition relations
the paper's title describes.  Saturations as fractions; pressures in psi.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Brooks-Corey -----------

def effective_saturation(sw, swr, snwr=0.0):
    """Normalized wetting saturation  Se = (Sw - Swr)/(1 - Swr - Snwr)."""
    return np.clip((np.asarray(sw, float) - swr) / (1.0 - swr - snwr), 0.0, 1.0)


def brooks_corey_pc(sw, swr, pe, lam):
    """Brooks-Corey capillary pressure  Pc = Pe*Se^(-1/lambda)."""
    # Normalize and guard-clip Se locally (this article's convention), then
    # delegate the Pe*Se^(-1/lam) kernel (Se pre-normalized -> swirr=0).
    se = effective_saturation(sw, swr)
    se = np.clip(se, 1e-6, 1.0)
    return petrolib.capillary_pressure.brooks_corey_pc(
        se, pc_entry=pe, lam=lam, swirr=0.0)


def brooks_corey_krw(sw, swr, lam, snwr=0.0):
    """Brooks-Corey wetting (water) rel-perm  krw = Se^((2+3*lambda)/lambda)."""
    se = effective_saturation(sw, swr, snwr)
    return se ** ((2.0 + 3.0 * lam) / lam)


def brooks_corey_krg(sw, swr, lam, snwr=0.0):
    """Brooks-Corey nonwetting (gas) rel-perm  krg = (1-Se)^2*(1-Se^((2+lambda)/lambda))."""
    se = effective_saturation(sw, swr, snwr)
    return (1.0 - se) ** 2 * (1.0 - se ** ((2.0 + lam) / lam))


# ---------------------------------------------- injection-pressure -----

def trapped_water_saturation(inj_pressure_psi, swr0=0.20, dsw_dp=2e-5, sw_max=0.65):
    """Trapped-water saturation rising with injection pressure (forced imbibition).

        Sw_trap = min(Swr0 + dSw_dP * P_inj, Sw_max)
    Higher injection pressure pushes more water into the shale matrix.
    """
    return min(swr0 + dsw_dp * inj_pressure_psi, sw_max)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Shale Imbibition Rel-Perm & Capillary Pressure")
    print("=" * 60)

    swr, pe, lam = 0.2, 500.0, 1.5

    # Capillary pressure decreases as water saturation increases
    pc = brooks_corey_pc(np.array([0.3, 0.5, 0.8]), swr, pe, lam)
    print(f"  Pc @Sw=0.3/0.5/0.8     = {np.array2string(pc, precision=0)} psi")
    assert np.all(np.diff(pc) < 0) and np.all(pc >= pe)

    # Relative-permeability endpoints and crossover
    sw = np.linspace(swr, 1.0, 17)
    krw = brooks_corey_krw(sw, swr, lam)
    krg = brooks_corey_krg(sw, swr, lam)
    assert abs(krw[0]) < 1e-9 and abs(krw[-1] - 1.0) < 1e-9
    assert krg[0] > krg[-1]                              # gas kr falls as water rises
    assert np.all(np.diff(krw) >= -1e-12)

    # Injection-pressure effect: more injection pressure -> more imbibed water.
    # Evaluate gas rel-perm at the resulting water saturation (fixed connate
    # Swr): higher injection pressure -> higher Sw -> lower krg.
    sw_lo = trapped_water_saturation(1000.0)
    sw_hi = trapped_water_saturation(15000.0)
    print(f"  imbibed Sw @1000/15000 psi = {sw_lo:.2f} / {sw_hi:.2f}")
    assert sw_hi > sw_lo
    krg_lo = brooks_corey_krg(sw_lo, swr, lam)
    krg_hi = brooks_corey_krg(sw_hi, swr, lam)
    print(f"  krg low/high inj P     = {krg_lo:.3f} / {krg_hi:.3f}")
    assert krg_hi < krg_lo
    print("  PASS")
    return {"sw_trap_hi": sw_hi, "krg_lowP": float(krg_lo), "krg_hiP": float(krg_hi)}


if __name__ == "__main__":
    test_all()
