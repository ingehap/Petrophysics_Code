"""
Article 10: Enhanced Assessment of Fluid Saturation in the Wolfcamp Formation
            of the Permian Basin
Dash, Heidari (2021)
DOI: 10.30632/PJV62N6-2021a10

*Methodology proxy.*  This article's body was not present in the source-PDF
text extract used to build this folder (only the table of contents and the
editor's narrative were available).  This module is a methodology proxy that
demonstrates the resistivity-based rock-physics workflow the editor describes:
an integrated conductivity model for organic-rich mudrocks, an inversion that
recovers model parameters without core calibration, and improved water
saturation versus Archie's model.  The editor reports a ~33% improvement in
hydrocarbon-reserve estimates (about +70,000 bbl/acre) over Archie in the
Wolfcamp.

Implements:

  - Archie water saturation (baseline)
  - Dual-conductivity (Waxman-Smits-style) saturation for shaly/organic rock
  - Core-free inversion for the cementation exponent m from a clean-water zone
  - Hydrocarbon pore volume per acre and the Archie-vs-new improvement

These are standard correlations consistent with the editor's description;
replace with the paper's published model and parameters when its body is
available.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# bbl of pore volume per acre-ft (7758 bbl = 1 acre-ft)
BBL_PER_ACRE_FT = 7758.0


# ---------------------------------------------- Archie baseline ---------

def archie_sw(rt, phi, rw, a=1.0, m=2.0, n=2.0):
    """Archie  Sw = (a*Rw / (phi^m * Rt))^(1/n)."""
    # HAZARD (LIBRARY_MERGE_PLAN.md section 9): this article's argument order
    # is (rt, phi, rw) — the canonical order is (rt, rw, phi=).  Mapped
    # explicitly.
    return petrolib.saturation_resistivity.archie_sw(rt, rw, phi=phi, a=a, m=m, n=n, clip=(0.0, 1.0))


# ---------------------------------------------- dual-conductivity -------

def waxman_smits_sw(rt, phi, rw, qv, B, m_star=2.0, n_star=2.0, a=1.0,
                    iters=60):
    """Waxman-Smits saturation including clay/organic excess conductivity.

    Ct = phi^m* / a * Sw^n* * (1/Rw + B*Qv/Sw)   solved iteratively for Sw.
    The extra B*Qv conductance lowers Sw versus Archie in organic-rich rock.
    """
    ct = 1.0 / np.asarray(rt, float)
    phi = np.asarray(phi, float)
    f = phi ** m_star / a
    sw = archie_sw(rt, phi, rw, a=a, m=m_star, n=n_star)   # initial guess
    for _ in range(iters):
        # Ct = f * (Sw^n / Rw + B*Qv*Sw^(n-1))  -> solve for Sw (fixed point)
        sw = (ct / f - B * qv * sw ** (n_star - 1.0)) ** (1.0 / n_star) * rw ** (1.0 / n_star)
        sw = np.clip(sw, 1e-3, 1.0)
    return sw


# ---------------------------------------------- core-free inversion -----

def invert_cementation_m(rt_wet, phi, rw, a=1.0):
    """Recover the cementation exponent m from a 100%-water zone (no core).

    In a wet zone Sw = 1, so Rt = a*Rw/phi^m  =>  m = log(a*Rw/Rt)/log(phi).
    """
    rt_wet = np.asarray(rt_wet, float); phi = np.asarray(phi, float)
    return np.log(a * rw / rt_wet) / np.log(phi)


# ---------------------------------------------- reserves ----------------

def hydrocarbon_pv_per_acre(phi, sw, thickness_ft):
    """Hydrocarbon pore volume (bbl/acre) = 7758 * h * phi * (1 - Sw)."""
    return BBL_PER_ACRE_FT * thickness_ft * np.asarray(phi, float) * (1.0 - np.asarray(sw, float))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 10: Resistivity Rock-Physics in Wolfcamp (proxy)")
    print("=" * 60)

    rw = 0.04
    phi = 0.10
    rt = 25.0

    # Core-free inversion recovers a planted cementation exponent
    m_true = 1.9
    rt_wet = 1.0 * rw / phi ** m_true        # synthetic wet-zone resistivity
    m_hat = invert_cementation_m(rt_wet, phi, rw)
    print(f"  inverted m (core-free) = {m_hat:.3f} (true {m_true})")
    assert abs(m_hat - m_true) < 1e-6

    # New (dual-conductivity) Sw is lower than Archie in organic-rich rock
    sw_archie = archie_sw(rt, phi, rw, m=m_true, n=2.0)
    sw_new = waxman_smits_sw(rt, phi, rw, qv=0.6, B=4.0, m_star=m_true, n_star=2.0)
    print(f"  Sw Archie / new        = {sw_archie:.3f} / {sw_new:.3f}")
    assert sw_new < sw_archie, "excess conductivity must lower Sw vs Archie"

    # Reserves improvement (more hydrocarbon when Sw is lower)
    h = 100.0
    hcpv_archie = hydrocarbon_pv_per_acre(phi, sw_archie, h)
    hcpv_new = hydrocarbon_pv_per_acre(phi, sw_new, h)
    improvement = (hcpv_new - hcpv_archie) / hcpv_archie
    print(f"  HCPV Archie / new      = {hcpv_archie:.0f} / {hcpv_new:.0f} bbl/acre")
    print(f"  improvement            = {100*improvement:.0f}%")
    assert hcpv_new > hcpv_archie
    assert improvement > 0.0
    print("  PASS")
    return {"m_hat": m_hat, "sw_archie": float(sw_archie),
            "sw_new": float(sw_new), "improvement": float(improvement)}


if __name__ == "__main__":
    test_all()
