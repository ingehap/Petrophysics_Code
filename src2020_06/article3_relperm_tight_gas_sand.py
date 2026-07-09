"""
Article 3: Reliable Measurement of Saturation-Dependent Relative Permeability in
           Tight Gas Sand Formations
Gonzalez, Tandon, Heidari, Gramin, Merle (2020)
DOI: 10.30632/PJV61N3-2020a3

Relative permeability in tight gas sands is measured with stationary-liquid
stepwise desaturation (porous plate + centrifuge) monitored by NMR, plus
unsteady-state gas injection processed with the JBN method.  Gas relative
permeability is fitted with a modified Corey-Brooks model and brine relative
permeability with an SDR (NMR T2) model.

Implements:

  - Centrifuge capillary pressure  Pc = 0.5*drho*w^2*(LR^2-(LR-L)^2)  (Eq. 1)
  - Modified Corey-Brooks gas rel-perm                                (Eq. 2)
  - SDR (NMR T2) brine rel-perm                                       (Eqs. 3-4)
  - Klinkenberg gas-slippage correction

Note: this issue's PDF text layer kept the equation numbers and variable
definitions but dropped the typeset glyphs, so these are the standard
centrifuge / Corey-Brooks / SDR forms anchored to those definitions.  Paper
anchors: Corey exponent ng in 0.5-3.75, Klinkenberg gas permeability
0.009-2.120 uD, method reliable for porosity > 0.10.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- centrifuge Pc -----------

def centrifuge_pc(drho, rpm, L, LR):
    """Centrifuge capillary pressure  Pc = 0.5*drho*w^2*(LR^2-(LR-L)^2)  (Eq. 1).

    drho kg/m^3, L (core length) and LR (rotation-center to outer face) in m,
    rpm -> w = 2*pi*rpm/60.  Returns Pa.
    """
    # Hassler-Brunner with the two radii as (LR - L) inner and LR outer face;
    # rpm -> omega conversion kept here at the facade.
    omega = 2.0 * np.pi * rpm / 60.0
    return petrolib.capillary_pressure.centrifuge_pc(
        omega, delta_rho=drho, r1=LR - L, r2=LR)


# ---------------------------------------------- Corey-Brooks ------------

def corey_brooks_krg(sw, swr, sgc, ng, krg_max=1.0):
    """Modified Corey-Brooks gas relative permeability  (Eq. 2).

        krg = krg_max * [(1 - Sw - Sgc)/(1 - Swr - Sgc)]^ng
    Clipped to [0, krg_max]; zero once gas is immobile (Sw >= 1 - Sgc).
    """
    # Sw-framework gas maps to the library's Sg = 1 - Sw form (swc=Swr, sorg=0);
    # the library's Se clip [0,1] reproduces this article's num + output clips.
    return petrolib.relperm_wettability.corey_krg(
        1.0 - np.asarray(sw, float), sgc=sgc, swc=swr, sorg=0.0, krg_max=krg_max, ng=ng)


# ---------------------------------------------- SDR brine kr ------------

def sdr_krw(sw, t2lm, t2lm100, a=1.0, m=1.0, n=2.0):
    """SDR (NMR) brine relative permeability  krw = a*(T2LM/T2LM100)^m*Sw^n  (Eq. 3)."""
    return a * (np.asarray(t2lm, float) / t2lm100) ** m * np.asarray(sw, float) ** n


# ---------------------------------------------- Klinkenberg -------------

def klinkenberg(k_inf, b, p_mean):
    """Gas-slippage corrected permeability  k_app = k_inf*(1 + b/p_mean)."""
    return k_inf * (1.0 + b / np.asarray(p_mean, float))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Relative Permeability in Tight Gas Sand")
    print("=" * 60)

    # Centrifuge Pc rises with rotational speed
    pc_lo = centrifuge_pc(200.0, 3000.0, 0.05, 0.10)
    pc_hi = centrifuge_pc(200.0, 6000.0, 0.05, 0.10)
    print(f"  Pc 3000/6000 rpm       = {pc_lo:.0f} / {pc_hi:.0f} Pa")
    assert pc_hi > pc_lo and abs(pc_hi / pc_lo - 4.0) < 1e-6   # ~ w^2

    # Corey-Brooks krg: max at irreducible water, zero once gas is immobile
    swr, sgc, ng = 0.30, 0.05, 2.0
    assert abs(corey_brooks_krg(swr, swr, sgc, ng) - 1.0) < 1e-9
    assert corey_brooks_krg(1.0 - sgc, swr, sgc, ng) < 1e-12
    # monotonically decreasing as water saturation rises
    sw = np.linspace(swr, 1 - sgc, 12)
    krg = corey_brooks_krg(sw, swr, sgc, ng)
    assert np.all(np.diff(krg) <= 1e-12)
    # the exponent sits in the paper's 0.5-3.75 range
    for ng_test in (0.5, 1.5, 3.75):
        assert corey_brooks_krg(0.6, swr, sgc, ng_test) > 0
    print(f"  krg(0.6) ng=2          = {corey_brooks_krg(0.6, swr, sgc, 2.0):.3f}")

    # SDR brine kr: zero at zero water, unity at full saturation
    assert abs(sdr_krw(1.0, 100.0, 100.0, a=1.0, m=1.0, n=2.0) - 1.0) < 1e-9
    assert sdr_krw(0.5, 60.0, 100.0) < sdr_krw(0.8, 90.0, 100.0)

    # Klinkenberg: apparent gas permeability falls toward k_inf as pressure rises
    k_lo = klinkenberg(1.0, 20.0, 100.0)
    k_hi = klinkenberg(1.0, 20.0, 1000.0)
    print(f"  k_app 100/1000 psi     = {k_lo:.3f} / {k_hi:.3f}")
    assert k_lo > k_hi > 1.0
    print("  PASS")
    return {"pc_hi": float(pc_hi), "krg_06": float(corey_brooks_krg(0.6, swr, sgc, 2.0)),
            "k_app_low": float(k_lo)}


if __name__ == "__main__":
    test_all()
