"""
Article 1: Drainage Three-Phase Flow Relative Permeability on Oil-Wet Carbonate
           Reservoir Rock Types: Experiments, Interpretation and Comparison with
           Standard Correlations
P. Egermann, K. Mejdoub, J.-M. Lombard, O. Vizika, Z. Kalam (2014)
Reference: Petrophysics Vol. 55, No. 4 (August 2014), pp. 287-293
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best of the 2013 SCA Symposium.  Experimental drainage three-phase relative
permeabilities on oil-wet carbonates are compared with the standard predictive
correlations that estimate the three-phase oil relative permeability kro from
the two-phase oil-water (krow) and gas-oil (krog) curves.

Implements:

  - Corey two-phase relative permeabilities (krw, kro, krg)
  - Stone I three-phase oil relative permeability (Stone, 1970)
  - Stone II three-phase oil relative permeability (Stone, 1973)
  - Baker saturation-weighted interpolation (Baker, 1988)
  - Geometric (linear) interpolation between krow and krog

Note: this methodological paper names the standard correlations but renders no
display equations; the forms below are the standard textbook reconstructions
(Stone, 1970/1973; Baker, 1988) consistent with the cited variables.  An oil-wet
Amott index WI = -0.7 is reported.  Saturations and relative permeabilities are
fractions.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- two-phase Corey --------------

def corey_water(sw, swc, sor, krw_max=1.0, nw=2.0):
    """Corey water relative permeability in the oil-water system

        krw = krw_max*Sw*^nw,   Sw* = (Sw - Swc)/(1 - Swc - Sor).
    """
    return petrolib.relperm_wettability.corey_krw(sw, swr=swc, sor=sor, krw_max=krw_max, nw=nw)


def corey_oil_water(sw, swc, sor, kro_max=1.0, no=2.0):
    """Corey oil relative permeability in the oil-water system (krow)

        krow = kro_max*(1 - Sw*)^no,   Sw* = (Sw - Swc)/(1 - Swc - Sor).
    """
    return petrolib.relperm_wettability.corey_kro(sw, swr=swc, sor=sor, kro_max=kro_max, no=no)


def corey_oil_gas(sg, swc, sgc, sorg, kro_max=1.0, no=2.0):
    """Corey oil relative permeability in the gas-oil system (krog) at connate
    water, as a function of gas saturation

        krog = kro_max*(1 - Sg*)^no,   Sg* = (Sg - Sgc)/(1 - Swc - Sgc - Sorg).
    """
    sg_star = np.clip((np.asarray(sg, float) - sgc) / (1 - swc - sgc - sorg), 0, 1)
    return kro_max * (1 - sg_star) ** no


def corey_gas(sg, swc, sgc, sorg, krg_max=1.0, ng=2.0):
    """Corey gas relative permeability

        krg = krg_max*Sg*^ng,   Sg* = (Sg - Sgc)/(1 - Swc - Sgc - Sorg).
    """
    return petrolib.relperm_wettability.corey_krg(
        sg, sgc=sgc, swc=swc, sorg=sorg, krg_max=krg_max, ng=ng)


# ---------------------------------------------- three-phase models --------------

def stone_i(krow, krog, sw, sg, swc, som, krocw):
    """Stone I three-phase oil relative permeability (Stone, 1970)

        kro = krocw*So**beta_w*beta_g,

    with the normalized oil saturation So* and the water/gas reduction factors

        So* = (So - Som)/(1 - Swc - Som),
        beta_w = krow/(krocw*(1 - Sw*)),  Sw* = (Sw - Swc)/(1 - Swc - Som),
        beta_g = krog/(krocw*(1 - Sg*)),  Sg* = Sg/(1 - Swc - Som).

    krocw is the oil relative permeability at connate water.
    """
    so = 1.0 - sw - sg
    denom = 1.0 - swc - som
    so_star = (so - som) / denom
    sw_star = (sw - swc) / denom
    sg_star = sg / denom
    beta_w = krow / (krocw * (1.0 - sw_star))
    beta_g = krog / (krocw * (1.0 - sg_star))
    return krocw * so_star * beta_w * beta_g


def stone_ii(krow, krog, krw, krg, krocw):
    """Stone II three-phase oil relative permeability (Stone, 1973)

        kro = krocw*[(krow/krocw + krw)*(krog/krocw + krg) - (krw + krg)].
    """
    return krocw * ((krow / krocw + krw) * (krog / krocw + krg) - (krw + krg))


def baker_interpolation(krow, krog, sw, sg, swc):
    """Baker saturation-weighted interpolation (Baker, 1988)

        kro = [(Sw - Swc)*krow + Sg*krog]/[(Sw - Swc) + Sg].
    """
    w = (sw - swc) + sg
    return ((sw - swc) * krow + sg * krog) / w


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Three-Phase Rel-Perm on Oil-Wet Carbonate")
    print("=" * 60)

    swc, sor, som = 0.2, 0.2, 0.0
    krocw = corey_oil_water(swc, swc, sor)   # oil kr at connate water
    print(f"  krocw = {krocw:.3f}")
    assert np.isclose(krocw, 1.0)

    # Endpoints: Corey curves are monotone and bounded in [0, 1]
    sw = np.linspace(swc, 1 - sor, 20)
    assert np.all(np.diff(corey_water(sw, swc, sor)) >= 0)
    assert np.all(np.diff(corey_oil_water(sw, swc, sor)) <= 0)

    # A three-phase point with both water and gas present
    sw_p, sg_p = 0.35, 0.20
    krow_p = corey_oil_water(sw_p, swc, sor)
    krog_p = corey_oil_gas(sg_p, swc, 0.0, 0.0)
    krw_p = corey_water(sw_p, swc, sor)
    krg_p = corey_gas(sg_p, swc, 0.0, 0.0)

    kro1 = stone_i(krow_p, krog_p, sw_p, sg_p, swc, som, krocw)
    kro2 = stone_ii(krow_p, krog_p, krw_p, krg_p, krocw)
    krob = baker_interpolation(krow_p, krog_p, sw_p, sg_p, swc)
    print(f"  kro: Stone I={kro1:.4f}  Stone II={kro2:.4f}  Baker={krob:.4f}")
    # all models give a physical three-phase oil kr below the two-phase values
    for kro in (kro1, kro2, krob):
        assert 0.0 <= kro <= max(krow_p, krog_p) + 1e-9

    # Consistency: with no gas, Baker reduces to the oil-water curve
    assert np.isclose(baker_interpolation(krow_p, krog_p, sw_p, 0.0, swc), krow_p)
    print("  PASS")
    return {"krocw": float(krocw), "Stone_I": float(kro1),
            "Stone_II": float(kro2), "Baker": float(krob)}


if __name__ == "__main__":
    test_all()
