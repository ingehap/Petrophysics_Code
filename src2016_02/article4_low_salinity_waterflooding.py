"""
Article 4: Low-Salinity Waterflooding: Facts, Inconsistencies and the Way Forward
Hamon (2016)
Reference: Petrophysics Vol. 57, No. 1 (February 2016), pp. 41-50
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best Papers of the 2015 SCA Symposium.  A critical review of low-salinity water
injection (LSWI): many tertiary LSWI corefloods do not, in fact, produce
significant incremental oil within a few pore volumes, even when the "required
conditions" (clay, connate water, mixed wettability) are met.  The discussion
turns on Buckley-Leverett fractional-flow theory (the saturation shock, and the
double shock when wettability is altered), residual-oil-saturation reduction
(Sorw high- vs low-salinity), dispersion in the water phase, and Amott
wettability.

Implements:

  - Fractional flow of water (Buckley-Leverett)
  - Corey two-phase relative permeability (water and oil)
  - Welge tangent: shock-front and average saturation behind the front
  - Recovery factor and LSWI incremental recovery (Sorw_HS - Sorw_LS)
  - Amott-Harvey wettability index

Note: this is a review/discussion paper with no original display equations; the
relations below are the standard waterflood / wettability analysis tools it
relies on (Buckley-Leverett 1942, Welge 1952, Amott 1959).  Saturations and
fractional flow dimensionless, viscosities in cP (consistent).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- relative permeability --------------

def _normalized_sw(sw, swc, sorw):
    return (np.asarray(sw, float) - swc) / (1.0 - swc - sorw)


def corey_relperm(sw, swc, sorw, krw_max=0.3, kro_max=1.0, nw=3.0, no=2.0):
    """Corey water and oil relative permeabilities

        krw = krw_max*Swn^nw,   kro = kro_max*(1 - Swn)^no.
    Returns (krw, kro).
    """
    return petrolib.relperm_wettability.corey_kr(
        sw, swr=swc, sor=sorw, krw_max=krw_max, kro_max=kro_max, nw=nw, no=no)


# ---------------------------------------------- fractional flow --------------

def fractional_flow_water(krw, kro, mu_w, mu_o):
    """Water fractional flow (Buckley-Leverett, neglecting gravity/capillarity)

        fw = (krw/mu_w) / (krw/mu_w + kro/mu_o).
    """
    return petrolib.relperm_wettability.fractional_flow(krw, kro, mu_w=mu_w, mu_nw=mu_o)


def welge_shock_front(swc, sorw, mu_w, mu_o, krw_max=0.3, kro_max=1.0, nw=3.0, no=2.0,
                      n=2000):
    """Welge (1952) tangent construction for the waterflood shock front.

    Builds the fw(Sw) curve and finds the shock-front saturation Swf as the
    tangent from (Swc, 0); returns (Swf, fwf, Sw_avg) where Sw_avg is the average
    water saturation behind the front at breakthrough:
        Sw_avg = Swf + (1 - fwf)/(dfw/dSw at Swf).
    """
    sw = np.linspace(swc + 1e-4, 1.0 - sorw - 1e-4, n)
    krw, kro = corey_relperm(sw, swc, sorw, krw_max, kro_max, nw, no)
    fw = fractional_flow_water(krw, kro, mu_w, mu_o)
    # The grid/Corey/fw setup is this article's; delegate the tangent construction.
    return petrolib.relperm_wettability.welge_shock(sw, fw, swc)


# ---------------------------------------------- recovery / wettability --------------

def recovery_factor(soi, sor):
    """Oil recovery factor  RF = (Soi - Sor)/Soi  (fraction of oil in place)."""
    return petrolib.relperm_wettability.displacement_efficiency(soi, sor)


def lswi_incremental_recovery(sorw_hs, sorw_ls):
    """Incremental oil from low-salinity injection

        delta = Sorw_HS - Sorw_LS,

    the reduction in waterflood residual oil saturation (pore-volume fraction)
    achieved by switching from high- to low-salinity brine.
    """
    return sorw_hs - sorw_ls


def amott_harvey_index(v_sp_water, v_total_water, v_sp_oil, v_total_oil):
    """Amott-Harvey wettability index

        I_AH = Vsp_water/Vtot_water - Vsp_oil/Vtot_oil,

    in [-1, 1]: positive = water-wet, ~0 = mixed/neutral, negative = oil-wet.
    """
    # amott_indices takes (spont, forced); this article passes totals, so the
    # forced volume is total - spont.  It returns (Iw, Io, Iah); we want Iah.
    return petrolib.relperm_wettability.amott_indices(
        v_sp_water, v_total_water - v_sp_water,
        v_sp_oil, v_total_oil - v_sp_oil)[2]


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Low-Salinity Waterflooding")
    print("=" * 60)

    # Fractional flow is 0 at connate water and rises monotonically with Sw
    krw0, kro0 = corey_relperm(0.2, 0.2, 0.2)
    assert np.isclose(fractional_flow_water(krw0, kro0, 1.0, 5.0), 0.0)
    krw1, kro1 = corey_relperm(0.5, 0.2, 0.2)
    fw_mid = fractional_flow_water(krw1, kro1, 1.0, 5.0)
    assert 0.0 < fw_mid < 1.0

    # Welge shock front lies between connate water and 1 - Sorw
    swf, fwf, sw_avg = welge_shock_front(swc=0.2, sorw=0.2, mu_w=1.0, mu_o=5.0)
    print(f"  shock Swf / fwf        = {swf:.3f} / {fwf:.3f}")
    print(f"  avg Sw behind front    = {sw_avg:.3f}")
    assert 0.2 < swf < sw_avg < 0.8

    # Recovery factor and LSWI increment
    rf = recovery_factor(soi=0.8, sor=0.3)
    inc = lswi_incremental_recovery(sorw_hs=0.35, sorw_ls=0.28)
    print(f"  recovery factor        = {rf:.3f}")
    print(f"  LSWI incremental Sor   = {inc:.3f}")
    assert np.isclose(rf, 0.625) and np.isclose(inc, 0.07)

    # Amott-Harvey index: water-wet positive, oil-wet negative
    assert amott_harvey_index(0.6, 0.7, 0.05, 0.6) > 0      # mostly spontaneous water imbibition
    assert amott_harvey_index(0.05, 0.6, 0.6, 0.7) < 0
    print("  PASS")
    return {"Swf": swf, "RF": float(rf), "LSWI_inc": float(inc)}


if __name__ == "__main__":
    test_all()
