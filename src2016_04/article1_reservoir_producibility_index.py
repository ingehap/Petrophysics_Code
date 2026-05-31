"""
Article 1: The Reservoir Producibility Index: a Metric to Assess Reservoir
           Quality in Tight-Oil Plays from Logs
Reeder, Craddock, Rylander, Pirie, Lewis, Kausik, Kleinberg, Yang, Pomerantz (2016)
Reference: Petrophysics Vol. 57, No. 2 (April 2016), pp. 83-95
DOI: none assigned (this issue predates SPWLA DOI assignment)

In tight-oil plays the producible oil is a positive reservoir-quality (RQ)
indicator while the immobile kerogen and bitumen are negative indicators.  The
Oil Saturation Index (OSI = 100*S1/TOC) captures this from core/cuttings but
suffers from oil loss at surface and lean-zone ambiguity.  The Reservoir
Producibility Index (RPI) uses in-situ (log) data: it scales the OSI-like oil
fraction by the oil richness, with oil content WC_oil measured by NMR (T2 > 3
ms after removing bitumen / clay-bound water) and total organic carbon WC_org
from nuclear spectroscopy.

Implements:

  - Oil Saturation Index  OSI = 100*S1/TOC  (Eq. 1)
  - Reservoir Producibility Index  RPI = WC_oil^2/WC_org  (Eq. 2)
  - Dry-weight TOC -> WC_org conversion (Eq. 3)
  - Oil content WC_oil from NMR porosities (Eqs. 4-5)
  - Clay-bound water from wet-clay porosity and clay volume

Note: this issue's PDF has a text layer; the OSI/RPI and WC_org/WC_oil relations
(Eqs. 1-5) are transcribed from the body, while the typeset glyphs were dropped
and reconstructed in standard form.  Carbon/oil contents as mass fractions,
porosities/saturations as fractions, densities in g/cm^3, S1 in mg/g.
"""

import numpy as np

T2_CUTOFF_MS = 3.0            # bitumen / clay-bound water cutoff
CARBON_PER_CH2 = 12.0 / 14.0  # carbon mass fraction of a CH2 unit


# ---------------------------------------------- indices --------------

def osi(s1, toc):
    """Oil Saturation Index (Jarvie, 2012; Eq. 1)

        OSI = 100*S1/TOC   [mg oil / g TOC],

    with S1 the Rock-Eval thermally vaporizable oil and TOC the total organic
    carbon (weight fractions or consistent units).  OSI > 100 indicates a
    productive tight-oil target.
    """
    return 100.0 * np.asarray(s1, float) / toc


def rpi(wc_oil, wc_org):
    """Reservoir Producibility Index (Kausik et al., 2015a; Eq. 2)

        RPI = (WC_oil/WC_org) * WC_oil = WC_oil^2/WC_org,

    the OSI-like oil fraction scaled by the oil richness WC_oil (in-situ, from
    logs), so lean zones are not flagged as productive.
    """
    return np.asarray(wc_oil, float) ** 2 / wc_org


# ---------------------------------------------- log-derived inputs --------------

def wc_org_from_dry(wc_dry, phi_w, rho_w, rho_b):
    """Convert dry-weight TOC to total organic carbon per mass of formation (Eq. 3)

        WC_org = WC_dry * (1 - phi_w*rho_w/rho_b),

    accounting for the water mass in the as-received formation.
    """
    return wc_dry * (1.0 - phi_w * rho_w / rho_b)


def wc_oil(phi_t, phi_bitumen, phi_w, rho_oil, rho_b):
    """Oil-carbon content from NMR porosities (Eqs. 4-5)

        WC_oil = (phi_T - phi_bitumen - phi_w) * rho_oil * (12/14) / rho_b,

    where (phi_T - phi_bitumen - phi_w) is the oil volume fraction (NMR T2 > 3 ms
    with bitumen and clay-bound water removed) and 12/14 converts the oil (CH2)
    signal to its carbon mass.
    """
    return (phi_t - phi_bitumen - phi_w) * rho_oil * CARBON_PER_CH2 / rho_b


def clay_bound_water(phi_clay, v_clay):
    """Clay-bound water volume  CBW = phi_clay*V_clay  (phi_clay ~ 0.10 typical)."""
    return phi_clay * v_clay


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Reservoir Producibility Index")
    print("=" * 60)

    # OSI threshold: > 100 mg oil/g TOC indicates a productive target
    # (S1 in mg HC/g rock, TOC in wt%)
    assert osi(2.0, 1.0) > 100.0 and osi(0.3, 1.0) < 100.0
    print(f"  OSI (S1=2, TOC=1%)      = {osi(2.0, 1.0):.0f} mg/g")

    # WC_org from dry-weight TOC is reduced by the water mass
    wco = wc_org_from_dry(0.05, phi_w=0.05, rho_w=1.0, rho_b=2.4)
    print(f"  WC_org from dry         = {wco:.4f}")
    assert wco < 0.05

    # Oil content is positive only when oil volume exceeds bitumen + water
    wcoil = wc_oil(phi_t=0.10, phi_bitumen=0.02, phi_w=0.03, rho_oil=0.8, rho_b=2.4)
    print(f"  WC_oil                  = {wcoil:.4f}")
    assert wcoil > 0
    assert np.isclose(wc_oil(0.05, 0.02, 0.03, 0.8, 2.4), 0.0)   # no free oil

    # RPI rises with oil richness; lean zone (low WC_oil) gives low RPI
    rich = rpi(0.03, 0.06)
    lean = rpi(0.005, 0.06)
    print(f"  RPI rich/lean           = {rich:.5f} / {lean:.5f}")
    assert rich > lean > 0
    # Two zones with equal OSI-like ratio but different richness rank correctly
    assert rpi(0.04, 0.08) > rpi(0.02, 0.04)           # same ratio, more oil

    # Clay-bound water scales with clay volume
    assert np.isclose(clay_bound_water(0.10, 0.3), 0.03)
    print("  PASS")
    return {"WC_oil": float(wcoil), "RPI_rich": float(rich)}


if __name__ == "__main__":
    test_all()
