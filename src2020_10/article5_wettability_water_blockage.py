"""
Article 5: Revisiting the Concept of Wettability for Organic-Rich Tight Rocks -
           Application in Formation Damage-Water Blockage
Mukherjee, Dang, Rai, Sondergeld (2020)
DOI: 10.30632/PJV61N5-2020a5

Organic-rich tight rocks contain water-wet inorganic pores, oil-wet organic
pores, and mixed-wet pores.  A spontaneous counter-imbibition experiment with
two fluids (brine = water-wet proxy, dodecane = oil-wet proxy) under step
pressurization partitions the pore system by wettability and explains
water-blockage formation damage: above a threshold capillary pressure the oil
phase regains continuity.

Implements:

  - Young-Laplace capillary pressure  Pc = 2*sigma*cos(theta)/r
  - Washburn pore-throat radius (inverse)
  - Wettability pore-type fractions from imbibed volumes      (Eqs. 1-3)
  - Water-blockage trapped-water saturation and the
    threshold capillary pressure to restore oil continuity

Note: this issue's PDF text layer dropped the typeset glyphs of Eqs. 1-3; the
pore-type partition here follows the described spontaneous-imbibition logic.
The paper's anchors are reproduced: 2.5 wt% KCl brine, step pressurization to
7,000 psi, and the ~1,500 psi threshold to restore oil-phase continuity.
"""

import numpy as np

PSI_TO_PA = 6894.76
THRESHOLD_PSI = 1500.0       # restore oil-phase continuity above this Pc
MAX_PRESS_PSI = 7000.0       # step-pressurization endpoint -> 100% saturation


# ---------------------------------------------- capillary --------------

def capillary_pressure(sigma_Nm, theta_deg, r_m):
    """Young-Laplace capillary pressure  Pc = 2*sigma*cos(theta)/r  (Pa)."""
    return 2.0 * sigma_Nm * np.cos(np.radians(theta_deg)) / np.asarray(r_m, float)


def pore_throat_radius(pc_pa, sigma_Nm, theta_deg):
    """Washburn pore-throat radius  r = 2*sigma*cos(theta)/Pc  (m)."""
    return 2.0 * sigma_Nm * np.cos(np.radians(theta_deg)) / np.asarray(pc_pa, float)


# ---------------------------------------------- wettability -------------

def pore_type_fractions(v_brine_spont, v_dodecane_spont, v_total):
    """Wettability pore-type fractions from spontaneous-imbibition volumes (Eqs. 1-3).

    Water-wet pores spontaneously imbibe brine; oil-wet pores spontaneously
    imbibe dodecane; the remainder (filled only under forced pressurization) is
    mixed-wet / unconnected.  Returns a dict summing to 1.
    """
    fw = v_brine_spont / v_total
    fo = v_dodecane_spont / v_total
    fm = max(1.0 - fw - fo, 0.0)
    return {"water_wet": fw, "oil_wet": fo, "mixed_unconnected": fm}


def water_blockage_saturation(sw_initial, sw_irreducible):
    """Trapped (blocking) water saturation above irreducible.

    The extra water held in water-wet pores beyond Swirr blocks hydrocarbon
    flow:  Sw_block = max(Sw_initial - Sw_irr, 0).
    """
    return max(sw_initial - sw_irreducible, 0.0)


def oil_continuity_restored(applied_pc_psi, threshold_psi=THRESHOLD_PSI):
    """True when the applied capillary pressure exceeds the restoration threshold."""
    return applied_pc_psi >= threshold_psi


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Wettability & Water Blockage (Tight Rocks)")
    print("=" * 60)

    # Capillary pressure / Washburn radius round-trip
    sigma, theta = 0.05, 30.0          # N/m, deg (water-wet)
    r = 10e-9                           # 10 nm throat
    pc = capillary_pressure(sigma, theta, r)
    print(f"  Pc at 10 nm            = {pc/PSI_TO_PA:.0f} psi")
    assert abs(pore_throat_radius(pc, sigma, theta) - r) < 1e-18
    # smaller throats need higher Pc (the water-blockage mechanism)
    assert capillary_pressure(sigma, theta, 5e-9) > pc

    # Pore-type fractions from an imbibition experiment (sum to 1)
    frac = pore_type_fractions(v_brine_spont=0.45, v_dodecane_spont=0.35,
                               v_total=1.0)
    print(f"  pore types             = {frac}")
    assert abs(sum(frac.values()) - 1.0) < 1e-9
    assert frac["water_wet"] > frac["oil_wet"] > 0

    # Water blockage: trapped water beyond irreducible
    sb = water_blockage_saturation(sw_initial=0.55, sw_irreducible=0.30)
    print(f"  blocking water sat     = {sb:.2f}")
    assert abs(sb - 0.25) < 1e-9

    # Oil continuity is restored only above the ~1,500 psi threshold, well
    # within the 7,000 psi step-pressurization endpoint
    assert not oil_continuity_restored(1000.0)
    assert oil_continuity_restored(2000.0)
    assert oil_continuity_restored(MAX_PRESS_PSI)
    print(f"  threshold / max press  = {THRESHOLD_PSI:.0f} / {MAX_PRESS_PSI:.0f} psi")
    print("  PASS")
    return {"Pc_10nm_psi": float(pc / PSI_TO_PA), "fractions": frac,
            "block_sat": sb}


if __name__ == "__main__":
    test_all()
