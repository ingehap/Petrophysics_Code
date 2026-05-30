"""
Article 4: Crushed-Rock Analysis Workflow Based on Advanced Fluid
           Characterization for Improved Interpretation of Core Data
Nikitin, Durand, McMullen, Blount, Driskill, Hows (2019)
DOI: 10.30632/PJV60N6-2019a4

An improved GRI crushed-rock retort workflow ("GRI+") for tight, liquid-rich
mudstones.  NMR T2 on as-received and crushed samples quantifies the fluid lost
during crushing (beta_crush); near-100%-efficiency closed retorting (alpha_ret)
and a bulk/grain volume balance reconstruct total porosity and water saturation,
correcting the classic understatement of Sw by legacy crushed-rock analysis.

Implements:

  - Fluid-summation porosity  Phitot = Phiair + BVO + BVW           (Eq. 1)
  - Bulk/grain porosity  Phitot = (V_bulk - V_grain)/V_bulk         (Eq. 2)
  - Conventional crushed-rock water saturation  S_wCR               (Eq. 3)
  - GRI+ water saturation  S_wGRI+                                  (Eq. 4)

Note: this issue's PDF text layer kept the equation numbers and variable
definitions but dropped the typeset glyphs, so these are faithful standard-form
reconstructions.  Paper anchors: crushing loses ~14% of total fluid
(beta_crush ~ 0.86), closed-retort efficiency ~98% (alpha_ret = 0.98), and the
GRI+ water saturation is ~30% higher than the conventional value.
"""

import numpy as np


# ---------------------------------------------- porosity ----------------

def fluid_summation_porosity(phi_air, bvo, bvw):
    """Fluid-summation total porosity  Phitot = Phiair + BVO + BVW  (Eq. 1)."""
    return phi_air + bvo + bvw


def bulk_grain_porosity(v_bulk, v_grain):
    """Bulk/grain total porosity  Phitot = (V_bulk - V_grain)/V_bulk  (Eq. 2)."""
    return (v_bulk - v_grain) / v_bulk


# ---------------------------------------------- saturation --------------

def sw_conventional(phi_air, bvo, bvw, alpha_ret=0.90):
    """Conventional crushed-rock water saturation  (Eq. 3).

        S_wCR = (BVW/alpha_ret) / (Phiair + (BVW + BVO)/alpha_ret)
    alpha_ret is the (open) retort fluid-collection efficiency.
    """
    bvw_c = bvw / alpha_ret
    return bvw_c / (phi_air + (bvw + bvo) / alpha_ret)


def sw_gri_plus(bvw, v_bulk, v_grain, alpha_ret=0.98, beta_crush=0.86):
    """GRI+ water saturation  (Eq. 4).

        S_wGRI+ = [BVW/(alpha_ret*beta_crush)] / [(V_bulk - V_grain)/V_bulk]
    beta_crush corrects for fluid lost during crushing (NMR-derived); alpha_ret
    is the closed-retort efficiency.
    """
    bvw_true = bvw / (alpha_ret * beta_crush)
    return bvw_true / bulk_grain_porosity(v_bulk, v_grain)


def crushing_loss_factor(nmr_intact, nmr_crushed):
    """beta_crush = (fluid retained after crushing)/(intact fluid) from NMR."""
    return nmr_crushed / nmr_intact


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Crushed-Rock GRI+ Workflow")
    print("=" * 60)

    # The two porosity methods agree for consistent inputs
    phi_air, bvo, bvw = 0.02, 0.03, 0.03
    phi1 = fluid_summation_porosity(phi_air, bvo, bvw)
    # bulk/grain consistent with phi1 = 0.08
    v_bulk, v_grain = 1.0, 0.92
    phi2 = bulk_grain_porosity(v_bulk, v_grain)
    print(f"  Phitot summation/bulk  = {phi1:.3f} / {phi2:.3f}")
    assert abs(phi1 - phi2) < 1e-9

    # Crushing-loss factor from NMR (~14% loss -> beta ~ 0.86)
    beta = crushing_loss_factor(1.0, 0.86)
    assert abs(beta - 0.86) < 1e-9

    # GRI+ water saturation is substantially higher than the conventional value
    sw_cr = sw_conventional(phi_air, bvo, bvw, alpha_ret=0.90)
    sw_plus = sw_gri_plus(bvw, v_bulk, v_grain, alpha_ret=0.98, beta_crush=0.86)
    uplift = (sw_plus - sw_cr) / sw_cr * 100.0
    print(f"  Sw conventional / GRI+ = {sw_cr:.3f} / {sw_plus:.3f}  (+{uplift:.0f}%)")
    assert sw_plus > sw_cr
    assert 10.0 < uplift < 60.0      # legacy understates Sw (paper reports ~30%)

    # More crushing loss (lower beta) raises the corrected Sw further
    assert sw_gri_plus(bvw, v_bulk, v_grain, beta_crush=0.75) > sw_plus
    print("  PASS")
    return {"phitot": phi1, "sw_cr": float(sw_cr), "sw_plus": float(sw_plus),
            "uplift_pct": float(uplift)}


if __name__ == "__main__":
    test_all()
