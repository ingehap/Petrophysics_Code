"""
Article 1 (Tutorial): Organic-Mudstone Petrophysics: Workflow to Estimate
                      Storage Capacity (Part 1)
Newsham, Comisky, Chemali (2019)
DOI: 10.30632/PJV60N1Y2019t1

Part 1 of the organic-mudstone tutorial series introduces the storage-capacity
workflow and ties the physical (storage) results to production potential through
fractional flow and water cut.  Storage is partitioned into bound and movable
fluids; producibility is assessed from the relative-permeability fractional-flow
relation.

Implements:

  - Bulk volume water / hydrocarbon and movable-fluid index
  - Free gas G_free = phi*(1 - Sw)/Bg and Langmuir adsorbed gas
  - Buckley-Leverett fractional flow  fw = 1/(1 + (kro*muw)/(krw*muo))
  - Water cut from the fractional flow at reservoir conditions

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the storage-capacity / producibility relations the tutorial introduces.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- storage -----------------

def bulk_volume_water(phi, sw):
    """Bulk volume water  BVW = phi*Sw."""
    return petrolib.saturation_resistivity.bulk_volume_water(phi, sw)


def movable_fluid_index(phi, sw_irr):
    """Movable-fluid bulk volume  MFI = phi*(1 - Sw_irr)."""
    return phi * (1.0 - sw_irr)


def free_gas(phi, sw, Bg):
    """Free gas content  G_free = phi*(1 - Sw)/Bg."""
    return phi * (1.0 - np.asarray(sw, float)) / Bg


def langmuir(rho_b, VL, PL, P):
    """Langmuir adsorbed-gas capacity  Gc = rho_b*VL*P/(PL + P)."""
    P = np.asarray(P, float)
    return rho_b * VL * P / (PL + P)


# ---------------------------------------------- producibility -----------

def fractional_flow(krw, kro, mu_w, mu_o):
    """Buckley-Leverett water fractional flow  fw = 1/(1 + (kro*muw)/(krw*muo))."""
    return 1.0 / (1.0 + (kro * mu_w) / (krw * mu_o))


def water_cut(krw, kro, mu_w, mu_o):
    """Surface water cut (= fractional flow for incompressible flow), %."""
    return 100.0 * fractional_flow(krw, kro, mu_w, mu_o)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1 (Tutorial): Organic-Mudstone Storage, Part 1")
    print("=" * 60)

    # Storage volumetrics
    assert abs(bulk_volume_water(0.10, 0.4) - 0.04) < 1e-12
    assert movable_fluid_index(0.10, 0.3) > 0
    assert free_gas(0.08, 0.3, 0.005) > free_gas(0.04, 0.3, 0.005)
    rho_b, VL, PL = 2.45, 0.006, 1800.0
    assert abs(langmuir(rho_b, VL, PL, PL) - rho_b * VL / 2.0) < 1e-9

    # Fractional flow: rises with water relative permeability
    fw_lo = fractional_flow(0.05, 0.8, 0.5, 5.0)     # mostly oil flowing
    fw_hi = fractional_flow(0.5, 0.1, 0.5, 5.0)      # mostly water flowing
    print(f"  fractional flow lo/hi  = {fw_lo:.2f} / {fw_hi:.2f}")
    assert fw_hi > fw_lo and 0 <= fw_lo <= 1 and 0 <= fw_hi <= 1

    # Water cut tracks fractional flow (0-100%)
    wc = water_cut(0.3, 0.3, 0.5, 5.0)
    print(f"  water cut (equal kr)   = {wc:.0f} %")
    assert abs(wc - 100.0 / (1.0 + 0.5 / 5.0)) < 1e-9
    # more viscous oil flows less readily, so the water cut is higher
    assert water_cut(0.3, 0.3, 0.5, 20.0) > water_cut(0.3, 0.3, 0.5, 1.0)
    print("  PASS")
    return {"fw_hi": float(fw_hi), "water_cut": float(wc)}


if __name__ == "__main__":
    test_all()
