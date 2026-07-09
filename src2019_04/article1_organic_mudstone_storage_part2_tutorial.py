"""
Article 1 (Tutorial): Organic Mudstone Petrophysics, Part 2: Workflow to
                      Estimate Storage Capacity
Newsham, Comisky, Chemali (2019)
DOI: 10.30632/PJV60N2-2019t1

Part 2 of the organic-mudstone tutorial series gives a workflow to estimate
storage capacity by partitioning the total porosity into kerogen, clay-bound
water, capillary-bound water and free (movable) hydrocarbon, then converting the
hydrocarbon-filled pore volume into oil- or gas-in-place.

Implements:

  - Kerogen volume from TOC and the porosity partition
  - Clay-bound / capillary / free-fluid porosity components
  - Bulk volume hydrocarbon (BVH) and water saturation
  - Gas-in-place / oil-in-place from BVH and the formation volume factor

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the storage-capacity volumetrics the tutorial describes.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- porosity partition ------

def kerogen_volume(toc, rho_b, rho_k=1.30, carbon_frac=0.80):
    """Kerogen volume fraction from TOC  V_k = (TOC/carbon_frac)*rho_b/rho_k."""
    return petrolib.porosity_lithology.kerogen_volume_from_toc(
        toc, rho_b, rho_k=rho_k, carbon_frac=carbon_frac)


def partition_porosity(phi_total, cbw, sw_cap):
    """Partition total porosity into clay-bound, capillary and free components.

    Returns (clay_bound_water, capillary_water, free_hydrocarbon) pore volumes,
    where cbw is the clay-bound-water fraction of phi_total and sw_cap the
    capillary-water saturation of the remaining (effective) porosity.
    """
    phi_e = phi_total * (1.0 - cbw)
    cbw_vol = phi_total * cbw
    cap_vol = phi_e * sw_cap
    free_hc = phi_e * (1.0 - sw_cap)
    return cbw_vol, cap_vol, free_hc


def bulk_volume_hydrocarbon(phi_total, sw):
    """Bulk volume hydrocarbon  BVH = phi_total*(1 - Sw)."""
    return petrolib.porosity_lithology.hydrocarbon_pore_volume(phi_total, sw)


def water_saturation(cbw_vol, cap_vol, phi_total):
    """Total water saturation from bound + capillary water volumes."""
    return (cbw_vol + cap_vol) / phi_total


# ---------------------------------------------- in-place ----------------

def gas_in_place(bvh, Bg):
    """Free gas-in-place per unit bulk volume  GIP = BVH/Bg."""
    return bvh / Bg


def oil_in_place(bvh, Bo):
    """Oil-in-place per unit bulk volume  OIP = BVH/Bo."""
    return bvh / Bo


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1 (Tutorial): Organic Mudstone Storage, Part 2")
    print("=" * 60)

    # Kerogen volume rises with TOC
    assert kerogen_volume(0.08, 2.45) > kerogen_volume(0.03, 2.45)

    # Porosity partition components sum back to total porosity
    phi_t = 0.10
    cbw_vol, cap_vol, free_hc = partition_porosity(phi_t, cbw=0.25, sw_cap=0.4)
    print(f"  CBW / capillary / free = {cbw_vol:.4f} / {cap_vol:.4f} / {free_hc:.4f}")
    assert abs(cbw_vol + cap_vol + free_hc - phi_t) < 1e-9
    assert free_hc > 0

    # Water saturation from bound + capillary water
    sw = water_saturation(cbw_vol, cap_vol, phi_t)
    print(f"  water saturation       = {sw:.3f}")
    assert 0.0 < sw < 1.0

    # Bulk volume hydrocarbon consistent with the partition's free-HC volume
    bvh = bulk_volume_hydrocarbon(phi_t, sw)
    print(f"  BVH                    = {bvh:.4f}")
    assert abs(bvh - free_hc) < 1e-9

    # In-place: gas (small Bg) and oil (Bo ~ 1.2)
    gip = gas_in_place(bvh, Bg=0.005)
    oip = oil_in_place(bvh, Bo=1.2)
    print(f"  GIP / OIP              = {gip:.3f} / {oip:.4f}")
    assert gip > oip > 0                          # gas expands -> more standard volume
    print("  PASS")
    return {"free_hc": float(free_hc), "sw": float(sw), "bvh": float(bvh),
            "GIP": float(gip)}


if __name__ == "__main__":
    test_all()
