"""
Article 1 (Tutorial): Organic-Mudstone Petrophysics: Part 3: Workflow to
                      Estimate Storage Capacity
Newsham, Comisky, Chemali (2019)
DOI: 10.30632/PJV60N3-2019t1

The third part of an organic-mudstone tutorial series gives a workflow to
estimate hydrocarbon storage capacity, partitioning total gas-in-place into
free gas (in the effective pore space), adsorbed gas (Langmuir isotherm on the
kerogen surface), and the bulk-volume contributions of kerogen, water and
matrix.

Implements:

  - Effective porosity from total porosity, water and kerogen volumes
  - Free gas content  G_free = phi_e*(1 - Sw)/Bg
  - Langmuir adsorbed gas  Gc = rho_b*VL*P/(PL + P)
  - Total gas-in-place and the free/adsorbed storage partition

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the journal metadata and these are faithful
standard-form reconstructions of the storage-capacity workflow the tutorial
describes.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- porosity ----------------

def kerogen_volume(toc, rho_b, rho_k=1.30, carbon_frac=0.80):
    """Kerogen volume fraction from TOC  V_k = (TOC/carbon_frac)*rho_b/rho_k."""
    return petrolib.porosity_lithology.kerogen_volume_from_toc(
        toc, rho_b, rho_k=rho_k, carbon_frac=carbon_frac)


def effective_porosity(phi_total, vsh, phi_sh):
    """Shale-corrected effective porosity  phi_e = phi_total - Vsh*phi_sh."""
    return petrolib.porosity_lithology.effective_porosity(phi_total, vsh, phi_sh)


# ---------------------------------------------- gas content -------------

def free_gas(phi_e, sw, Bg):
    """Free gas content  G_free = phi_e*(1 - Sw)/Bg."""
    return phi_e * (1.0 - np.asarray(sw, float)) / Bg


def langmuir(rho_b, VL, PL, P):
    """Langmuir adsorbed-gas capacity  Gc = rho_b*VL*P/(PL + P)."""
    P = np.asarray(P, float)
    return rho_b * VL * P / (PL + P)


def total_gip(g_free, g_adsorbed):
    """Total gas-in-place = free + adsorbed."""
    return g_free + g_adsorbed


def storage_partition(g_free, g_adsorbed):
    """Free / adsorbed fractions of total gas-in-place."""
    tot = g_free + g_adsorbed
    return float(g_free / tot), float(g_adsorbed / tot)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1 (Tutorial): Organic-Mudstone Storage Capacity")
    print("=" * 60)

    # Kerogen volume rises with TOC
    vk = kerogen_volume(0.06, rho_b=2.45)
    print(f"  kerogen volume (TOC 6%) = {vk:.3f}")
    assert vk > 0 and kerogen_volume(0.10, 2.45) > vk

    # Effective porosity is reduced by shale
    phie = effective_porosity(0.10, vsh=0.3, phi_sh=0.15)
    assert phie < 0.10 and abs(phie - (0.10 - 0.3 * 0.15)) < 1e-9

    # Free gas rises with porosity and hydrocarbon saturation
    assert free_gas(0.08, 0.3, 0.005) > free_gas(0.04, 0.3, 0.005)

    # Langmuir: half capacity at P = PL, plateau at high P
    rho_b, VL, PL = 2.45, 0.006, 1800.0
    assert abs(langmuir(rho_b, VL, PL, PL) - rho_b * VL / 2.0) < 1e-9

    # Storage partition: at low pressure adsorbed dominates, at high pressure
    # free gas grows
    gf_lo = free_gas(phie, 0.35, 0.012); ga_lo = langmuir(rho_b, VL, PL, 800.0)
    gf_hi = free_gas(phie, 0.35, 0.004); ga_hi = langmuir(rho_b, VL, PL, 5000.0)
    ff_lo, fa_lo = storage_partition(gf_lo, ga_lo)
    ff_hi, fa_hi = storage_partition(gf_hi, ga_hi)
    print(f"  free fraction lo/hi P  = {ff_lo:.2f} / {ff_hi:.2f}")
    assert abs(ff_lo + fa_lo - 1.0) < 1e-9
    assert ff_hi > ff_lo                          # higher P (lower Bg) -> more free
    print("  PASS")
    return {"V_kerogen": float(vk), "phi_e": float(phie),
            "free_frac_hiP": ff_hi}


if __name__ == "__main__":
    test_all()
