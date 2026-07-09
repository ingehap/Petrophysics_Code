"""
Article 1 (Tutorial): What is it about Shaly Sands? Shaly Sand Tutorial No. 2
                      of 3
Thomas (2018)
DOI: 10.30632/PJV59N2-2018t1

The second shaly-sand tutorial explains why clay perturbs the porosity logs.
The thermal-neutron tool responds to hydrogen index, so the hydroxyl hydrogen
bound in clay is counted as if it were pore water and the neutron porosity reads
far too high in shales; the overstatement scales with the product of clay volume
and clay hydrogen index, and 1:1 clays (kaolinite, chlorite) contribute about
twice the hydrogen index of 2:1 clays (illite, smectite).  The gamma ray tracks
adsorbed Th/U and intrinsic K, so it follows exchange capacity, not clay volume
- hence Vsh is not Vclay.

Implements:

  - Neutron porosity overstatement in shale  phi_N = phi_w + Vclay*HI_clay
  - Clay hydrogen-index lookup (1:1 vs 2:1 clays)
  - Spectral-source gamma ray  GR = 4*Th + 8*U + 16*K
  - Vsh-to-Vclay caution (GR-based Vsh overstates clay volume)

Note: this installment is conceptual (no closed-form saturation equations), so
the relations below capture the quantitative ideas the tutorial teaches; the
neutron-overstatement and spectral-GR forms are standard reconstructions.
Hydrogen index and porosity are fractional.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# Representative clay hydrogen indices; 1:1 clays ~ 2x the 2:1 clays
CLAY_HI = {"kaolinite": 0.55, "chlorite": 0.50, "illite": 0.45, "smectite": 0.45}


# ---------------------------------------------- neutron --------------

def neutron_porosity_shale(phi_water, vclay, hi_clay):
    """Apparent neutron porosity in shale  phi_N = phi_w + Vclay*HI_clay.

    The clay-bound hydroxyl hydrogen is counted as pore water, so the tool
    overstates porosity by Vclay*HI_clay above the true water-filled porosity.
    """
    return phi_water + np.asarray(vclay, float) * hi_clay


def clay_hydrogen_index(clay_type):
    """Hydrogen index of a clay; 1:1 clays carry ~2x the HI of 2:1 clays."""
    return CLAY_HI[clay_type]


# ---------------------------------------------- gamma ray --------------

def gr_api(th_ppm, u_ppm, k_wt):
    """Spectral-source gamma ray  GR = 4*Th + 8*U + 16*K  (API).

    Th and U are surface-adsorbed onto high-specific-surface clays and K is
    intrinsic to illite, so GR follows exchange capacity rather than clay volume.
    """
    return petrolib.nuclear.gr_api(k_wt, u_ppm, th_ppm, coeff=(16.0, 8.0, 4.0))


def vsh_from_gr(gr, gr_clean, gr_shale):
    """Linear shale volume from gamma ray  Vsh = (GR - GRclean)/(GRshale - GRclean).

    This Vsh is a *shale* indicator and overstates the true clay volume (Vsh !=
    Vclay), per the tutorial's caution.
    """
    return petrolib.porosity_lithology.gamma_ray_index(gr, gr_clean, gr_shale)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1 (Tutorial): Shaly Sand No. 2 of 3")
    print("=" * 60)

    # A 10-p.u. water-filled, 55%-clay shale reads 30-40 p.u. on neutron
    phi_n = neutron_porosity_shale(0.10, 0.55, clay_hydrogen_index("illite"))
    print(f"  apparent neutron phi   = {phi_n:.3f}  (true water 0.10)")
    assert 0.30 <= phi_n <= 0.40 and phi_n > 0.10

    # 1:1 clay (kaolinite) overstates more than 2:1 clay (illite)
    assert clay_hydrogen_index("kaolinite") > clay_hydrogen_index("illite")

    # Spectral GR and a (deliberately overstated) Vsh from it
    gr = gr_api(th_ppm=12.0, u_ppm=4.0, k_wt=3.0)
    print(f"  spectral GR            = {gr:.0f} API")
    assert np.isclose(gr, 4 * 12 + 8 * 4 + 16 * 3)
    vsh = vsh_from_gr(gr, gr_clean=20.0, gr_shale=140.0)
    assert 0.0 <= vsh <= 1.0
    print("  PASS")
    return {"phi_neutron": float(phi_n), "Vsh": float(vsh)}


if __name__ == "__main__":
    test_all()
