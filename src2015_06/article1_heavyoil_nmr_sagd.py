"""
Article 1: Improvement in Heavy-Oil Reservoir Evaluation Using Nuclear Magnetic
           Resonance: Long Lake and Kinosis SAGD Projects, Alberta, Canada
Cheng, Kotov, Pyke, Hanif (2015)
Reference: Petrophysics Vol. 56, No. 3 (June 2015), pp. 239-250
DOI: none assigned (this issue predates SPWLA DOI assignment)

In the McMurray bitumen sands (SAGD), conventional resistivity-based Sw is
unreliable (variable, unknown Rw).  NMR is salinity-insensitive: bitumen relaxes
mainly by bulk relaxation at very short T2 (< 1 ms in clean sand; the method
counts signal up to a 4-ms cutoff to cover viscosity shifts and shale), so only
a small fraction of the bitumen is visible to NMR.  Comparing the NMR porosity
(water + visible bitumen) with the density total porosity (all water + bitumen)
yields the bitumen content, after a clay-bound-water correction.

Implements:

  - Density total porosity (matrix 2.65, fluid 1.0 g/cm^3)
  - Gamma-ray index and Clavier shale volume
  - NMR-visible bitumen volume below the T2 cutoff (clay-bound-water corrected)
  - Bitumen bulk volume from the density-vs-NMR porosity difference
  - Bitumen weight fraction from the bulk volume and densities

Note: this issue's PDF has a text layer; the porosity/shale/bitumen relations
(Eqs. 1-8) are transcribed from the body (variable definitions survived), while
the typeset glyphs were dropped and reconstructed in standard form.  Densities
in g/cm^3, porosity/volumes/saturation as fractions, T2 in ms.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

T2_CUTOFF_BITUMEN_MS = 4.0    # upper limit of shale/bitumen signal
RHO_BITUMEN = 1.0123          # g/cm^3, Long Lake produced-bitumen average


# ---------------------------------------------- porosity / shale --------------

def density_porosity(rho_b, rho_ma=2.65, rho_fl=1.0):
    """Density total porosity  phi = (rho_ma - rho_b)/(rho_ma - rho_fl)."""
    return petrolib.porosity_lithology.density_porosity(rho_b, rho_ma, rho_fl)


def gamma_ray_index(gr, gr_clean, gr_shale):
    """Gamma-ray index  IGR = (GR - GR_clean)/(GR_shale - GR_clean)."""
    return petrolib.porosity_lithology.gamma_ray_index(gr, gr_clean, gr_shale, clip=None)


def vsh_clavier(igr):
    """Clavier shale volume from the gamma-ray index

        Vsh = 1.7 - sqrt(3.38 - (IGR + 0.7)^2),

    clipped to [0, 1].
    """
    return petrolib.porosity_lithology.vshale_from_gr(
        igr, 0.0, 1.0, method="clavier", clip=(0.0, 1.0))


# ---------------------------------------------- bitumen --------------

def clay_bound_water(vsh, phi_shale):
    """Clay-bound water volume  CBW = Vsh*phi_shale."""
    return vsh * phi_shale


def nmr_visible_bitumen(phi_nmr_below_cutoff, cbw):
    """NMR-visible bitumen volume = (porosity below the T2 cutoff) - clay-bound water."""
    return np.maximum(phi_nmr_below_cutoff - cbw, 0.0)


def bitumen_bulk_volume(phi_density, phi_nmr_water):
    """Total bitumen bulk volume from the density-vs-NMR porosity difference

        BVO_bit = phi_density_total - phi_NMR_water,

    where the NMR water porosity excludes the (mostly invisible) bitumen, so the
    difference is the total bitumen bulk volume.
    """
    return np.maximum(phi_density - phi_nmr_water, 0.0)


def bitumen_weight_fraction(bvo_bit, rho_b, rho_bit=RHO_BITUMEN):
    """Bitumen bulk-mass fraction  Weight_bit = BVO_bit*rho_bit/rho_b  (wt/wt)."""
    return bvo_bit * rho_bit / rho_b


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Heavy-Oil NMR (SAGD)")
    print("=" * 60)

    # Density porosity against quartz matrix / unit fluid density
    phi = density_porosity(2.18)
    print(f"  density porosity       = {phi:.3f}")
    assert np.isclose(phi, (2.65 - 2.18) / (2.65 - 1.0)) and density_porosity(2.65) == 0.0

    # Clavier Vsh: 0 at clean sand, increases with the gamma-ray index
    assert np.isclose(vsh_clavier(gamma_ray_index(30.0, 30.0, 150.0)), 0.0, atol=1e-6)
    vsh = vsh_clavier(gamma_ray_index(90.0, 30.0, 150.0))
    print(f"  Clavier Vsh (mid GR)   = {vsh:.3f}")
    assert 0 < vsh < 1

    # NMR-visible bitumen is the sub-cutoff signal minus clay-bound water
    cbw = clay_bound_water(vsh, phi_shale=0.35)
    vis = nmr_visible_bitumen(0.08, cbw)
    assert vis >= 0

    # Bitumen bulk volume from density-vs-NMR porosity difference
    phi_nmr_water = 0.22
    bvo = bitumen_bulk_volume(phi, phi_nmr_water) if phi > phi_nmr_water else bitumen_bulk_volume(0.30, 0.22)
    bvo = bitumen_bulk_volume(0.30, 0.22)
    wbit = bitumen_weight_fraction(bvo, rho_b=2.0)
    print(f"  bitumen BVO / weight   = {bvo:.3f} / {wbit:.3f} wt/wt")
    assert np.isclose(bvo, 0.08) and 0 < wbit < 1
    print("  PASS")
    return {"phi": float(phi), "Vsh": float(vsh), "BVO_bit": float(bvo), "Wbit": float(wbit)}


if __name__ == "__main__":
    test_all()
