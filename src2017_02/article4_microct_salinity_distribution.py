"""
Article 4: Fast X-Ray Micro-CT Study of the Impact of Brine Salinity on the
           Pore-Scale Fluid Distribution During Waterflooding
Bartels, Rucker, Berg, Mahani, Georgiadis, Fadili, Brussee, Coorn, van der Linde,
Hinz, Jacob, Wagner, Henkel, Enzmann, Bonnin, Stampanoni, Ott, Blunt,
Hassanizadeh (2017)
Reference: Petrophysics Vol. 58, No. 1 (February 2017), pp. 36-47
DOI: none assigned (this issue predates SPWLA DOI assignment)

Fast synchrotron micro-CT images the pore-scale fluid distribution during
tertiary low-salinity flooding.  This module implements the image-analysis
bookkeeping the study relies on: phase saturation from segmented voxel counts, a
granulometry pore-size distribution, the oil fraction resolved by pore size, and
the wettability-alteration signature (oil shifting from small to large pores as
the rock becomes more water-wet).

Implements:

  - Phase saturation from segmented voxels  S = phase_voxels/pore_voxels
  - Granulometry pore-size distribution (normalized)
  - Oil volume fraction resolved by pore size
  - Mean oil-occupied pore size (wettability-shift signature)

Note: this is an imaging study (no closed-form physics equation), so the
relations below are the standard segmentation/saturation bookkeeping it uses.
Saturations and fractions dimensionless; pore sizes in micrometres.
"""

import numpy as np


# ---------------------------------------------- saturation / PSD --------------

def phase_saturation(phase_voxels, pore_voxels):
    """Phase saturation from segmented voxel counts  S = phase_voxels/pore_voxels."""
    return np.asarray(phase_voxels, float) / pore_voxels


def granulometry_psd(counts):
    """Normalized pore-size distribution from sphere-fitting (granulometry) counts."""
    c = np.asarray(counts, float)
    return c / c.sum()


def oil_fraction_by_pore_size(oil_volume_per_bin):
    """Oil volume fraction resolved by pore-size bin (normalized to total oil)."""
    o = np.asarray(oil_volume_per_bin, float)
    return o / o.sum()


def mean_oil_pore_size(pore_sizes, oil_volume_per_bin):
    """Oil-volume-weighted mean pore size (rises as oil moves to larger pores)."""
    s = np.asarray(pore_sizes, float)
    w = np.asarray(oil_volume_per_bin, float)
    return float(np.sum(s * w) / np.sum(w))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Micro-CT Salinity Fluid Distribution")
    print("=" * 60)

    # Saturation from voxel counts (Ketton oil ~34% at end of high-salinity)
    so = phase_saturation(340, 1000)
    print(f"  oil saturation         = {so:.2f}")
    assert np.isclose(so, 0.34)

    # Granulometry PSD and oil-by-pore-size both normalize to 1
    psd = granulometry_psd([10, 40, 30, 20.0])
    assert np.isclose(psd.sum(), 1.0)
    assert np.isclose(oil_fraction_by_pore_size([5, 3, 2.0]).sum(), 1.0)

    # Wettability shift: oil moves from small to large pores -> mean size rises
    pore_sizes = np.array([5.0, 15.0, 30.0, 60.0])      # micrometres
    oil_before = np.array([6.0, 3.0, 1.0, 0.0])         # oil in small pores
    oil_after = np.array([1.0, 2.0, 3.0, 4.0])          # oil in large pores
    before = mean_oil_pore_size(pore_sizes, oil_before)
    after = mean_oil_pore_size(pore_sizes, oil_after)
    print(f"  mean oil pore size b/a = {before:.1f} / {after:.1f} um")
    assert after > before
    print("  PASS")
    return {"So": float(so), "mean_pore_shift": float(after - before)}


if __name__ == "__main__":
    test_all()
