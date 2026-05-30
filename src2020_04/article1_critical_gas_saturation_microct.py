"""
Article 1: Determination of Critical Gas Saturation by Micro-CT
Berg, Gao, Georgiadis, Brussee, Coorn, van der Linde, Dietderich, Alpak,
Eriksen, Mooijer-van den Heuvel, Southwick, Appel, Wilson (2020)
DOI: 10.30632/PJV61N2-2020a1

Solution-gas drive below the bubblepoint is imaged with X-ray micro-CT in a
Bentheimer minicore: gas bubbles are segmented in 3D and the critical gas
saturation Sgc is identified as the percolation threshold - the saturation at
which the segmented gas phase first forms a sample-spanning connected cluster.
Relative permeability then follows from single-phase flow on the connected
sub-network of each phase.

Implements:

  - 3D gas-phase segmentation as a random field thresholded by gas fraction
  - Percolation detection (spanning connected cluster) by labeling
  - Critical gas saturation Sgc = percolation-threshold saturation
  - Connected-phase fraction and a connectivity-based kr proxy

Note: this issue's PDF has a text layer; the paper is an experimental /
image-analysis study with no numbered equations, so this module implements the
percolation method it relies on.  Paper anchors: Bentheimer porosity 19.1%,
permeability 2500 mD, Sgc = 0.20-0.25.
"""

import numpy as np
from scipy import ndimage


# ---------------------------------------------- percolation -------------

def spans(binary_grid, axis=0):
    """True if the True-phase in binary_grid forms a cluster spanning `axis`."""
    labels, n = ndimage.label(binary_grid)
    if n == 0:
        return False
    lo = set(np.unique(labels.take(0, axis=axis))) - {0}
    hi = set(np.unique(labels.take(-1, axis=axis))) - {0}
    return len(lo & hi) > 0


def connected_fraction(binary_grid, axis=0):
    """Fraction of the True-phase voxels that belong to a spanning cluster."""
    labels, n = ndimage.label(binary_grid)
    if n == 0:
        return 0.0
    lo = set(np.unique(labels.take(0, axis=axis))) - {0}
    hi = set(np.unique(labels.take(-1, axis=axis))) - {0}
    spanning = lo & hi
    if not spanning:
        return 0.0
    mask = np.isin(labels, list(spanning))
    return float(mask.sum() / binary_grid.sum())


def critical_gas_saturation(field, fractions):
    """Lowest gas saturation at which the gas phase first percolates.

    `field` is a per-voxel random scalar; for each trial gas fraction the
    lowest-`field` voxels are designated gas, and Sgc is the first fraction that
    spans the sample.  Returns Sgc (or None if never percolating).
    """
    flat_sorted = np.sort(field, axis=None)
    for f in fractions:
        thresh = flat_sorted[int(f * field.size) - 1]
        gas = field <= thresh
        if spans(gas):
            return float(gas.mean())
    return None


def kr_connected(binary_grid, axis=0):
    """Connectivity-based relative-permeability proxy (connected-fraction)."""
    return connected_fraction(binary_grid, axis)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Critical Gas Saturation by Micro-CT")
    print("=" * 60)

    # Bentheimer-like properties recorded as context
    poro, perm = 0.191, 2500.0
    print(f"  Bentheimer poro / perm = {poro} / {perm:.0f} mD")
    assert 0.18 < poro < 0.20

    # A smooth random field: gas nucleates in the lowest-pressure voxels and
    # percolates once enough is connected -> Sgc in a physical range
    rng = np.random.default_rng(1)
    field = ndimage.gaussian_filter(rng.standard_normal((40, 40, 40)), sigma=1.5)
    fractions = np.arange(0.05, 0.60, 0.01)
    sgc = critical_gas_saturation(field, fractions)
    print(f"  critical gas saturation = {sgc:.3f}")
    assert sgc is not None and 0.10 < sgc < 0.45

    # Below Sgc the gas does not span; above it, it does
    flat = np.sort(field, axis=None)
    below = field <= flat[int((sgc - 0.05) * field.size) - 1]
    above = field <= flat[int((sgc + 0.10) * field.size) - 1]
    assert not spans(below)
    assert spans(above)

    # Connectivity-based kr proxy rises with gas saturation above Sgc
    kr_lo = kr_connected(field <= flat[int((sgc + 0.02) * field.size) - 1])
    kr_hi = kr_connected(field <= flat[int((sgc + 0.15) * field.size) - 1])
    print(f"  connected kr proxy lo/hi = {kr_lo:.3f} / {kr_hi:.3f}")
    assert 0.0 < kr_lo <= kr_hi <= 1.0
    print("  PASS")
    return {"Sgc": sgc, "poro": poro, "kr_hi": kr_hi}


if __name__ == "__main__":
    test_all()
