"""
Article 4: Uncertainty Quantification in Image Segmentation for Image-Based Rock
           Physics in a Shaly Sandstone
Howard, Lin, Zhang (2019)
DOI: 10.30632/PJV60N2-2019a2

Image-based rock physics starts by segmenting a grayscale 3D image into pore and
grain (and, in a shaly sandstone, an intermediate microporous clay phase).  The
chosen segmentation threshold is uncertain, and that uncertainty propagates into
the computed porosity (and downstream permeability / elastic properties).  This
module quantifies the porosity uncertainty from the threshold uncertainty.

Implements:

  - Porosity from a grayscale threshold (pore = dark)
  - Otsu-style threshold from the gray-level histogram
  - Porosity sensitivity dPhi/dThreshold and propagated uncertainty
  - Three-phase (pore / clay / grain) segmentation fractions

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the segmentation / uncertainty-propagation method the paper applies.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- segmentation -----------

def porosity_from_threshold(image, threshold):
    """Porosity = fraction of voxels below the threshold (pore = dark)."""
    return petrolib.borehole_image.porosity_from_mask(np.asarray(image, float) < threshold)


def otsu_threshold(image, bins=256):
    """Otsu threshold maximizing between-class variance of the histogram."""
    return petrolib.borehole_image.otsu_threshold(image, bins=bins)


def porosity_uncertainty(image, threshold, threshold_std):
    """Propagate threshold uncertainty into porosity uncertainty.

    sigma_phi ~ |dPhi/dThreshold| * sigma_threshold, with the derivative
    estimated by central differences.
    """
    h = max(threshold_std, 1e-6)
    dphi = (porosity_from_threshold(image, threshold + h)
            - porosity_from_threshold(image, threshold - h)) / (2.0 * h)
    return abs(dphi) * threshold_std


def three_phase_fractions(image, t_pore, t_clay):
    """Pore (<t_pore), clay (t_pore..t_clay), grain (>=t_clay) fractions."""
    pore, clay, grain = petrolib.borehole_image.class_fractions(image, [t_pore, t_clay])
    return float(pore), float(clay), float(grain)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Image-Segmentation Uncertainty (Shaly Sandstone)")
    print("=" * 60)

    # Synthetic bimodal image: 25% pore (dark ~0.2) + 75% grain (bright ~0.8)
    rng = np.random.default_rng(2)
    n = 40000
    true_poro = 0.25
    npore = int(true_poro * n)
    img = np.concatenate([rng.normal(0.20, 0.05, npore),
                          rng.normal(0.80, 0.05, n - npore)])
    img = np.clip(img, 0, 1)

    # Otsu threshold lands between the two modes (0.2 and 0.8), biased toward the
    # lower-weight pore class; the recovered porosity matches the planted value
    thr = otsu_threshold(img)
    poro = porosity_from_threshold(img, thr)
    print(f"  Otsu threshold / porosity = {thr:.3f} / {poro:.3f}")
    assert 0.25 < thr < 0.75 and abs(poro - true_poro) < 0.02

    # Porosity uncertainty grows with threshold uncertainty
    u_small = porosity_uncertainty(img, thr, 0.01)
    u_large = porosity_uncertainty(img, thr, 0.05)
    print(f"  sigma_phi (thr std 0.01/0.05) = {u_small:.4f} / {u_large:.4f}")
    assert u_large > u_small >= 0.0
    # at the histogram valley the porosity is least sensitive to the threshold
    u_valley = porosity_uncertainty(img, 0.5, 0.03)
    u_slope = porosity_uncertainty(img, 0.30, 0.03)   # on the pore-mode flank
    assert u_valley < u_slope

    # Three-phase fractions sum to 1
    pore, clay, grain = three_phase_fractions(img, 0.35, 0.65)
    print(f"  pore/clay/grain        = {pore:.2f} / {clay:.2f} / {grain:.2f}")
    assert abs(pore + clay + grain - 1.0) < 1e-9
    print("  PASS")
    return {"otsu": thr, "porosity": poro, "sigma_phi_0.05": u_large}


if __name__ == "__main__":
    test_all()
