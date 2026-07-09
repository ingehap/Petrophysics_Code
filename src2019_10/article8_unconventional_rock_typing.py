"""
Article 8: Presenting a Multifaceted Approach to Unconventional Rock Typing and
           Technical Validation - Case Study in the Permian Basin and Impacts on
           Reservoir Characterization Workflows
Perry, Hayes (2019)
DOI: 10.30632/PJV60N5-2019a8

A multifaceted rock-typing workflow integrates core analysis and triple-combo
logs: a cluster-based electrofacies model is cross-validated against discrete
hydraulic flow units (HFUs) from the Amaefule flow-zone-indicator method.  This
module implements the standard HFU / Winland relations the workflow rests on
(the article body in the available extract truncated mid-references).

Implements:

  - Reservoir quality index  RQI = 0.0314*sqrt(k/phi)
  - Normalized porosity  phi_z = phi/(1 - phi)
  - Flow zone indicator  FZI = RQI/phi_z
  - Winland R35  log R35 = 0.732 + 0.588*log k - 0.864*log phi
  - HFU assignment by clustering log(FZI)

Note: this issue's source-PDF extract truncated mid-Article 8 (references), so
this module implements the standard Amaefule (1993) HFU / Winland (1972) forms
the paper cites.  Paper anchors: >= 7 discrete HFUs cross-validating an 8- to
14-cluster petrofacies model.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- HFU / Winland ----------

def rqi(k_md, phi):
    """Reservoir quality index  RQI = 0.0314*sqrt(k/phi)  (um), k in mD."""
    return petrolib.flow_transport.rqi(k_md, phi)


def normalized_porosity(phi):
    """Normalized porosity (pore/grain ratio)  phi_z = phi/(1 - phi)."""
    return petrolib.flow_transport.phi_z(phi)


def fzi(k_md, phi):
    """Flow zone indicator  FZI = RQI/phi_z  (um)."""
    return petrolib.flow_transport.fzi(k_md, phi)


def winland_r35(k_md, phi):
    """Winland pore-throat radius at 35% mercury  log R35 = 0.732 + 0.588 log k - 0.864 log phi.

    phi as a percentage; returns R35 in microns.
    """
    return petrolib.flow_transport.winland_r35(k_md, phi)


def assign_hfu(k_md, phi, n_units, seed=0):
    """Assign hydraulic flow units by clustering log10(FZI) into n_units bins."""
    logfzi = np.log10(fzi(k_md, phi))
    # 1-D k-means on log(FZI)
    rng = np.random.default_rng(seed)
    centers = np.linspace(logfzi.min(), logfzi.max(), n_units)
    labels = np.zeros(len(logfzi), int)
    for _ in range(100):
        new = np.argmin(np.abs(logfzi[:, None] - centers[None, :]), axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(n_units):
            if np.any(labels == c):
                centers[c] = logfzi[labels == c].mean()
    return labels, centers


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Unconventional Rock Typing (Permian)")
    print("=" * 60)

    # FZI is constant within a hydraulic flow unit (same k/phi^3 trend)
    phi = np.array([0.08, 0.12, 0.16, 0.20])
    # build k so FZI is identical: k = (FZI*phi_z/0.0314)^2 * phi
    fzi_target = 1.5
    phiz = normalized_porosity(phi)
    k = (fzi_target * phiz / 0.0314) ** 2 * phi
    fz = fzi(k, phi)
    print(f"  FZI within one HFU     = {np.array2string(fz, precision=3)}")
    assert np.allclose(fz, fzi_target, rtol=1e-6)

    # RQI rises with permeability; Winland R35 too
    assert rqi(10.0, 0.15) > rqi(1.0, 0.15)
    r35 = winland_r35(10.0, 0.15)
    print(f"  Winland R35 (10mD,15%) = {r35:.2f} um")
    assert r35 > 0 and winland_r35(100.0, 0.15) > r35

    # HFU clustering separates three planted flow units
    rng = np.random.default_rng(8)
    units = []
    for fz_u in (0.5, 1.5, 4.0):
        p = rng.uniform(0.08, 0.22, 30)
        kk = (fz_u * normalized_porosity(p) / 0.0314) ** 2 * p
        kk *= np.exp(0.05 * rng.standard_normal(30))     # mild scatter
        units.append((kk, p, fz_u))
    k_all = np.concatenate([u[0] for u in units])
    phi_all = np.concatenate([u[1] for u in units])
    labels, centers = assign_hfu(k_all, phi_all, n_units=3)
    print(f"  HFU log(FZI) centers   = {np.array2string(np.sort(centers), precision=2)}")
    # three well-separated FZI centers recovered
    assert len(np.unique(labels)) == 3
    assert np.ptp(np.sort(centers)) > 0.5
    print("  PASS")
    return {"R35": float(r35), "fzi_centers": np.sort(centers).tolist()}


if __name__ == "__main__":
    test_all()
