"""
Article 7: Digital and Conventional Techniques to Study Permeability
           Heterogeneity in Complex Carbonate Rocks
Dernaika, Al Mansoori, Singh, Al Dayyani, Kalam, Bhakta, Koronfol, Uddin (2018)
DOI: 10.30632/PJV59N3-2018a6  (inferred - see note)

Whole-core carbonate samples are imaged, segmented by flow phase, populated with
effective porosity/permeability, and simulated (Darcy/Brinkman) to give
directional permeability and Kv/Kh anisotropy.  This *methodology proxy*
implements the standard permeability-heterogeneity quantities the paper relies
on: directional (arithmetic vs. harmonic) averaging of layered permeability, the
Kv/Kh anisotropy ratio, and the Dykstra-Parsons and Lorenz heterogeneity
coefficients.

Implements:

  - Horizontal (arithmetic) and vertical (harmonic) layer permeability
  - Kv/Kh anisotropy ratio (<= 1 for layered media)
  - Dykstra-Parsons coefficient  V = (k50 - k84.1)/k50
  - Lorenz coefficient from the flow-capacity vs storage-capacity curve

Note: this article's body was beyond this issue's machine extraction (the source
text ended at journal p372), so - as with the other methodology proxies in this
repository - the relations below are the standard heterogeneity formulas the
described digital/conventional workflow uses, not formulas transcribed from the
paper.  The DOI suffix (a6) is inferred from the issue's confirmed pattern.
"""

import numpy as np


# ---------------------------------------------- directional averaging --------------

def horizontal_permeability(perms):
    """Horizontal (parallel-bedding) permeability = arithmetic mean of layers."""
    return float(np.mean(perms))


def vertical_permeability(perms):
    """Vertical (series, across-bedding) permeability = harmonic mean of layers."""
    k = np.asarray(perms, float)
    return float(len(k) / np.sum(1.0 / k))


def kv_kh_ratio(perms):
    """Kv/Kh anisotropy ratio (1 for homogeneous, < 1 for layered media)."""
    return vertical_permeability(perms) / horizontal_permeability(perms)


# ---------------------------------------------- heterogeneity --------------

def dykstra_parsons(perms):
    """Dykstra-Parsons coefficient  V = (k50 - k84.1)/k50.

    Read off the log-normal permeability distribution: k50 is the median and
    k84.1 is the value one standard deviation below it (84.1st percentile of the
    descending cumulative).  V = 0 homogeneous, -> 1 highly heterogeneous.
    """
    k = np.asarray(perms, float)
    k50 = np.percentile(k, 50)
    k841 = np.percentile(k, 100 - 84.1)                 # one sigma below the median
    return (k50 - k841) / k50


def lorenz_coefficient(perms, phis, thicknesses=None):
    """Lorenz coefficient from the flow-capacity vs storage-capacity curve.

    Layers are ordered by descending k/phi; the area between the cumulative
    flow-capacity (k*h) and storage-capacity (phi*h) curve and the 45-degree
    line, doubled, is the coefficient (0 homogeneous, -> 1 heterogeneous).
    """
    k = np.asarray(perms, float)
    phi = np.asarray(phis, float)
    h = np.ones_like(k) if thicknesses is None else np.asarray(thicknesses, float)
    order = np.argsort(-(k / phi))
    kh = (k * h)[order]
    ph = (phi * h)[order]
    fc = np.concatenate([[0], np.cumsum(kh) / kh.sum()])     # flow capacity
    sc = np.concatenate([[0], np.cumsum(ph) / ph.sum()])     # storage capacity
    area = np.trapz(fc, sc)                                   # area under the curve
    return 2.0 * (area - 0.5)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Carbonate Permeability Heterogeneity (proxy)")
    print("=" * 60)

    # Harmonic (vertical) <= arithmetic (horizontal); equal only if homogeneous
    layers = [500.0, 50.0, 5.0, 200.0, 20.0]            # mD
    kh, kv = horizontal_permeability(layers), vertical_permeability(layers)
    print(f"  Kh / Kv                = {kh:.1f} / {kv:.2f} mD,  Kv/Kh = {kv / kh:.3f}")
    assert kv <= kh and kv_kh_ratio(layers) < 1.0

    # A homogeneous stack: isotropic and zero heterogeneity
    homo = [100.0] * 5
    assert np.isclose(kv_kh_ratio(homo), 1.0)
    assert np.isclose(dykstra_parsons(homo), 0.0, atol=1e-9)
    assert np.isclose(lorenz_coefficient(homo, [0.2] * 5), 0.0, atol=1e-9)

    # The heterogeneous stack has positive DP and Lorenz coefficients
    dp = dykstra_parsons(layers)
    lc = lorenz_coefficient(layers, [0.28, 0.18, 0.08, 0.22, 0.12])
    print(f"  Dykstra-Parsons / Lorenz = {dp:.3f} / {lc:.3f}")
    assert dp > 0 and 0.0 < lc < 1.0
    print("  PASS")
    return {"Kv_Kh": float(kv_kh_ratio(layers)), "Dykstra_Parsons": float(dp), "Lorenz": float(lc)}


if __name__ == "__main__":
    test_all()
