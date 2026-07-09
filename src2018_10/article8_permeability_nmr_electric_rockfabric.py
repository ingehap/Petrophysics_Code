"""
Article 8: Integrated Workflow to Estimate Permeability Through Quantification of
           Rock Fabric Using Joint Interpretation of Nuclear Magnetic Resonance
           and Electric Measurements
Garcia, Han, Heidari (2018)
DOI: 10.30632/PJV59N5-2018a7

Permeability is estimated by quantifying rock fabric from a joint interpretation
of NMR (pore-size / T2) and electrical (formation factor / cementation exponent)
measurements: NMR gives a Timur-Coates permeability while the electrical
cementation exponent constrains the pore connectivity, and the two are combined
into a rock-fabric-consistent permeability.

Implements:

  - Timur-Coates NMR permeability
  - Formation factor and cementation exponent m
  - Tortuosity from m (connectivity proxy)
  - Joint NMR + electrical permeability estimate

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard NMR / electrical rock-fabric permeability
relations the paper's title describes.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- NMR / electrical --------

def timur_coates(phi, ffi, bvi, C=10.0):
    """Timur-Coates permeability  k = (phi/C)^4*(FFI/BVI)^2  (mD)."""
    return (np.asarray(phi, float) / C) ** 4 * (ffi / bvi) ** 2 * 1e6


def formation_factor(phi, a=1.0, m=2.0):
    """Archie formation factor  F = a/phi^m."""
    return petrolib.saturation_resistivity.formation_factor(phi, a=a, m=m)


def cementation_exponent(F, phi, a=1.0):
    """Cementation exponent from F and phi  m = ln(F/a)/ln(1/phi)."""
    return petrolib.saturation_resistivity.cementation_exponent_at_point(phi, F, a=a)


def joint_permeability(phi, ffi, bvi, m, m_ref=2.0):
    """Joint NMR + electrical permeability: scale Timur-Coates by connectivity.

    A higher cementation exponent (poorer connectivity) reduces permeability
    relative to the NMR-only estimate.
    """
    k_nmr = timur_coates(phi, ffi, bvi)
    return k_nmr * (m_ref / m) ** 2


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Permeability From Rock Fabric (NMR + Electric)")
    print("=" * 60)

    # Timur-Coates: more free fluid -> higher permeability
    assert timur_coates(0.22, 0.18, 0.04) > timur_coates(0.22, 0.10, 0.12)

    # Cementation exponent inverts the formation factor
    F = formation_factor(0.18, m=2.1)
    m = cementation_exponent(F, 0.18)
    print(f"  cementation exponent m = {m:.3f}")
    assert abs(m - 2.1) < 1e-9

    # Joint estimate: poorer connectivity (higher m) lowers the permeability
    k_well = joint_permeability(0.20, 0.16, 0.04, m=1.8)
    k_poor = joint_permeability(0.20, 0.16, 0.04, m=2.6)
    print(f"  joint k (m=1.8 / 2.6)  = {k_well:.1f} / {k_poor:.1f} mD")
    assert k_well > k_poor > 0
    # at m = m_ref the joint estimate equals the NMR-only value
    assert abs(joint_permeability(0.20, 0.16, 0.04, m=2.0)
               - timur_coates(0.20, 0.16, 0.04)) < 1e-6
    print("  PASS")
    return {"m": float(m), "k_well": float(k_well), "k_poor": float(k_poor)}


if __name__ == "__main__":
    test_all()
