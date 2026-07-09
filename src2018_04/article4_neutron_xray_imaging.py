"""
Article 4: Simultaneous Neutron and X-Ray Imaging of 3D Structure of Organic
           Matter and Fracture in Shales
Chiang, LaManna, Hussey, Jacobson, Liu, Zhang, Georgi, Kone, Chen (2018)
DOI: 10.30632/PJV59N2-2018a3

Simultaneous neutron and X-ray tomography separates organic matter, high-Z
minerals, and fractures in 3D.  Both modalities follow Lambert-Beer attenuation
with an attenuation coefficient that sums the per-atom cross sections times
number densities, but the contrast is orthogonal: hydrogen-rich organic matter
is bright to neutrons (independent of atomic number), high-Z minerals are bright
to X-rays, and open fractures are dark in both - so intersecting the two volumes
segments kerogen, heavy minerals, and void.

Implements:

  - Lambert-Beer attenuation  It = I0*exp(-mu*d)
  - Attenuation coefficient  mu = sum_i sigma_i*N_i
  - Optical density / transmittance from the two modalities
  - Voxel segmentation from the (neutron, X-ray) contrast rule

Note: this issue's PDF has a text layer and this article's physics equations
(Eqs. 1-4) largely survived; the relations below are transcribed/standard-form.
Consistent units (mu in 1/cm, d in cm, cross section x number density -> 1/cm).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- attenuation --------------

def lambert_beer(i0, mu, d):
    """Transmitted intensity  It = I0*exp(-mu*d)  (Eq. 1)."""
    return petrolib.nuclear.beer_lambert(i0, mu, d)


def attenuation_coefficient(cross_sections, number_densities):
    """Attenuation coefficient  mu = sum_i sigma_i*N_i  (Eq. 2)."""
    return float(np.sum(np.asarray(cross_sections, float) * np.asarray(number_densities, float)))


def optical_density(i0, it):
    """Optical density (absorbance)  OD = -ln(It/I0) = mu*d."""
    return petrolib.nuclear.attenuation_map(it, i0)


def segment_voxel(neutron_atten, xray_atten, hi_thresh, z_thresh):
    """Classify a voxel from the (neutron, X-ray) attenuation contrast rule.

      - high neutron, low X-ray  -> organic matter (H-rich, low Z)
      - high X-ray               -> high-Z mineral
      - low in both              -> open fracture / void
    """
    if xray_atten >= z_thresh:
        return "mineral"
    if neutron_atten >= hi_thresh:
        return "organic"
    return "void"


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Neutron & X-Ray Imaging")
    print("=" * 60)

    # Lambert-Beer: thicker / more attenuating sample transmits less
    assert lambert_beer(1000.0, 0.5, 2.0) < lambert_beer(1000.0, 0.5, 1.0)

    # mu sums the per-component cross-section x number-density contributions
    mu = attenuation_coefficient([2.0, 0.5], [0.1, 0.4])
    print(f"  mu (sum sigma*N)       = {mu:.3f} 1/cm")
    assert np.isclose(mu, 2.0 * 0.1 + 0.5 * 0.4)

    # Optical density recovers mu*d from the transmitted intensity
    it = lambert_beer(1000.0, mu, 1.5)
    assert np.isclose(optical_density(1000.0, it), mu * 1.5)

    # Orthogonal-contrast segmentation of three voxel types
    org = segment_voxel(neutron_atten=0.8, xray_atten=0.1, hi_thresh=0.5, z_thresh=0.5)
    mineral = segment_voxel(0.2, 0.9, 0.5, 0.5)
    void = segment_voxel(0.1, 0.1, 0.5, 0.5)
    print(f"  voxel classes          = {org} / {mineral} / {void}")
    assert (org, mineral, void) == ("organic", "mineral", "void")
    print("  PASS")
    return {"mu": float(mu)}


if __name__ == "__main__":
    test_all()
