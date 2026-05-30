"""
Article 6: Depth: A Love and Hate Story
Theys (2019)
DOI: 10.30632/PJV60N1Y2019a5

A reflective essay on the fundamental challenge of depth measurement in well
logging.  Wireline depth is the cable length paid out, but the cable stretches
elastically under tension (tool weight + friction) and expands thermally with
the geothermal gradient, so the apparent depth must be corrected to recover true
depth.

Implements:

  - Elastic cable stretch  dL = T*L/(E*A)
  - Thermal cable expansion  dL = alpha*L*dT
  - Total depth correction (elastic + thermal)
  - Depth uncertainty combining systematic and random terms

Note: this issue's PDF has a text layer but the piece is a narrative essay; this
module implements the standard cable-stretch / depth-correction relations the
essay discusses (typeset glyphs were dropped in extraction).  SI units.
"""

import numpy as np

E_STEEL = 2.0e11         # Pa, cable effective Young's modulus
ALPHA_STEEL = 1.2e-5     # 1/K, thermal expansion coefficient


# ---------------------------------------------- stretch -----------------

def elastic_stretch(tension, length, area, E=E_STEEL):
    """Elastic cable stretch  dL = T*L/(E*A)  (m)."""
    return tension * length / (E * area)


def thermal_expansion(length, dT, alpha=ALPHA_STEEL):
    """Thermal cable expansion  dL = alpha*L*dT  (m)."""
    return alpha * length * dT


def true_depth(apparent_depth, tension, area, dT, E=E_STEEL, alpha=ALPHA_STEEL):
    """True depth = apparent (paid-out) depth minus elastic + thermal stretch.

    The cable reads deeper than the tool actually is, so the corrections are
    subtracted from the apparent depth.
    """
    return (apparent_depth
            - elastic_stretch(tension, apparent_depth, area, E)
            - thermal_expansion(apparent_depth, dT, alpha))


def depth_uncertainty(systematic, random_per_km, depth_km):
    """Combine a systematic depth error with a random term growing with depth.

        sigma = sqrt(systematic^2 + (random_per_km*depth_km)^2)
    """
    return np.hypot(systematic, random_per_km * depth_km)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Depth - A Love and Hate Story")
    print("=" * 60)

    # Elastic stretch grows with tension and depth (length)
    area = 1e-4                                    # m^2 effective cable area
    s_shallow = elastic_stretch(2e4, 1000.0, area)
    s_deep = elastic_stretch(2e4, 4000.0, area)
    print(f"  elastic stretch 1/4 km = {s_shallow:.3f} / {s_deep:.3f} m")
    assert s_deep > s_shallow > 0
    assert elastic_stretch(4e4, 1000.0, area) > s_shallow   # more tension

    # Thermal expansion grows with the temperature increase
    assert thermal_expansion(4000.0, 80.0) > thermal_expansion(4000.0, 20.0)

    # True depth is shallower than the apparent (paid-out) depth
    td = true_depth(4000.0, tension=3e4, area=area, dT=90.0)
    correction = 4000.0 - td
    print(f"  apparent 4000 -> true {td:.2f} m  (correction {correction:.2f} m)")
    assert td < 4000.0 and correction > 0

    # Depth uncertainty grows with depth
    u1 = depth_uncertainty(0.3, 0.2, 1.0)
    u4 = depth_uncertainty(0.3, 0.2, 4.0)
    print(f"  depth uncertainty 1/4 km = {u1:.2f} / {u4:.2f} m")
    assert u4 > u1
    print("  PASS")
    return {"stretch_4km": float(s_deep), "correction_m": float(correction)}


if __name__ == "__main__":
    test_all()
