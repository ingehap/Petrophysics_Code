"""
Article 8: Correction of Driller's Depth: Field Example Using Driller's
           Way-Point Depth Correction Methodology
Bolt (2019)
DOI: 10.30632/PJV60N1Y2019a7

Driller's (along-hole) depth from pipe tallies is biased because the drillstring
stretches elastically under its own buoyed weight and the hook load, and expands
thermally with the geothermal gradient.  A way-point methodology corrects the
raw driller's depth at calibration points and interpolates the correction
between them.

Implements:

  - Drillstring elastic stretch under buoyed weight + hook load
  - Thermal elongation of the drillstring
  - Corrected driller's depth (raw + stretch + thermal)
  - Way-point correction by linear interpolation between calibration depths

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the driller's-depth-correction methodology the paper presents.  SI units.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

E_STEEL = 2.0e11
ALPHA_STEEL = 1.2e-5
G = 9.81


# ---------------------------------------------- stretch -----------------

def own_weight_stretch(length, weight_per_m, area, buoyancy=0.85, E=E_STEEL):
    """Elastic stretch of a hanging string under its own buoyed weight.

        dL = (buoyancy*w*g*L^2)/(2*E*A)
    (the L/2 average tension acts over length L).
    """
    return petrolib.depth_correction.distributed_stretch(
        length, weight_per_m * G, E * area, buoyancy=buoyancy)


def hookload_stretch(hook_load, length, area, E=E_STEEL):
    """Elastic stretch from an added hook/overpull load  dL = F*L/(E*A)."""
    return petrolib.depth_correction.elastic_stretch(hook_load, length, area, E=E)


def thermal_elongation(length, dT, alpha=ALPHA_STEEL):
    """Thermal elongation  dL = alpha*L*dT."""
    return petrolib.depth_correction.thermal_elongation(length, dT, alpha=alpha)


def corrected_depth(raw_depth, weight_per_m, area, dT, hook_load=0.0):
    """Corrected driller's depth = raw + own-weight + hook-load + thermal stretch.

    The pipe is longer downhole than the surface tally indicates, so the
    stretch is added to the raw along-hole depth.
    """
    return (raw_depth
            + own_weight_stretch(raw_depth, weight_per_m, area)
            + hookload_stretch(hook_load, raw_depth, area)
            + thermal_elongation(raw_depth, dT))


# ---------------------------------------------- way-point ---------------

def waypoint_correction(depth, wp_depths, wp_corrections):
    """Interpolate the depth correction between calibration way-points."""
    return np.interp(np.asarray(depth, float), wp_depths, wp_corrections)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Correction of Driller's Depth (Way-Point)")
    print("=" * 60)

    area = 35.8e-4           # m^2 (drillpipe steel cross-section, ~5.5 in^2)
    w = 30.0                 # kg/m drillpipe weight

    # Own-weight stretch grows with the square of length
    s2 = own_weight_stretch(2000.0, w, area)
    s4 = own_weight_stretch(4000.0, w, area)
    print(f"  own-weight stretch 2/4 km = {s2:.3f} / {s4:.3f} m")
    assert abs(s4 / s2 - 4.0) < 1e-6              # ~ L^2

    # Hook load and thermal both add elongation
    assert hookload_stretch(5e5, 4000.0, area) > 0
    assert thermal_elongation(4000.0, 90.0) > thermal_elongation(4000.0, 30.0)

    # Corrected depth is deeper than the raw tally (pipe stretches downhole)
    cd = corrected_depth(4000.0, w, area, dT=90.0, hook_load=2e5)
    correction = cd - 4000.0
    print(f"  raw 4000 -> corrected {cd:.2f} m  (+{correction:.2f} m)")
    assert cd > 4000.0 and correction > 0

    # Way-point interpolation between calibration corrections
    wp_d = [0.0, 2000.0, 4000.0]
    wp_c = [0.0, 0.5, 2.0]
    c_mid = waypoint_correction(3000.0, wp_d, wp_c)
    print(f"  way-point correction @3000 m = {c_mid:.3f} m")
    assert abs(c_mid - 1.25) < 1e-9               # midway between 0.5 and 2.0
    print("  PASS")
    return {"stretch_4km": float(s4), "correction_m": float(correction)}


if __name__ == "__main__":
    test_all()
