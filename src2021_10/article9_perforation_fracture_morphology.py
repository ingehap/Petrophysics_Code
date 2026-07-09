"""
Article 9: Research of Near-Wellbore Fracture Morphology, Formation Mechanism,
           and Propagation Law for Different Perforation Modes During the
           Perforation Process
Wang, Li, Xu, Jia, Zhang (2021)
DOI: 10.30632/PJV62N5-2021a9

*Methodology proxy.*  Only the abstract, introduction and experimental setup
of this article were present in the source-PDF text extract used to build
this folder (the extract was truncated partway through the results), and the
paper itself - an experimental perforation study - transcribes no equations.
This module is a methodology proxy that (a) encodes the experimental
perforation-mode parameter taxonomy and three-stage / three-microfracture-type
classification the paper describes, and (b) adds the standard near-wellbore
stress relations (Kirsch / breakdown pressure) that govern fracture initiation,
flagged as standard forms not transcribed from this paper.

Implements:

  - Kirsch tangential stress at the borehole wall                (R1)
  - Tensile fracture-initiation / breakdown pressure             (R2)
  - Perforation-mode classification (spiral / directional / fixed-plane)
  - Microfracture-type taxonomy (radial / oblique / tip-divergent)

Stresses in MPa, angles in degrees.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

MICROFRACTURE_TYPES = {
    1: "radial",            # Type I  - radial from the perforation
    2: "oblique",           # Type II - oblique to the perforation axis
    3: "tip-divergent",     # Type III- divergent at the perforation tip
}


# ---------------------------------------------- R1: Kirsch --------------

def kirsch_tangential_stress(s_H, s_h, p_w, theta_deg):
    """Tangential (hoop) stress at the borehole wall (R1).

    sigma_theta = (s_H + s_h) - 2*(s_H - s_h)*cos(2*theta) - p_w,
    with theta measured from the maximum horizontal stress azimuth.
    """
    return petrolib.acoustic_geomech.kirsch_hoop_stress(s_H, s_h, p_w, theta_deg)


# ---------------------------------------------- R2: breakdown -----------

def breakdown_pressure(s_H, s_h, p0, tensile_strength):
    """Fracture-initiation (breakdown) pressure for an impermeable wall (R2).

    P_b = 3*s_h - s_H - p0 + T  (Hubbert-Willis / Haimson-Fairhurst form).
    """
    return petrolib.acoustic_geomech.breakdown_pressure(
        s_h, s_H, p0, tensile_strength=tensile_strength)


# ---------------------------------------------- perforation modes -------

def classify_perforation(mode, fixed_plane_angle=None, interlaced_angle=None):
    """Classify a perforation arrangement.

    An interlaced fixed-plane gun with (fixed + interlaced) summing to a full
    realignment (60+120 or 90+180) degenerates to a conventional fixed plane.
    """
    if mode == "interlaced":
        # interlaced == 2*fixed-plane (60->120, 90->180) realigns to conventional
        if interlaced_angle is not None and fixed_plane_angle is not None \
                and abs(interlaced_angle - 2 * fixed_plane_angle) < 1e-9:
            return "conventional"
        return "interlaced"
    return mode


def microfracture_label(type_id):
    """Name a microfracture type from its integer id."""
    return MICROFRACTURE_TYPES[type_id]


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: Near-Wellbore Perforation Fracture (proxy)")
    print("=" * 60)

    # Kirsch: hoop stress is lowest aligned with the maximum horizontal stress
    s_H, s_h, p_w = 45.0, 30.0, 0.0
    theta = np.linspace(0, 180, 181)
    hoop = kirsch_tangential_stress(s_H, s_h, p_w, theta)
    theta_min = theta[int(np.argmin(hoop))]
    print(f"  hoop-stress minimum at = {theta_min:.0f} deg (expect 0/180)")
    assert theta_min in (0.0, 180.0)

    # Breakdown pressure (R2)
    pb = breakdown_pressure(s_H=45.0, s_h=30.0, p0=20.0, tensile_strength=5.0)
    print(f"  breakdown pressure     = {pb:.0f} MPa")
    assert abs(pb - 30.0) < 1e-9

    # Perforation-mode classification
    assert classify_perforation("spiral") == "spiral"
    assert classify_perforation("interlaced", 60, 120) == "conventional"
    assert classify_perforation("interlaced", 90, 180) == "conventional"
    assert classify_perforation("interlaced", 90, 45) == "interlaced"
    print(f"  perforation modes      = spiral / directional / fixed-plane OK")

    # Microfracture taxonomy
    assert microfracture_label(1) == "radial"
    assert microfracture_label(3) == "tip-divergent"
    print("  PASS")
    return {"theta_min": theta_min, "breakdown_MPa": pb}


if __name__ == "__main__":
    test_all()
