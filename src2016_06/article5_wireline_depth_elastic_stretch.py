"""
Article 5: Wireline Logging Depth Quality Improvement: Methodology Review and
           Elastic-Stretch Correction
Bolt (2016)
Reference: Petrophysics Vol. 57, No. 3 (June 2016), pp. 294-310
DOI: none assigned (this issue predates SPWLA DOI assignment)

Wireline depth accuracy is limited by the elastic stretch of the logging cable
under the weight of the tool string and the cable itself (plus friction and
temperature).  The cable obeys Hooke's law, so the tension at any point is the
weight suspended below it and the cumulative stretch is the integral of
strain (tension / (E*A)) along the cable.  Correcting the recorded depth for
this stretch improves depth quality and the tie between logging runs.

Implements:

  - Cable tension vs. depth  T(z) = W_tool + w_cable*(L - z)
  - Stretch coefficient  ks = 1/(E*A)  (per unit tension and length)
  - Total elastic stretch  dL = ks*(W_tool*L + 0.5*w_cable*L^2)
  - Stretch-corrected depth

Note: this article's body was beyond the PDF text extraction for this issue
(the source text truncates within the preceding article), so this module is a
methodology proxy implementing the standard Hooke's-law cable-stretch correction
the title describes, consistent with how other truncated articles are handled in
this repository.  Depths/lengths in m (or ft), weights/tension in N (or lbf),
E*A in consistent force units.
"""

import numpy as np


# ---------------------------------------------- cable mechanics --------------

def cable_tension(depth, total_depth, tool_weight, cable_weight_per_length):
    """Cable tension at a measured depth z (force suspended below the point)

        T(z) = W_tool + w_cable*(L - z),

    with L the tool depth, W_tool the (buoyant) tool-string weight and w_cable
    the (buoyant) cable weight per unit length.
    """
    return tool_weight + cable_weight_per_length * (total_depth - np.asarray(depth, float))


def stretch_coefficient(youngs_modulus, cross_section_area):
    """Cable stretch coefficient  ks = 1/(E*A)  (strain per unit tension)."""
    return 1.0 / (youngs_modulus * cross_section_area)


def elastic_stretch(total_depth, tool_weight, cable_weight_per_length, ea):
    """Total elastic cable stretch from surface to the tool (Hooke's law)

        dL = (1/(E*A)) * integral_0^L T(z) dz
           = (W_tool*L + 0.5*w_cable*L^2)/(E*A),

    integrating the tension T(z) = W_tool + w_cable*(L - z) along the cable.
    """
    return (tool_weight * total_depth
            + 0.5 * cable_weight_per_length * total_depth ** 2) / ea


def stretch_corrected_depth(measured_depth, total_depth, tool_weight,
                            cable_weight_per_length, ea):
    """Stretch-corrected depth at the tool

        z_corrected = measured_depth + dL,

    adding the elastic stretch (the cable reads short because it is stretched).
    """
    dL = elastic_stretch(total_depth, tool_weight, cable_weight_per_length, ea)
    return measured_depth + dL


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Wireline Depth Elastic-Stretch Correction")
    print("=" * 60)

    L = 3000.0                      # m
    w_tool = 2000.0                 # N (tool-string buoyant weight)
    w_cable = 5.0                   # N/m (cable buoyant weight per length)
    ea = 3.0e7                      # N (E*A of the cable)

    # Tension is largest at surface (whole string below) and least at the tool
    t_surface = cable_tension(0.0, L, w_tool, w_cable)
    t_tool = cable_tension(L, L, w_tool, w_cable)
    print(f"  tension surface/tool   = {t_surface:.0f} / {t_tool:.0f} N")
    assert t_surface > t_tool and np.isclose(t_tool, w_tool)

    # Stretch coefficient is positive and stretch grows with depth (super-linear)
    ks = stretch_coefficient(youngs_modulus=1.0e11, cross_section_area=3.0e-4)
    assert ks > 0
    dL1 = elastic_stretch(1500.0, w_tool, w_cable, ea)
    dL2 = elastic_stretch(3000.0, w_tool, w_cable, ea)
    print(f"  stretch @1500/3000 m   = {dL1:.3f} / {dL2:.3f} m")
    assert dL2 > 2.0 * dL1 > 0      # cable-weight term makes it super-linear

    # Corrected depth exceeds the measured depth by the stretch
    zc = stretch_corrected_depth(L, L, w_tool, w_cable, ea)
    assert np.isclose(zc - L, dL2)
    print(f"  corrected depth        = {zc:.3f} m (+{zc - L:.3f})")
    print("  PASS")
    return {"stretch_3000m": float(dL2), "tension_surface": float(t_surface)}


if __name__ == "__main__":
    test_all()
