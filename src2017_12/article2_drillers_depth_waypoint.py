"""
Article 2: Driller's Depth Quality Improvement: Way-Point Methodology
Bolt (2017)
Reference: Petrophysics Vol. 58, No. 6 (December 2017), pp. 564-575
DOI: none assigned (this issue predates SPWLA DOI assignment)

Driller's depth from the pipe tally is biased because the drillstring elongates
under temperature and axial load.  The way-point methodology corrects it
station-by-station during a constant-speed pull out of hole, where friction and
hydraulic terms repeat and only thermal expansion and elastic stretch remain.
This module computes the per-station thermal and elastic-stretch corrections,
sums them into a corrected (true along-hole) depth, and combines the error terms.

Implements:

  - Thermal correction  dL = L*alpha*((T_top + T_btm)/2 - T_calib)
  - Drillpipe metal cross-section  A = (pi/4)*(OD^2 - ID^2)
  - Elastic stretch  dL = F*L/(E*A)  and the stretch coefficient 1/(E*A)
  - Way-point summed correction and quadrature uncertainty

Note: this issue's PDF has a text layer but its typeset display-equation glyphs
were dropped in extraction, so the numbered relations (Eqs. 1-12) are faithful
standard-form reconstructions from the surviving variable definitions.  SI units
(length m, force N, stress Pa, temperature consistent with alpha).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

E_STEEL = 2.0e11             # Young's modulus, steel (Pa)
ALPHA_STEEL = 1.5e-5         # linear thermal-expansion coefficient, steel (1/K)


# ---------------------------------------------- corrections --------------

def thermal_correction(length, t_top, t_btm, t_calib, alpha=ALPHA_STEEL):
    """Thermal elongation of a segment  dL = L*alpha*(Tmean - T_calib)  (Eq. 1).

    Tmean = (T_top + T_btm)/2 over the segment; T_calib = pipe temperature when
    the tally length was measured.
    """
    t_mean = 0.5 * (np.asarray(t_top, float) + np.asarray(t_btm, float))
    return petrolib.depth_correction.thermal_elongation(length, t_mean - t_calib, alpha=alpha)


def cross_section_area(od, id_):
    """Drillpipe metal cross-sectional area  A = (pi/4)*(OD^2 - ID^2)  (Eq. 6)."""
    return (np.pi / 4.0) * (od ** 2 - id_ ** 2)


def stretch_coefficient(area, youngs=E_STEEL):
    """Elastic-stretch coefficient  St = 1/(E*A)  (Eq. 6)."""
    return 1.0 / (youngs * area)


def elastic_stretch(force, length, area, youngs=E_STEEL):
    """Elastic stretch of a segment under axial load  dL = F*L/(E*A)  (Eqs. 2-8)."""
    return petrolib.depth_correction.elastic_stretch(force, length, area, E=youngs)


def waypoint_correction(lengths, t_top, t_btm, tension, area,
                        t_calib=20.0, alpha=ALPHA_STEEL, youngs=E_STEEL):
    """Total way-point depth correction = sum of thermal + elastic-stretch per station."""
    lengths = np.asarray(lengths, float)
    therm = thermal_correction(lengths, t_top, t_btm, t_calib, alpha).sum()
    stretch = elastic_stretch(np.asarray(tension, float), lengths, area, youngs).sum()
    return float(therm + stretch)


def correction_uncertainty(thermal_err, stretch_err):
    """Combined correction uncertainty by quadrature  sqrt(eth^2 + estr^2)  (Eqs. 9-12)."""
    return float(np.hypot(thermal_err, stretch_err))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Driller's Depth Way-Point Methodology")
    print("=" * 60)

    # Hotter downhole than at calibration -> the pipe lengthens
    dL = thermal_correction(1000.0, t_top=80.0, t_btm=120.0, t_calib=20.0)
    print(f"  thermal elongation     = {dL:.3f} m over 1000 m")
    assert dL > 0 and thermal_correction(1000.0, 20.0, 20.0, 20.0) == 0.0

    # Thicker wall -> larger area -> less stretch under the same load
    a_thin = cross_section_area(0.127, 0.109)
    a_thick = cross_section_area(0.127, 0.095)
    assert a_thick > a_thin
    assert elastic_stretch(5e5, 1000.0, a_thin) > elastic_stretch(5e5, 1000.0, a_thick)

    # Way-point sum over three stations (constant-speed POOH)
    lengths = np.full(3, 1000.0)
    total = waypoint_correction(lengths, t_top=[60, 90, 120], t_btm=[90, 120, 150],
                                tension=[8e5, 5e5, 2e5], area=a_thin)
    print(f"  total depth correction = {total:.3f} m")
    assert total > 0

    # Uncertainty combines in quadrature
    assert np.isclose(correction_uncertainty(0.3, 0.4), 0.5)
    print("  PASS")
    return {"thermal_dL": float(dL), "total_correction": total}


if __name__ == "__main__":
    test_all()
