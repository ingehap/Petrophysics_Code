"""Cable / drill-string depth corrections: elastic stretch, thermal, tension.

The wireline- and driller's-depth papers correct a measured depth for the
elastic stretch of the load-bearing member (point load and its own hanging
weight), thermal elongation, and evaluate the tension profile that drives the
stretch.  Lengths, forces and moduli are in consistent units (SI by default:
metres, newtons, pascals); temperature differences in kelvin/celsius.

Sign convention for :func:`corrected_depth`: the ``'tally'`` convention treats
the surface tally as reading short (pipe/cable is longer downhole) and **adds**
the corrections, while ``'payout'`` treats paid-out cable length as overstating
the tool depth and **subtracts** them -- both are used in the corpus.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2016_06/article5_wireline_depth_elastic_stretch -- Article 5: Wireline Logging Depth Quality
  Improvement: Methodology Review and Elastic-Stretch Correction. Bolt (2016). Petrophysics Vol.
  57, No. 3 (June 2016), pp. 294-310. DOI: none assigned (this issue predates SPWLA DOI
  assignment).
src2017_12/article2_drillers_depth_waypoint -- Article 2: Driller's Depth Quality Improvement: Way-
  Point Methodology. Bolt (2017). Petrophysics Vol. 58, No. 6 (December 2017), pp. 564-575. DOI:
  none assigned (this issue predates SPWLA DOI assignment).
src2019_02/article6_depth_love_hate_essay -- Article 6: Depth: A Love and Hate Story. Theys (2019).
  DOI: 10.30632/PJV60N1Y2019a5. Petrophysics Vol. 60 No. 1 (Feb 2019).
src2019_02/article8_drillers_depth_correction -- Article 8: Correction of Driller's Depth: Field
  Example Using Driller's Way-Point Depth Correction Methodology. Bolt (2019). DOI:
  10.30632/PJV60N1Y2019a7. Petrophysics Vol. 60 No. 1 (Feb 2019).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]

#: Effective Young's modulus of steel drill pipe / wireline cable, Pa.
E_STEEL = 2.0e11
#: Linear thermal-expansion coefficient of steel, 1/K.
ALPHA_STEEL = 1.2e-5


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


def elastic_stretch(
    force: ArrayLike, length: ArrayLike, area: ArrayLike, *, E: float = E_STEEL
) -> _Float:
    """Point-load Hookean stretch ``dL = F L / (E A)``.

    ``force`` is the axial load (N), ``length`` the suspended length (m),
    ``area`` the metal cross-section (m^2), ``E`` Young's modulus (Pa).

    Sources: src2017_12/article2_drillers_depth_waypoint,
    src2019_02/article6_depth_love_hate_essay, src2019_02/article8_drillers_depth_correction.
    """
    return np.asarray(_arr(force) * _arr(length) / (E * _arr(area)))


def distributed_stretch(
    length: ArrayLike,
    weight_per_length: ArrayLike,
    ea: ArrayLike,
    *,
    end_load: ArrayLike = 0.0,
    buoyancy: float = 1.0,
) -> _Float:
    """Hanging-string stretch ``dL = (F L + 0.5 b w L^2) / (E A)``.

    Integrates the tension profile of a string of buoyant weight-per-length
    ``w`` carrying an ``end_load`` ``F`` at the bottom: the linear term is the
    end load, the quadratic term the string's own weight (scaled by the
    ``buoyancy`` factor).  ``ea`` is the lumped ``E*A`` stiffness.

    Sources: src2016_06/article5_wireline_depth_elastic_stretch,
    src2019_02/article8_drillers_depth_correction.
    """
    length_a = _arr(length)
    return np.asarray(
        (_arr(end_load) * length_a + 0.5 * buoyancy * _arr(weight_per_length) * length_a**2)
        / _arr(ea)
    )


def thermal_elongation(length: ArrayLike, dT: ArrayLike, *, alpha: float = ALPHA_STEEL) -> _Float:
    """Thermal elongation ``dL = alpha L dT``.

    ``dT`` is the temperature rise over the calibration temperature; ``alpha``
    the linear expansion coefficient (1/K).

    Sources: src2017_12/article2_drillers_depth_waypoint,
    src2019_02/article6_depth_love_hate_essay, src2019_02/article8_drillers_depth_correction.
    """
    return np.asarray(alpha * _arr(length) * _arr(dT))


def cable_tension(
    depth: ArrayLike,
    total_depth: float,
    tool_weight: float,
    cable_weight_per_length: float,
) -> _Float:
    """Tension profile ``T(z) = W_tool + w (L - z)`` along the cable.

    The tension at depth ``z`` supports the tool weight plus the buoyant weight
    of the cable hanging below it (``L = total_depth``), so it is greatest at
    surface and equals ``tool_weight`` at the tool.

    Sources: src2016_06/article5_wireline_depth_elastic_stretch.
    """
    return np.asarray(tool_weight + cable_weight_per_length * (total_depth - _arr(depth)))


def corrected_depth(
    measured: ArrayLike,
    *,
    stretch: ArrayLike = 0.0,
    thermal: ArrayLike = 0.0,
    convention: str = "tally",
) -> _Float:
    """Apply stretch and thermal corrections to a measured depth.

    ``convention='tally'`` adds the corrections (surface tally reads short);
    ``convention='payout'`` subtracts them (paid-out length overstates depth).

    Sources: src2016_06/article5_wireline_depth_elastic_stretch.
    """
    measured_a = _arr(measured)
    total = _arr(stretch) + _arr(thermal)
    if convention == "tally":
        return np.asarray(measured_a + total)
    if convention == "payout":
        return np.asarray(measured_a - total)
    raise ValueError(f"unknown convention {convention!r}; use 'tally' or 'payout'")
