"""Formation-pressure gradients and fluid contacts.

Fit a pressure-depth gradient, convert a gradient to fluid density, and locate a
fluid contact (GOC/OWC) from the intersection of two gradients.

SI by default: depth in m, pressure in Pa (``density_from_gradient`` accepts other
gradient units via ``p_unit=``).  Sources: src2017_08/article4,
src2021_02/article6, src2023_10/article_10.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2017_08/article4 -- Article 4: The Impact of Depth and Pressure Measurement Errors on the
  Estimation of Pressure Gradients. Bowers, Schnacke, Hermance (2017). Petrophysics Vol. 58, No. 4
  (August 2017), pp. 376-396. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2021_02/article6 -- Article 6: Formation Evaluation With NMR, Resistivity, and Pressure Data - A
  Case Study of a Carbonate Oil Field Offshore West Africa. Li, Drinkwater, Whittlesey, Condon
  (2021). DOI: 10.30632/PJV62N1-2021a5. Petrophysics Vol. 62 No. 1 (Feb 2021).
src2021_02/article6_nmr_resistivity_pressure_carbonate -- Article 6: Formation Evaluation With NMR,
  Resistivity, and Pressure Data - A Case Study of a Carbonate Oil Field Offshore West Africa. Li,
  Drinkwater, Whittlesey, Condon (2021). DOI: 10.30632/PJV62N1-2021a5. Petrophysics Vol. 62 No. 1
  (Feb 2021).
src2023_10/article_10 -- Mohamed, T. S., Torres-Verdin, C., and Mullins, O. C. (2023). "Enhanced
  Reservoir Description via Areal Data Integration and Reservoir Fluid Geodynamics: A Case Study
  From Deepwater Gulf of Mexico." Petrophysics, 64(5), 773-795. DOI: 10.30632/PJV64N5-2023a10.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]

G_STANDARD = 9.80665  # m/s2
_PSI_TO_PA = 6894.757
_FT_TO_M = 0.3048
_BAR_TO_PA = 1.0e5


def fit_pressure_gradient(depth_m: ArrayLike, p: ArrayLike) -> tuple[float, float]:
    """Least-squares pressure-depth gradient -> ``(dP/dz, P0)`` (slope, intercept).

    Sources: src2021_02/article6_nmr_resistivity_pressure_carbonate.
    """
    d = np.asarray(depth_m, np.float64)
    pa = np.asarray(p, np.float64)
    a = np.vstack([d, np.ones_like(d)]).T
    slope, intercept = np.linalg.lstsq(a, pa, rcond=None)[0]
    return float(slope), float(intercept)


def density_from_gradient(dpdz: ArrayLike, *, p_unit: str = "Pa") -> _Float:
    """Fluid density (kg/m3) from a pressure gradient (no clipping).

    ``p_unit``: ``"Pa"`` (Pa/m), ``"psi/ft"``, or ``"bar/m"``.  ``rho = grad_SI/g``.
    """
    grad = np.asarray(dpdz, np.float64)
    if p_unit == "Pa":
        grad_si = grad
    elif p_unit == "psi/ft":
        grad_si = grad * _PSI_TO_PA / _FT_TO_M
    elif p_unit == "bar/m":
        grad_si = grad * _BAR_TO_PA
    else:
        raise ValueError(f"p_unit must be 'Pa', 'psi/ft' or 'bar/m', got {p_unit!r}")
    return np.asarray(grad_si / G_STANDARD)


def fluid_contact(depth_a: ArrayLike, p_a: ArrayLike, depth_b: ArrayLike, p_b: ArrayLike) -> float:
    """Contact depth (m) where two fitted pressure gradients intersect.

    ``(depth_a, p_a)`` and ``(depth_b, p_b)`` are the point sets for the two
    fluid columns; each is fit to a line and the intersection depth returned.

    Sources: src2021_02/article6_nmr_resistivity_pressure_carbonate.
    """
    s1, b1 = fit_pressure_gradient(depth_a, p_a)
    s2, b2 = fit_pressure_gradient(depth_b, p_b)
    return float((b2 - b1) / (s1 - s2))
