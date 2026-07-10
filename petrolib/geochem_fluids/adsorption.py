"""Gas adsorption isotherms and shale gas-in-place volumetrics.

Langmuir isotherm (2-parameter, with an optional bulk-density scaling for the
sorbed-gas-content form), Gibbs excess correction, BET isotherm and a BET
surface-area fit, free-gas porosity volumetrics, and gas-in-place.

Units: pressure and Langmuir pressure in the same unit; Langmuir volume V_L and
BET volumes in cm3(STP)/g; SSA in m2/g; Bg the gas formation-volume factor.
Sources: src2018_02/article10, src2019_06/article2, src2019_10/article2,
src2020_10/article2, src2025_12/nanopore_adsorption.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2018_02/article10 -- Article 10: New Perspectives on the Effects of Gas Adsorption on Storage
  and Production of Natural Gas From Shale Formations. Tinni, Sondergeld, Rai (2018). DOI:
  10.30632/petro_059_1_a9 (contents-only - see note). Petrophysics Vol. 59 No. 1 (Feb 2018).
src2018_02/article10_shale_gas_adsorption -- Article 10: New Perspectives on the Effects of Gas
  Adsorption on Storage and Production of Natural Gas From Shale Formations. Tinni, Sondergeld, Rai
  (2018). DOI: 10.30632/petro_059_1_a9 (contents-only - see note). Petrophysics Vol. 59 No. 1 (Feb
  2018).
src2019_02/article7_groningen_depth_control -- Article 7 (Technical Note): Connecting the Dots -
  The Relevance of Proper Depth Control in the Discovery of the Groningen Field. Fokkema, Visser
  (2019). DOI: 10.30632/PJV60N1Y2019a6. Petrophysics Vol. 60 No. 1 (Feb 2019).
src2019_06/article2 -- Article 2: Composition of the Shales in Niutitang Formation at Huijunba
  Syncline and its Influence on Microscopic Pore Structure and Gas Adsorption. Fu, Xu, Tian, Qin,
  Yang (2019). DOI: 10.30632/PJV60N3-2019a1. Petrophysics Vol. 60 No. 3 (Jun 2019).
src2019_06/article2_niutitang_shale_pore_adsorption -- Article 2: Composition of the Shales in
  Niutitang Formation at Huijunba Syncline and its Influence on Microscopic Pore Structure and Gas
  Adsorption. Fu, Xu, Tian, Qin, Yang (2019). DOI: 10.30632/PJV60N3-2019a1. Petrophysics Vol. 60
  No. 3 (Jun 2019).
src2019_10/article2 -- Article 2: More Accurate Quantification of Free and Adsorbed Gas in Shale
  Reservoirs. Ansari, Merletti, Gramin, Armitage (2019). DOI: 10.30632/PJV60N5-2019a2. Petrophysics
  Vol. 60 No. 5 (Oct 2019).
src2019_10/article2_free_adsorbed_gas_shale -- Article 2: More Accurate Quantification of Free and
  Adsorbed Gas in Shale Reservoirs. Ansari, Merletti, Gramin, Armitage (2019). DOI:
  10.30632/PJV60N5-2019a2. Petrophysics Vol. 60 No. 5 (Oct 2019).
src2020_10/article2 -- Article 2: Classification of Adsorption Isotherm Curves for Shale Based on
  Pore Structure. Tian, Chen, Yan, Deng, He (2020). DOI: 10.30632/PJV61N5-2020a2. Petrophysics Vol.
  61 No. 5 (Oct 2020).
src2020_10/article2_adsorption_isotherm_classification -- Article 2: Classification of Adsorption
  Isotherm Curves for Shale Based on Pore Structure. Tian, Chen, Yan, Deng, He (2020). DOI:
  10.30632/PJV61N5-2020a2. Petrophysics Vol. 61 No. 5 (Oct 2020).
src2025_12/co2_uptake -- CO₂ Uptake Capacity Measurement in Source-Rock Shales for GCS.
  Petrophysics, 66(6), 982–994. DOI: 10.30632/PJV66N6-2025a5.
src2025_12/nanopore_adsorption -- Wettability Effect on Adsorption and Capillary Condensation in
  Nanopores. Petrophysics, 66(6), 1061–1071. DOI: 10.30632/PJV66N6-2025a10.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]

N_AVOGADRO = 6.022e23  # 1/mol
V_MOLAR_STP_CM3 = 22414.0  # cm3/mol at STP
N2_CROSS_M2 = 0.162e-18  # N2 molecular cross-section, m2


def langmuir(
    p: ArrayLike, v_l: ArrayLike, p_l: ArrayLike, *, rho_b: ArrayLike | None = None
) -> _Float:
    """Langmuir isotherm ``V = V_L * P/(P + P_L)`` (half capacity at ``P = P_L``).

    With ``rho_b`` given, returns the sorbed gas *content* ``rho_b * V_L * P/(P+P_L)``.

    Sources: src2018_02/article10_shale_gas_adsorption,
    src2019_06/article2_niutitang_shale_pore_adsorption,
    src2019_10/article2_free_adsorbed_gas_shale, src2025_12/co2_uptake.
    """
    pa = np.asarray(p, np.float64)
    v = np.asarray(v_l, np.float64) * pa / (pa + np.asarray(p_l, np.float64))
    if rho_b is not None:
        v = v * np.asarray(rho_b, np.float64)
    return np.asarray(v)


def gibbs_excess(gc: ArrayLike, rho_free: ArrayLike, rho_ads: ArrayLike) -> _Float:
    """Gibbs excess correction ``G_excess = Gc*(1 - rho_free/rho_adsorbed)``.

    Sources: src2019_10/article2_free_adsorbed_gas_shale.
    """
    return np.asarray(
        np.asarray(gc, np.float64)
        * (1.0 - np.asarray(rho_free, np.float64) / np.asarray(rho_ads, np.float64))
    )


def bet_isotherm(x_rel: ArrayLike, vm: ArrayLike, c: ArrayLike) -> _Float:
    """BET isotherm ``V = Vm*C*x/((1-x)*(1 + (C-1)*x))`` with ``x = P/P0``.

    Sources: src2020_10/article2_adsorption_isotherm_classification.
    """
    x = np.asarray(x_rel, np.float64)
    vma = np.asarray(vm, np.float64)
    ca = np.asarray(c, np.float64)
    return np.asarray(vma * ca * x / ((1.0 - x) * (1.0 + (ca - 1.0) * x)))


def bet_fit(
    x_rel: ArrayLike, v_ads: ArrayLike, *, cross_nm2: float = 0.162
) -> tuple[float, float, float]:
    """Linear BET fit -> ``(Vm, C, SSA_m2_g)``.

    Fits ``x/(V*(1-x)) = slope*x + intercept`` (least squares); ``Vm =
    1/(slope+intercept)``, ``C = slope/intercept + 1``, and the specific surface
    area ``SSA = (Vm/22414)*NA*cross`` (m2/g) with ``cross`` in nm2 (N2 = 0.162).

    Sources: src2020_10/article2_adsorption_isotherm_classification.
    """
    x = np.asarray(x_rel, np.float64)
    v = np.asarray(v_ads, np.float64)
    y = x / (v * (1.0 - x))
    a = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(a, y, rcond=None)[0]
    vm = 1.0 / (slope + intercept)
    c = slope / intercept + 1.0
    ssa = (vm / V_MOLAR_STP_CM3) * N_AVOGADRO * (cross_nm2 * 1e-18)
    return float(vm), float(c), float(ssa)


def free_gas(phi: ArrayLike, sw: ArrayLike, bg: ArrayLike) -> _Float:
    """Free-gas volume per bulk volume ``= phi*(1 - Sw)/Bg``.

    Sources: src2018_02/article10_shale_gas_adsorption,
    src2019_10/article2_free_adsorbed_gas_shale.
    """
    return np.asarray(
        np.asarray(phi, np.float64)
        * (1.0 - np.asarray(sw, np.float64))
        / np.asarray(bg, np.float64)
    )


def gas_in_place(
    area_m2: ArrayLike, h_m: ArrayLike, phi: ArrayLike, sg: ArrayLike, bg: ArrayLike
) -> _Float:
    """Volumetric free gas-in-place ``= A*h*phi*Sg/Bg``.

    Sources: src2019_02/article7_groningen_depth_control.
    """
    return np.asarray(
        np.asarray(area_m2, np.float64)
        * np.asarray(h_m, np.float64)
        * np.asarray(phi, np.float64)
        * np.asarray(sg, np.float64)
        / np.asarray(bg, np.float64)
    )
