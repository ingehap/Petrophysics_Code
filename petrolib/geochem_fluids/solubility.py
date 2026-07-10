"""Gas solubility in brine and oil (Henry / Setschenow / Duan-Sun style).

CO2 Henry constant vs temperature, the Setschenow salting-out factor, CO2
solubility in brine (with optional CH4 competition), and a Krichevsky-Kasarnovsky
gas-in-oil solubility.

Units: pressure in MPa, temperature in K, salinity as NaCl molality (mol/kg),
solubility in mol/kg.  Sources: src2022_04/article5, src2023_12/wang,
src2026_02/co2_sequestration.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2022_04/article5 -- Article 5: Experimental Investigation on the Effect of Methane Solubility in
  Oil-Based Mud Under Downhole Conditions. Song, Sukari, Wang, Jiang, Cai, Xu, Huang (2022). DOI:
  10.30632/PJV63N2-2022a5. Petrophysics Vol. 63 No. 2 (Apr 2022).
src2022_04/article5_methane_solubility_obm -- Article 5: Experimental Investigation on the Effect
  of Methane Solubility in Oil-Based Mud Under Downhole Conditions. Song, Sukari, Wang, Jiang, Cai,
  Xu, Huang (2022). DOI: 10.30632/PJV63N2-2022a5. Petrophysics Vol. 63 No. 2 (Apr 2022).
src2023_12/wang -- Wang and Ehlig-Economides (2023), Petrophysics 64(6): 970-977. CO2 solubility in
  saline water for CO2 trapping, accounting for pressure, temperature, salinity, and an often-
  neglected factor (e.g., dissolved CH4 or the activity-coefficient correction). Implements a Duan-
  Sun-style solubility model in simplified form.
src2023_12/wang_co2_solubility -- Wang and Ehlig-Economides (2023), Petrophysics 64(6): 970-977.
  CO2 solubility in saline water for CO2 trapping, accounting for pressure, temperature, salinity,
  and an often-neglected factor (e.g., dissolved CH4 or the activity-coefficient correction).
  Implements a Duan-Sun-style solubility model in simplified form.
src2026_02/co2_sequestration -- Article 8: Effect of CO2 Sequestration on Carbonate Formation
  Integrity - Aspects of Static Interactions. Al-Hamad et al. (2026), Petrophysics, 67(1), 106-122.
  DOI: 10.30632/PJV67N1-2026a8.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]

R_GAS = 8.314  # J/(mol*K)


def henry_constant_co2(t_k: ArrayLike) -> _Float:
    """CO2 Henry constant (MPa/molality), Duan-Sun style vs temperature (K).

    ``H = 10*exp(-9.4234 + 4.0087e-2*T - 4.5e-5*T**2)``.

    Sources: src2023_12/wang_co2_solubility.
    """
    t = np.asarray(t_k, np.float64)
    return np.asarray(10.0 * np.exp(-9.4234 + 4.0087e-2 * t - 4.5e-5 * t**2))


def setschenow_factor(m_nacl: ArrayLike, t_k: ArrayLike, *, ks25: float = 0.11) -> _Float:
    """Setschenow salting-out activity factor ``exp(2*ks*m)``.

    ``ks = ks25 + 1e-4*(T - 298)``; ``m`` is the NaCl molality (mol/kg).
    """
    ks = ks25 + 1.0e-4 * (np.asarray(t_k, np.float64) - 298.0)
    return np.asarray(np.exp(2.0 * ks * np.asarray(m_nacl, np.float64)))


def co2_solubility_brine(
    p_mpa: ArrayLike, t_k: ArrayLike, m_nacl: ArrayLike = 0.0, *, m_ch4: ArrayLike = 0.0
) -> _Float:
    """CO2 solubility in brine (mol/kg): ``P/(H*gamma) * 1/(1 + 0.6*m_CH4)``.

    ``H`` from :func:`henry_constant_co2`, ``gamma`` from :func:`setschenow_factor`;
    ``m_ch4`` accounts for methane competition.

    Sources: src2023_12/wang_co2_solubility.
    """
    h = henry_constant_co2(t_k)
    gamma = setschenow_factor(m_nacl, t_k)
    base = np.asarray(p_mpa, np.float64) / (h * gamma)
    return np.asarray(base / (1.0 + 0.6 * np.asarray(m_ch4, np.float64)))


def henry_solubility_ln(
    p_mpa: ArrayLike, t_k: ArrayLike, a: ArrayLike, b: ArrayLike, dh_j_mol: ArrayLike
) -> _Float:
    """Krichevsky-Kasarnovsky ``ln(x) = a + b*ln(P) - dH/(R*T)`` (gas-in-oil mole fraction).

    Sources: src2022_04/article5_methane_solubility_obm.
    """
    return np.asarray(
        np.asarray(a, np.float64)
        + np.asarray(b, np.float64) * np.log(np.asarray(p_mpa, np.float64))
        - np.asarray(dh_j_mol, np.float64) / (R_GAS * np.asarray(t_k, np.float64))
    )
