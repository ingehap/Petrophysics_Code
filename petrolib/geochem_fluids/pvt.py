"""Gas PVT: pseudo-reduced properties, z-factor, density, and phase equilibrium.

Pseudo-reduced pressure/temperature, the Beggs-Brill (1973) and Peng-Robinson
z-factor correlations, gas density and its inverse, mixture molecular weight and
gas gravity, Wilson K-values, and a Rachford-Rice vapor-fraction solve.

Strict SI: pressures in Pa, temperature in K, molar mass in kg/mol, density in
kg/m3.  Sources: src2016_04/article4 (Wilson/Rachford-Rice/Peng-Robinson),
src2019_08/article3 (Beggs-Brill), src2020_12/article6.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2016_04/article4 -- Article 4: A Multilevel Iterative Method to Quantify Effects of Pore-Size
  Distribution on Phase Equilibrium of Multicomponent Fluids in Unconventional Plays. Li,
  Mezzatesta, Li, Ma, Jamili (2016). Petrophysics Vol. 57, No. 2 (April 2016), pp. 121-139. DOI:
  none assigned (this issue predates SPWLA DOI assignment).
src2017_06/article2_tgip_nmr_gas_shale -- Article 2: A Novel Determination of Total Gas-In-Place
  (TGIP) for Gas Shale From Magnetic Resonance Logs. Kausik, Kleinberg, Rylander, Lewis, Sibbit,
  Westacott (2017). Petrophysics Vol. 58, No. 3 (June 2017), pp. 232-241. DOI: none assigned (this
  issue predates SPWLA DOI assignment).
src2019_08/article3 -- Article 3: The Compressibility Factor (Z) of Shale Gas at the Core Scale.
  Tran, Sakhaee-Pour (2019). DOI: 10.30632/PJV60N4-2019a3. Petrophysics Vol. 60 No. 4 (Aug 2019).
src2019_08/article3_shale_gas_z_factor -- Article 3: The Compressibility Factor (Z) of Shale Gas at
  the Core Scale. Tran, Sakhaee-Pour (2019). DOI: 10.30632/PJV60N4-2019a3. Petrophysics Vol. 60 No.
  4 (Aug 2019).
src2020_12/article6 -- Article 6: Maximizing the Value of Pulsed-Neutron Logs - A Complex Case
  Study of Gas Pressure Assessment Through Casing. Cavalleri, Brouwer, Kodri, Rose, Brinks (2020).
  DOI: 10.30632/PJV61N6-2020a6. Petrophysics Vol. 61 No. 6 (Dec 2020).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]

R_GAS = 8.314  # J/(mol*K)
MW_AIR = 0.028964  # kg/mol
MW_METHANE = 0.01604  # kg/mol


def pseudo_reduced(
    p: ArrayLike, t: ArrayLike, ppc: ArrayLike, tpc: ArrayLike
) -> tuple[_Float, _Float]:
    """Pseudo-reduced pressure and temperature ``(P/Ppc, T/Tpc)``.

    Sources: src2019_08/article3_shale_gas_z_factor.
    """
    ppr = np.asarray(p, np.float64) / np.asarray(ppc, np.float64)
    tpr = np.asarray(t, np.float64) / np.asarray(tpc, np.float64)
    return np.asarray(ppr), np.asarray(tpr)


def z_beggs_brill(ppr: ArrayLike, tpr: ArrayLike) -> _Float:
    """Beggs-Brill (1973) gas compressibility factor from pseudo-reduced P/T.

    Sources: src2019_08/article3_shale_gas_z_factor.
    """
    ppr_a = np.asarray(ppr, np.float64)
    tpr_a = np.asarray(tpr, np.float64)
    a = 1.39 * (tpr_a - 0.92) ** 0.5 - 0.36 * tpr_a - 0.101
    b = (
        (0.62 - 0.23 * tpr_a) * ppr_a
        + (0.066 / (tpr_a - 0.86) - 0.037) * ppr_a**2
        + 0.32 * ppr_a**6 / (10.0 ** (9.0 * (tpr_a - 1.0)))
    )
    c = 0.132 - 0.32 * np.log10(tpr_a)
    d = 10.0 ** (0.3106 - 0.49 * tpr_a + 0.1824 * tpr_a**2)
    return np.asarray(a + (1.0 - a) / np.exp(b) + c * ppr_a**d)


def z_peng_robinson(
    p_pa: float, t_k: float, tc_k: float, pc_pa: float, omega: float, *, phase: str = "vapor"
) -> float:
    """Peng-Robinson z-factor (scalar) for the vapor or liquid root.

    ``phase="vapor"`` returns the largest real root, ``"liquid"`` the smallest
    positive real root of the PR cubic.
    """
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1.0 + kappa * (1.0 - np.sqrt(t_k / tc_k))) ** 2
    a = 0.45724 * R_GAS**2 * tc_k**2 / pc_pa * alpha
    b = 0.07780 * R_GAS * tc_k / pc_pa
    a_big = a * p_pa / (R_GAS * t_k) ** 2
    b_big = b * p_pa / (R_GAS * t_k)
    coeffs = [
        1.0,
        -(1.0 - b_big),
        a_big - 3.0 * b_big**2 - 2.0 * b_big,
        -(a_big * b_big - b_big**2 - b_big**3),
    ]
    roots = np.roots(coeffs)
    real = np.real(roots[np.abs(np.imag(roots)) < 1e-9])
    real = real[real > 0]
    if real.size == 0:
        raise ValueError("no positive real PR root")
    if phase == "vapor":
        return float(np.max(real))
    if phase == "liquid":
        return float(np.min(real))
    raise ValueError(f"phase must be 'vapor' or 'liquid', got {phase!r}")


def gas_density(
    p_pa: ArrayLike, t_k: ArrayLike, *, m_kg_mol: ArrayLike = MW_METHANE, z: ArrayLike = 1.0
) -> _Float:
    """Real-gas density ``rho = P*M/(z*R*T)`` (kg/m3).

    Sources: src2019_08/article3_shale_gas_z_factor.
    """
    return np.asarray(
        np.asarray(p_pa, np.float64)
        * np.asarray(m_kg_mol, np.float64)
        / (np.asarray(z, np.float64) * R_GAS * np.asarray(t_k, np.float64))
    )


def pressure_from_gas_density(
    rho: ArrayLike, t_k: ArrayLike, *, m_kg_mol: ArrayLike = MW_METHANE, z: ArrayLike = 1.0
) -> _Float:
    """Pressure (Pa) from gas density -- inverse of ``gas_density``."""
    return np.asarray(
        np.asarray(rho, np.float64)
        * np.asarray(z, np.float64)
        * R_GAS
        * np.asarray(t_k, np.float64)
        / np.asarray(m_kg_mol, np.float64)
    )


def mixture_mw(y: ArrayLike, mw: ArrayLike, *, axis: int = -1) -> _Float:
    """Mixture molecular weight ``sum(y_i*MW_i)`` over ``axis``.

    Sources: src2017_06/article2_tgip_nmr_gas_shale.
    """
    return np.asarray(np.sum(np.asarray(y, np.float64) * np.asarray(mw, np.float64), axis=axis))


def gas_gravity(mw_kg_mol: ArrayLike) -> _Float:
    """Gas specific gravity ``MW_gas/MW_air`` (air = 0.028964 kg/mol)."""
    return np.asarray(np.asarray(mw_kg_mol, np.float64) / MW_AIR)


def wilson_k(
    p_pa: ArrayLike, t_k: ArrayLike, pc_pa: ArrayLike, tc_k: ArrayLike, omega: ArrayLike
) -> _Float:
    """Wilson (1969) K-value estimate ``(Pc/P)*exp(5.373*(1+omega)*(1 - Tc/T))``."""
    pc = np.asarray(pc_pa, np.float64)
    tc = np.asarray(tc_k, np.float64)
    return np.asarray(
        (pc / np.asarray(p_pa, np.float64))
        * np.exp(
            5.373 * (1.0 + np.asarray(omega, np.float64)) * (1.0 - tc / np.asarray(t_k, np.float64))
        )
    )


def rachford_rice(z: ArrayLike, k: ArrayLike, *, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Rachford-Rice vapor fraction ``beta`` solving ``sum(z*(k-1)/(1+beta*(k-1))) = 0``.

    Bisection on ``beta`` in [0, 1]; returns the root (clamped to the bracket).
    """
    z_a = np.asarray(z, np.float64)
    k_a = np.asarray(k, np.float64)

    def g(beta: float) -> float:
        return float(np.sum(z_a * (k_a - 1.0) / (1.0 + beta * (k_a - 1.0))))

    lo, hi = 0.0, 1.0
    g_lo, g_hi = g(lo), g(hi)
    if g_lo * g_hi > 0:
        return 0.0 if abs(g_lo) < abs(g_hi) else 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        g_mid = g(mid)
        if abs(g_mid) < tol:
            return mid
        if g_lo * g_mid < 0:
            hi = mid
        else:
            lo, g_lo = mid, g_mid
    return 0.5 * (lo + hi)
