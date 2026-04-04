"""
Wettability Effect on Adsorption and Capillary Condensation in Nanopores
=========================================================================

Implements the ideas of:

    Nguyen, A.T.T., Sharma, K.V., and Piri, M., 2025,
    "The Effect of Wettability on the Amount of Adsorbed Fluid and
    Capillary Condensation Pressure in Nanoporous Materials",
    Petrophysics, 66(6), 1061–1071.
    DOI: 10.30632/PJV66N6-2025a10

Key ideas
---------
* Washburn capillary-rise equation for pore-size and contact-angle
  measurement (Eq. 1).
* Modified Kelvin equation for capillary condensation pressure in
  cylindrical nanopores.
* BET isotherm for multilayer adsorption.
* Wettability (hydrophilic MCM-41 vs. HMDS-treated hydrophobic)
  controls adsorbed amount and condensation onset.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. Washburn Capillary Rise  (Eq. 1)
# ──────────────────────────────────────────────────────────────────────
def washburn_capillary_rise(
    sigma: float,
    theta_deg: float,
    mu: float,
    r: float,
    t: np.ndarray,
) -> np.ndarray:
    """Washburn equation relating capillary rise to time [Eq. 1].

    h²(t) = (r σ cos θ / 2μ) t

    Parameters
    ----------
    sigma : float
        Surface tension of the liquid (N/m).
    theta_deg : float
        Contact angle (degrees).
    mu : float
        Dynamic viscosity of the liquid (Pa·s).
    r : float
        Pore radius (m).
    t : array_like
        Time (s).

    Returns
    -------
    np.ndarray
        Height of capillary rise h(t) (m).
    """
    theta = math.radians(theta_deg)
    t = np.asarray(t, float)
    coeff = r * sigma * math.cos(theta) / (2.0 * mu)
    h_sq = coeff * t
    return np.sqrt(np.maximum(h_sq, 0.0))


def washburn_effective_radius(
    h: float,
    t: float,
    sigma: float,
    theta_deg: float,
    mu: float,
) -> float:
    """Invert the Washburn equation to estimate effective pore radius.

    r = 2 μ h² / (σ cos θ · t)
    """
    theta = math.radians(theta_deg)
    cos_theta = math.cos(theta)
    if cos_theta == 0 or t == 0:
        return float("inf")
    return 2.0 * mu * h ** 2 / (sigma * cos_theta * t)


# ──────────────────────────────────────────────────────────────────────
# 2. Modified Kelvin Equation for Capillary Condensation
# ──────────────────────────────────────────────────────────────────────
def kelvin_condensation_pressure(
    r_pore: float,
    sigma: float,
    V_m: float,
    theta_deg: float,
    T: float,
    t_film: float = 0.0,
    R_gas: float = 8.314,
) -> float:
    """Relative pressure P/P₀ at which capillary condensation occurs.

    ln(P/P₀) = − (2 σ V_m cos θ) / (R T (r − t_film))

    Parameters
    ----------
    r_pore : float
        Pore radius (m).
    sigma : float
        Surface tension of condensate (N/m).
    V_m : float
        Molar volume of condensate (m³/mol).
    theta_deg : float
        Contact angle (degrees).
    T : float
        Temperature (K).
    t_film : float
        Pre-existing adsorbed film thickness (m).
    R_gas : float
        Gas constant (J/(mol·K)).

    Returns
    -------
    float
        Relative pressure P/P₀ at condensation onset.
    """
    theta = math.radians(theta_deg)
    r_eff = r_pore - t_film
    if r_eff <= 0:
        return 1.0
    ln_P = -2.0 * sigma * V_m * math.cos(theta) / (R_gas * T * r_eff)
    return math.exp(ln_P)


# ──────────────────────────────────────────────────────────────────────
# 3. BET Multilayer Adsorption Isotherm
# ──────────────────────────────────────────────────────────────────────
def bet_isotherm(
    P_rel: np.ndarray,
    V_m_mono: float,
    C: float,
) -> np.ndarray:
    """BET adsorption isotherm.

    V = V_m * C * x / ((1 − x)(1 − x + C x))

    where x = P / P₀.

    Parameters
    ----------
    P_rel : array_like
        Relative pressure P/P₀ (0 < x < 1).
    V_m_mono : float
        Monolayer adsorption capacity (same units as V output).
    C : float
        BET constant related to heat of adsorption.

    Returns
    -------
    np.ndarray
        Adsorbed volume at each relative pressure.
    """
    x = np.asarray(P_rel, float)
    return V_m_mono * C * x / ((1.0 - x) * (1.0 - x + C * x))


def bet_surface_area(
    V_m_mono: float,
    sigma_molecule: float = 1.62e-19,
    V_molar_STP: float = 22414e-6,
) -> float:
    """BET specific surface area.

    S = V_m * N_A * σ_m / V_molar

    Parameters
    ----------
    V_m_mono : float
        Monolayer volume (cm³ STP / g).
    sigma_molecule : float
        Cross-sectional area of adsorbate molecule (m²).
        Default: N₂ = 0.162 nm² = 1.62 × 10⁻¹⁹ m².
    V_molar_STP : float
        Molar volume at STP (m³/mol).  Default: 22414 cm³ = 22.414 L.

    Returns
    -------
    float
        Specific surface area (m²/g).
    """
    N_A = 6.022e23
    return V_m_mono * 1e-6 * N_A * sigma_molecule / V_molar_STP


# ──────────────────────────────────────────────────────────────────────
# 4. Wettability Effect on Adsorption Amount
# ──────────────────────────────────────────────────────────────────────
def adsorption_amount_ratio(
    V_hydrophilic: float,
    V_hydrophobic: float,
) -> float:
    """Ratio of adsorbed amount on hydrophilic vs hydrophobic surfaces.

    Nguyen et al. showed that hydrophilic MCM-41 adsorbs more ethane
    than hydrophobic (HMDS-treated) MCM-41, especially in smaller pores.

    A ratio > 1 indicates stronger adsorption on the hydrophilic surface.
    """
    if V_hydrophobic <= 0:
        return float("inf")
    return V_hydrophilic / V_hydrophobic


# ──────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Washburn capillary rise for ethane in MCM-41 (r ~ 2 nm)
    t = np.linspace(0.01, 10.0, 50)  # seconds
    h = washburn_capillary_rise(sigma=0.016, theta_deg=0, mu=1e-4,
                                r=2e-9, t=t)
    print(f"Capillary rise at t=10 s: {h[-1]*1e6:.3f} µm")

    # Kelvin condensation pressure
    for r_nm in [2.0, 3.5]:
        P_cond = kelvin_condensation_pressure(
            r_pore=r_nm * 1e-9, sigma=0.016, V_m=5.2e-5,
            theta_deg=0, T=300,
        )
        print(f"P/P₀ condensation (r={r_nm} nm, hydrophilic): {P_cond:.4f}")
        P_cond_h = kelvin_condensation_pressure(
            r_pore=r_nm * 1e-9, sigma=0.016, V_m=5.2e-5,
            theta_deg=90, T=300,
        )
        print(f"P/P₀ condensation (r={r_nm} nm, hydrophobic): {P_cond_h:.4f}")

    # BET isotherm
    x = np.linspace(0.01, 0.35, 30)
    V = bet_isotherm(x, V_m_mono=50.0, C=80.0)
    print(f"\nBET adsorbed volume at P/P₀=0.3: {V[-1]:.1f} cm³/g STP")
