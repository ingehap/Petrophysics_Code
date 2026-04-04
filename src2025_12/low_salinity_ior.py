"""
Low-Salinity Brine Wettability Alteration / IOR for Presalt Carbonates
=======================================================================

Implements the ideas of:

    Karoussi, O., Wat, R.M.S., De Lima, C., and Ribeiro, L., 2025,
    "Potential Wettability Alteration/IOR Study for Presalt Carbonate
    by Low-Salinity Brines: From Experiments to Field-Scale Simulation",
    Petrophysics, 66(6), 1043–1060.
    DOI: 10.30632/PJV66N6-2025a9

Key ideas
---------
* Salinity-dependent contact angle and wettability alteration.
* Modified Amott–Harvey wettability index shift with modified seawater.
* Spontaneous imbibition rate as wettability indicator.
* Simplified DLVO-like surface-charge model for carbonate surfaces.
* Recovery factor uplift from low-salinity water injection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. Salinity-Dependent Contact Angle Model
# ──────────────────────────────────────────────────────────────────────
def contact_angle_salinity(
    salinity: np.ndarray,
    theta_high: float = 130.0,
    theta_low: float = 50.0,
    S_half: float = 20000.0,
    n: float = 2.0,
) -> np.ndarray:
    """Empirical salinity-dependent contact angle for carbonates.

    θ(S) = θ_low + (θ_high − θ_low) * (S^n / (S^n + S_half^n))

    At high salinity → oil-wet (θ → θ_high).
    At low salinity  → water-wet (θ → θ_low).

    Parameters
    ----------
    salinity : array_like
        Total dissolved solids (ppm or mg/L).
    theta_high : float
        Contact angle at very high salinity (degrees).
    theta_low : float
        Contact angle at very low salinity (degrees).
    S_half : float
        Salinity at which θ is midway between extremes.
    n : float
        Steepness parameter.

    Returns
    -------
    np.ndarray
        Contact angle (degrees) at each salinity.
    """
    S = np.asarray(salinity, float)
    return theta_low + (theta_high - theta_low) * (S**n / (S**n + S_half**n))


# ──────────────────────────────────────────────────────────────────────
# 2. Spontaneous Imbibition Rate
# ──────────────────────────────────────────────────────────────────────
def imbibition_recovery_expstretched(
    t: np.ndarray,
    RF_max: float = 0.15,
    tau: float = 100.0,
    beta: float = 0.7,
) -> np.ndarray:
    """Stretched-exponential model for spontaneous imbibition recovery.

    RF(t) = RF_max * (1 − exp(-(t/τ)^β))

    Faster imbibition → more water-wet.

    Parameters
    ----------
    t : array_like
        Time (hours or days — be consistent).
    RF_max : float
        Asymptotic recovery factor (fraction OOIP).
    tau : float
        Characteristic imbibition time.
    beta : float
        Stretching exponent (0 < β ≤ 1).
    """
    t = np.asarray(t, float)
    return RF_max * (1.0 - np.exp(-(t / tau) ** beta))


def imbibition_rate(
    t: np.ndarray,
    RF_max: float = 0.15,
    tau: float = 100.0,
    beta: float = 0.7,
) -> np.ndarray:
    """Time derivative of the stretched-exponential imbibition model.

    dRF/dt = RF_max * (β/τ) * (t/τ)^(β-1) * exp(-(t/τ)^β)
    """
    t = np.asarray(t, float)
    t_safe = np.where(t > 0, t, 1e-30)
    x = (t_safe / tau) ** beta
    return RF_max * (beta / tau) * (t_safe / tau) ** (beta - 1) * np.exp(-x)


# ──────────────────────────────────────────────────────────────────────
# 3. DLVO-Inspired Disjoining-Pressure Model (simplified)
# ──────────────────────────────────────────────────────────────────────
def double_layer_repulsion(
    h: np.ndarray,
    kappa_inv: float,
    A_dl: float = 1e-3,
) -> np.ndarray:
    """Electrostatic double-layer contribution to disjoining pressure.

    Π_dl = A_dl * exp(-h / κ⁻¹)

    Reducing salinity increases κ⁻¹ (Debye length), expanding the
    double layer and promoting water-film stability on carbonate surfaces.

    Parameters
    ----------
    h : array_like
        Water-film thickness (m).
    kappa_inv : float
        Debye length (m).
    A_dl : float
        Amplitude constant (Pa).
    """
    h = np.asarray(h, float)
    return A_dl * np.exp(-h / kappa_inv)


def van_der_waals_attraction(
    h: np.ndarray,
    A_H: float = 1e-20,
) -> np.ndarray:
    """Van der Waals attraction (simplified, flat surfaces).

    Π_vdW = −A_H / (6π h³)

    Parameters
    ----------
    h : array_like
        Film thickness (m).
    A_H : float
        Hamaker constant (J).
    """
    h = np.asarray(h, float)
    return -A_H / (6.0 * math.pi * h ** 3)


def disjoining_pressure(
    h: np.ndarray,
    kappa_inv: float,
    A_dl: float = 1e-3,
    A_H: float = 1e-20,
) -> np.ndarray:
    """Net disjoining pressure (double-layer + van der Waals).

    Positive Π → stable water film → water-wet behaviour.
    """
    return double_layer_repulsion(h, kappa_inv, A_dl) + van_der_waals_attraction(h, A_H)


# ──────────────────────────────────────────────────────────────────────
# 4. Field-Scale Recovery-Factor Uplift
# ──────────────────────────────────────────────────────────────────────
def rf_uplift(rf_high_salinity: float, rf_low_salinity: float) -> float:
    """Recovery factor uplift from low-salinity water injection.

    ΔRF = RF_LS − RF_HS
    """
    return rf_low_salinity - rf_high_salinity


def incremental_oil(ooip: float, delta_rf: float) -> float:
    """Incremental oil volume from wettability alteration.

    ΔN_p = OOIP × ΔRF
    """
    return ooip * delta_rf


# ──────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Contact angle vs salinity
    S = np.logspace(2, 5, 30)  # 100 – 100 000 ppm
    theta = contact_angle_salinity(S)
    print("Salinity (ppm)   θ (°)")
    for s, th in list(zip(S, theta))[::6]:
        print(f"  {s:>10.0f}    {th:5.1f}")

    # Spontaneous imbibition comparison
    t = np.linspace(0.1, 500, 100)
    RF_hs = imbibition_recovery_expstretched(t, 0.08, 200, 0.6)
    RF_ls = imbibition_recovery_expstretched(t, 0.15, 80, 0.7)
    print(f"\nRecovery at t=500: HS = {RF_hs[-1]*100:.1f}%  "
          f"LS = {RF_ls[-1]*100:.1f}%")
    print(f"Uplift: {rf_uplift(RF_hs[-1], RF_ls[-1])*100:.1f}% OOIP")
