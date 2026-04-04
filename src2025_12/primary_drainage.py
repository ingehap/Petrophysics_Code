"""
A Review of Primary Drainage Techniques
========================================

Implements the key models and calculations from:

    Fernandes, V., Nicot, B., Pairoys, F., Nono, F., Bertin, H.,
    Lachaud, J., and Caubit, C., 2025,
    "A Review of Primary Drainage Techniques",
    Petrophysics, 66(6), 957–968.
    DOI: 10.30632/PJV66N6-2025a3

Key ideas
---------
* Centrifuge capillary-pressure calculation (Hassler–Brunner, Eq. 1).
* Forbes (1994) saturation-averaging correction.
* Porous-plate equilibrium Pc model.
* Semi-dynamic (viscous-oil-flood) capillary pressure from local
  pressure balance (Lenormand et al., 1993).
* Archie saturation exponent from resistivity index.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────
# 1. Centrifuge Capillary Pressure  (Eq. 1, Hassler & Brunner 1945)
# ──────────────────────────────────────────────────────────────────────
def centrifuge_capillary_pressure(
    delta_rho: float,
    omega: float,
    r: np.ndarray,
    R: float,
) -> np.ndarray:
    """Capillary pressure at radius *r* inside a centrifuge [Eq. 1].

    Pc(r) = 0.5 * Δρ * ω² * (R² − r²)

    Parameters
    ----------
    delta_rho : float
        Density difference between phases (kg/m³).
    omega : float
        Rotational speed (rad/s).
    r : array_like
        Radial positions along the core (m).
    R : float
        Distance from centrifuge axis to sample outlet (m).

    Returns
    -------
    np.ndarray
        Capillary pressure (Pa) at each position.
    """
    r = np.asarray(r, float)
    return 0.5 * delta_rho * omega ** 2 * (R ** 2 - r ** 2)


def rpm_to_omega(rpm: float) -> float:
    """Convert revolutions per minute to rad/s."""
    return rpm * 2.0 * np.pi / 60.0


def centrifuge_Pc_at_inlet(
    delta_rho: float,
    omega: float,
    r_inlet: float,
    R: float,
) -> float:
    """Maximum Pc at the core inlet (face closest to the axis).

    This is the capillary pressure that drives primary drainage.
    """
    return float(centrifuge_capillary_pressure(delta_rho, omega,
                                               np.array([r_inlet]), R)[0])


# ──────────────────────────────────────────────────────────────────────
# 2. Forbes (1994) Saturation Averaging Correction
# ──────────────────────────────────────────────────────────────────────
def average_saturation_from_profile(
    Sw_profile: np.ndarray,
    r_positions: np.ndarray,
) -> float:
    """Compute mean saturation from a 1-D saturation profile.

    Uses trapezoidal integration along the core length.
    """
    return float(np.trapz(Sw_profile, r_positions)
                 / (r_positions[-1] - r_positions[0]))


def hassler_brunner_correction(
    Sw_avg: np.ndarray,
    Pc_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """First-order Hassler–Brunner correction for centrifuge data.

    Sw_local(Pc_max) ≈ d(Sw_avg * Pc_max) / d(Pc_max)

    Uses numerical differentiation.

    Parameters
    ----------
    Sw_avg : array_like
        Average saturations at each speed step.
    Pc_values : array_like
        Maximum capillary pressures at each speed step.

    Returns
    -------
    Sw_corrected, Pc : arrays
    """
    Sw_avg = np.asarray(Sw_avg, float)
    Pc = np.asarray(Pc_values, float)
    product = Sw_avg * Pc
    # Numerical derivative d(Sw*Pc)/dPc
    dproduct = np.gradient(product, Pc)
    Sw_corrected = dproduct
    return Sw_corrected, Pc


# ──────────────────────────────────────────────────────────────────────
# 3. Porous-Plate Equilibrium
# ──────────────────────────────────────────────────────────────────────
def porous_plate_Pc(
    pressure_steps: np.ndarray,
    Sw_at_steps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Pc–Sw curve from porous-plate measurements.

    Under porous-plate conditions, Pc equals the imposed oil injection
    pressure (assuming Pw at outlet equals zero and the semipermeable
    membrane maintains water-phase continuity).

    This is a direct determination — no inversion needed.
    """
    return np.asarray(pressure_steps, float), np.asarray(Sw_at_steps, float)


def porous_plate_equilibrium_time(
    k_membrane: float,
    L_membrane: float,
    mu_w: float,
    phi: float,
    L_core: float,
    A: float,
) -> float:
    """Estimate equilibrium time for a porous-plate step (order of magnitude).

    τ ≈ (φ * L_core * A * μ_w * L_membrane) / (k_membrane * A)

    Parameters
    ----------
    k_membrane : float
        Membrane permeability (m²).
    L_membrane : float
        Membrane thickness (m).
    mu_w : float
        Water viscosity (Pa·s).
    phi : float
        Core porosity (fraction).
    L_core : float
        Core length (m).
    A : float
        Cross-sectional area (m²).

    Returns
    -------
    float
        Characteristic equilibrium time (s).
    """
    return (phi * L_core * mu_w * L_membrane) / k_membrane


# ──────────────────────────────────────────────────────────────────────
# 4. Semi-Dynamic (Viscous Oil Flood) Capillary Pressure
# ──────────────────────────────────────────────────────────────────────
def semi_dynamic_Pc(
    P_oil: np.ndarray,
    P_water: float,
) -> np.ndarray:
    """Local capillary pressure from semi-dynamic measurement.

    Pc = Po − Pw   (Lenormand et al., 1993)

    Parameters
    ----------
    P_oil : array_like
        Oil pressure measured at discrete points along the core (Pa).
    P_water : float
        Water pressure (constant via outlet washing) (Pa).

    Returns
    -------
    np.ndarray
        Local capillary pressure at each measurement point.
    """
    return np.asarray(P_oil, float) - P_water


# ──────────────────────────────────────────────────────────────────────
# 5. Archie Saturation Exponent from Resistivity Index
# ──────────────────────────────────────────────────────────────────────
def resistivity_index(Rt: np.ndarray, R0: float) -> np.ndarray:
    """Resistivity Index.

    RI = Rt / R0

    Parameters
    ----------
    Rt : array_like
        True resistivity at saturation Sw.
    R0 : float
        Resistivity at 100 % water saturation.
    """
    return np.asarray(Rt, float) / R0


def archie_n_exponent(
    Sw: np.ndarray,
    RI: np.ndarray,
) -> float:
    """Determine Archie saturation exponent *n* from RI vs Sw data.

    RI = Sw^(-n)  →  log(RI) = -n * log(Sw)

    Uses least-squares fit in log–log space.

    Parameters
    ----------
    Sw : array_like
        Water saturation (fraction).
    RI : array_like
        Resistivity index (dimensionless).

    Returns
    -------
    float
        Best-fit Archie exponent n.
    """
    Sw = np.asarray(Sw, float)
    RI = np.asarray(RI, float)
    mask = (Sw > 0) & (RI > 0)
    log_Sw = np.log10(Sw[mask])
    log_RI = np.log10(RI[mask])
    # Linear regression: log_RI = -n * log_Sw  (forced through origin)
    n = -float(np.dot(log_Sw, log_RI) / np.dot(log_Sw, log_Sw))
    return n


# ──────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Centrifuge example
    delta_rho = 200.0        # kg/m³
    rpm = 3000
    omega = rpm_to_omega(rpm)
    R = 0.15                 # m (outlet radius)
    r_inlet = 0.12           # m
    r = np.linspace(r_inlet, R, 50)

    Pc = centrifuge_capillary_pressure(delta_rho, omega, r, R)
    print(f"Pc range: {Pc.min()/1e3:.1f} – {Pc.max()/1e3:.1f} kPa")

    # Archie n from synthetic data
    Sw = np.array([1.0, 0.8, 0.6, 0.4, 0.3, 0.2])
    n_true = 2.0
    RI_syn = Sw ** (-n_true) * (1 + 0.02 * np.random.randn(len(Sw)))
    n_fit = archie_n_exponent(Sw, RI_syn)
    print(f"Archie n (true={n_true:.1f}): fitted n = {n_fit:.3f}")
