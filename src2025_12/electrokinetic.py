"""
Electrokinetic Rock Properties and Dopant-Induced Wettability Changes
=====================================================================

Implements the ideas of:

    Halisch, M., Pairoys, F., Caubit, C., and Grelle, T., 2025,
    "Assessing the Impact of Dopants on Electrokinetic Rock Properties
    as Potential Indicators for Dopant-Induced Wettability Changes",
    Petrophysics, 66(6), 1013–1031.
    DOI: 10.30632/PJV66N6-2025a7

Key ideas
---------
* Zeta-potential from streaming-potential measurements using the
  Helmholtz–Smoluchowski equation.
* Electroacoustic measurement of zeta potential in colloidal/pore
  systems.
* Assessment of how ionic dopants (NaI, KI, BaCl₂) shift zeta potential
  and may bias wettability interpretations in SCAL workflows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# Physical constants
EPSILON_0 = 8.854e-12       # Vacuum permittivity (F/m)
EPSILON_R_WATER = 78.5      # Relative permittivity of water at 25 °C
VISCOSITY_WATER = 1.002e-3  # Dynamic viscosity of water at 20 °C (Pa·s)
BOLTZMANN = 1.381e-23       # Boltzmann constant (J/K)
ELEM_CHARGE = 1.602e-19     # Elementary charge (C)
AVOGADRO = 6.022e23         # Avogadro's number


# ──────────────────────────────────────────────────────────────────────
# 1. Helmholtz–Smoluchowski Equation
# ──────────────────────────────────────────────────────────────────────
def zeta_potential_HS(
    delta_V: float,
    delta_P: float,
    mu: float = VISCOSITY_WATER,
    epsilon_r: float = EPSILON_R_WATER,
    kappa_L: float = 1.0,
) -> float:
    """Zeta potential from streaming-potential measurement.

    ζ = (ΔV / ΔP) * (μ / ε₀ ε_r) * κ_L

    Parameters
    ----------
    delta_V : float
        Measured streaming potential (V).
    delta_P : float
        Applied pressure differential (Pa).
    mu : float
        Fluid viscosity (Pa·s).
    epsilon_r : float
        Relative permittivity of the fluid.
    kappa_L : float
        Conductivity correction factor (ratio of bulk to surface
        conductivity), default 1.0 (no surface conductivity correction).

    Returns
    -------
    float
        Zeta potential (V).
    """
    epsilon = EPSILON_0 * epsilon_r
    return (delta_V / delta_P) * (mu / epsilon) * kappa_L


def zeta_potential_mV(delta_V: float, delta_P: float, **kw) -> float:
    """Zeta potential in millivolts."""
    return zeta_potential_HS(delta_V, delta_P, **kw) * 1e3


# ──────────────────────────────────────────────────────────────────────
# 2. Streaming-Potential Coefficient
# ──────────────────────────────────────────────────────────────────────
def streaming_potential_coefficient(
    delta_V_values: np.ndarray,
    delta_P_values: np.ndarray,
) -> float:
    """Determine the streaming-potential coupling coefficient C_sp.

    C_sp = ΔV / ΔP  (slope of the linear V–P relationship).

    Uses least-squares linear regression.

    Parameters
    ----------
    delta_V_values : array_like
        Measured streaming potentials (V).
    delta_P_values : array_like
        Applied pressure differences (Pa).

    Returns
    -------
    float
        C_sp (V/Pa).
    """
    dV = np.asarray(delta_V_values, float)
    dP = np.asarray(delta_P_values, float)
    # Force through origin: C_sp = dot(dP, dV) / dot(dP, dP)
    return float(np.dot(dP, dV) / np.dot(dP, dP))


# ──────────────────────────────────────────────────────────────────────
# 3. Debye Length and Double-Layer Thickness
# ──────────────────────────────────────────────────────────────────────
def debye_length(
    ionic_strength: float,
    T: float = 298.15,
    epsilon_r: float = EPSILON_R_WATER,
) -> float:
    """Debye screening length.

    κ⁻¹ = sqrt( ε₀ ε_r k_B T / (2 N_A e² I) )

    Parameters
    ----------
    ionic_strength : float
        Ionic strength (mol/m³).
    T : float
        Temperature (K).
    epsilon_r : float
        Relative permittivity.

    Returns
    -------
    float
        Debye length (m).
    """
    numerator = EPSILON_0 * epsilon_r * BOLTZMANN * T
    denominator = 2.0 * AVOGADRO * ELEM_CHARGE ** 2 * ionic_strength
    return math.sqrt(numerator / denominator)


def ionic_strength_from_molarity(concentrations: dict[str, float],
                                  valences: dict[str, int]) -> float:
    """Ionic strength I = 0.5 Σ cᵢ zᵢ².

    Parameters
    ----------
    concentrations : dict
        Ion name → molar concentration (mol/L).
    valences : dict
        Ion name → charge number.

    Returns
    -------
    float
        Ionic strength (mol/L).
    """
    return 0.5 * sum(c * valences[ion] ** 2
                     for ion, c in concentrations.items())


# ──────────────────────────────────────────────────────────────────────
# 4. Dopant Effect Assessment
# ──────────────────────────────────────────────────────────────────────
@dataclass
class DopantTest:
    """Record of a streaming-potential test with a specific dopant.

    Attributes
    ----------
    brine_name : str
        E.g. "Formation brine", "NaI 10 wt%"
    dopant : str | None
        Active dopant, e.g. "NaI", "KI", "BaCl2"
    zeta_mV : float
        Measured zeta potential (mV).
    """
    brine_name: str
    dopant: Optional[str]
    zeta_mV: float


def zeta_shift(baseline: DopantTest, doped: DopantTest) -> float:
    """Zeta-potential shift caused by a dopant (mV).

    Δζ = ζ_doped − ζ_baseline

    A large shift suggests the dopant significantly alters the
    electrical double layer and may bias wettability measurements.
    """
    return doped.zeta_mV - baseline.zeta_mV


def flag_wettability_risk(
    shifts: list[float],
    threshold_mV: float = 10.0,
) -> list[bool]:
    """Flag dopants whose zeta-shift exceeds a threshold.

    Parameters
    ----------
    shifts : list of float
        Δζ for each dopant (mV).
    threshold_mV : float
        Threshold for significant wettability-alteration risk.

    Returns
    -------
    list of bool
        True if |Δζ| > threshold.
    """
    return [abs(s) > threshold_mV for s in shifts]


# ──────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Streaming-potential example
    dP = np.array([1e5, 2e5, 3e5, 4e5])    # Pa
    dV = np.array([-0.5e-3, -1.1e-3, -1.6e-3, -2.0e-3])  # V

    C_sp = streaming_potential_coefficient(dV, dP)
    zeta = zeta_potential_mV(C_sp * 1e5, 1e5)
    print(f"C_sp = {C_sp:.3e} V/Pa")
    print(f"Zeta potential ≈ {zeta:.1f} mV")

    # Debye length for 0.1 M NaCl
    I_nacl = ionic_strength_from_molarity(
        {"Na+": 0.1, "Cl-": 0.1},
        {"Na+": 1, "Cl-": 1},
    )
    lam_D = debye_length(I_nacl * 1e3)  # convert mol/L → mol/m³
    print(f"Debye length (0.1 M NaCl): {lam_D*1e9:.2f} nm")

    # Dopant assessment
    baseline = DopantTest("Formation brine", None, -25.0)
    nai = DopantTest("NaI 10 wt%", "NaI", -12.0)
    ki = DopantTest("KI 10 wt%", "KI", -8.0)
    bacl2 = DopantTest("BaCl2 5 wt%", "BaCl2", +5.0)

    for d in [nai, ki, bacl2]:
        shift = zeta_shift(baseline, d)
        risk = abs(shift) > 10
        print(f"{d.dopant:6s}: Δζ = {shift:+.1f} mV  "
              f"{'⚠ RISK' if risk else '  OK'}")
