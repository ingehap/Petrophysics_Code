"""
CO₂ Uptake Capacity Measurement in Source-Rock Shales for GCS
=============================================================

Implements the ideas of:

    Chen, J.-H., Cairns, A.J., Althaus, S.M., and Broyles, J.D., 2025,
    "Measurement of CO₂ Uptake Capacity in Source Rock Shales for GCS",
    Petrophysics, 66(6), 982–994.
    DOI: 10.30632/PJV66N6-2025a5

Key ideas
---------
* Traditional volumetric/gravimetric adsorption methods fail for
  fluid-bearing source rocks because they require clean, dry samples.
* High-field ¹³C NMR spectroscopy measures CO₂ uptake directly in
  preserved shale cores with in-situ fluids.
* Pore surface-area estimation for spherical-pore matrix models
  (Appendix A1).
* CO₂ storage capacity estimation from measured sorption data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. Pore Surface Area in a Uniform Spherical-Pore Model (App. A1)
# ──────────────────────────────────────────────────────────────────────
def pore_surface_area_spherical(
    phi: float,
    r: float,
    V_total: float = 1.0,
) -> float:
    """Total pore surface area assuming uniform spherical pores [Eq. A1.1-A1.2].

    φ = n * (4/3) π r³ / V_total    →    n = φ V_total / ((4/3)π r³)
    S_total = n * 4 π r²

    Parameters
    ----------
    phi : float
        Porosity (fraction).
    r : float
        Pore radius (m).
    V_total : float
        Total rock volume (m³).

    Returns
    -------
    float
        Total pore surface area (m²).
    """
    n_pores = phi * V_total / ((4.0 / 3.0) * math.pi * r ** 3)
    return n_pores * 4.0 * math.pi * r ** 2


def specific_surface_area_spherical(phi: float, r: float) -> float:
    """Specific surface area (m²/m³) for uniform spherical pores.

    S_sp = 3 φ / r
    """
    return 3.0 * phi / r


# ──────────────────────────────────────────────────────────────────────
# 2. NMR-Based CO₂ Sorption Quantification
# ──────────────────────────────────────────────────────────────────────
@dataclass
class NMR_CO2_Measurement:
    """Container for a ¹³C NMR CO₂ uptake measurement on a shale sample.

    Attributes
    ----------
    sample_id : str
    pressure_psi : float
        CO₂ exposure pressure (psi).
    temperature_C : float
        Temperature (°C).
    signal_intensity : float
        Integrated ¹³C NMR signal intensity for CO₂ (a.u.).
    reference_signal : float
        ¹³C NMR signal intensity of the reference standard.
    reference_moles : float
        Known moles of carbon in the reference standard.
    sample_mass_g : float
        Mass of the shale sample (g).
    """
    sample_id: str
    pressure_psi: float
    temperature_C: float
    signal_intensity: float
    reference_signal: float
    reference_moles: float
    sample_mass_g: float


def co2_sorption_moles(meas: NMR_CO2_Measurement) -> float:
    """Calculate absolute CO₂ sorption (moles) from NMR signal.

    n_CO2 = (signal_sample / signal_ref) * n_ref

    The NMR method requires no a-priori equation of state and works
    with preserved (fluid-bearing) samples.
    """
    if meas.reference_signal <= 0:
        raise ValueError("Reference signal must be positive.")
    return (meas.signal_intensity / meas.reference_signal) * meas.reference_moles


def co2_sorption_per_gram(meas: NMR_CO2_Measurement) -> float:
    """CO₂ sorption normalised by sample mass (mol/g)."""
    return co2_sorption_moles(meas) / meas.sample_mass_g


def co2_mass_from_moles(n_moles: float) -> float:
    """Convert moles CO₂ to mass (g).   M(CO₂) = 44.01 g/mol."""
    return n_moles * 44.01


# ──────────────────────────────────────────────────────────────────────
# 3. Reservoir-Scale CO₂ Storage Capacity Estimation
# ──────────────────────────────────────────────────────────────────────
def co2_storage_capacity(
    sorption_per_gram: float,
    rock_density: float,
    net_volume: float,
) -> float:
    """Estimated total CO₂ storage capacity for a reservoir unit.

    Parameters
    ----------
    sorption_per_gram : float
        CO₂ uptake per gram of rock (mol/g), from NMR measurement.
    rock_density : float
        Bulk rock density (g/m³ or g/cc — ensure unit consistency).
    net_volume : float
        Net rock volume of the storage target (m³ or cc).

    Returns
    -------
    float
        Total CO₂ that can be stored (moles).
    """
    return sorption_per_gram * rock_density * net_volume


def co2_storage_capacity_tonnes(
    sorption_per_gram: float,
    rock_density_kg_m3: float,
    net_volume_m3: float,
) -> float:
    """CO₂ storage capacity in metric tonnes.

    Parameters
    ----------
    sorption_per_gram : float
        mol CO₂ / g rock.
    rock_density_kg_m3 : float
        Bulk density (kg/m³).
    net_volume_m3 : float
        Net rock volume (m³).

    Returns
    -------
    float
        CO₂ storage capacity (tonnes).
    """
    mass_rock_g = rock_density_kg_m3 * 1e3 * net_volume_m3  # g
    n_moles = sorption_per_gram * mass_rock_g
    mass_co2_g = co2_mass_from_moles(n_moles)
    return mass_co2_g / 1e6  # tonnes


# ──────────────────────────────────────────────────────────────────────
# 4. Langmuir Adsorption Isotherm (comparison baseline)
# ──────────────────────────────────────────────────────────────────────
def langmuir_isotherm(
    P: np.ndarray,
    V_L: float,
    P_L: float,
) -> np.ndarray:
    """Langmuir adsorption isotherm.

    V = V_L * P / (P + P_L)

    Parameters
    ----------
    P : array_like
        Pressure (any consistent unit).
    V_L : float
        Langmuir volume (maximum adsorption capacity).
    P_L : float
        Langmuir pressure (pressure at half-maximum adsorption).
    """
    P = np.asarray(P, float)
    return V_L * P / (P + P_L)


# ──────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pore surface area for 5 nm pore radius, 10% porosity
    Ssp = specific_surface_area_spherical(0.10, 5e-9)
    print(f"Specific surface area (r=5 nm, φ=10%): {Ssp:.2e} m²/m³")

    # NMR CO₂ measurement example
    meas = NMR_CO2_Measurement(
        sample_id="Sample-A",
        pressure_psi=4000,
        temperature_C=60,
        signal_intensity=1250.0,
        reference_signal=5000.0,
        reference_moles=0.010,
        sample_mass_g=25.0,
    )
    n = co2_sorption_moles(meas)
    print(f"CO₂ sorption: {n:.4f} mol  ({co2_mass_from_moles(n):.2f} g)")
    print(f"Per gram: {co2_sorption_per_gram(meas):.5f} mol/g")

    # Reservoir capacity
    cap = co2_storage_capacity_tonnes(
        sorption_per_gram=co2_sorption_per_gram(meas),
        rock_density_kg_m3=2500,
        net_volume_m3=1e9,  # 1 km³
    )
    print(f"Reservoir CO₂ capacity: {cap:.1f} Mt")
