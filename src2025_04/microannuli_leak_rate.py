"""
Advanced Ultrasonic Log Analysis and Mechanistic Modeling for Leak Rate
Quantification Through Microannuli
========================================================================
Based on: Machicote et al., "The Road Through Microannuli: Advanced
Ultrasonic Log Analysis and Mechanistic Modeling for Leak Rate
Quantification",
Petrophysics, Vol. 66, No. 2, April 2025, pp. 331–347.

Implements:
  - Acoustic impedance to microannulus thickness mapping (Eq. 1)
  - Hagen-Poiseuille flow model in cylindrical coordinates (Eqs. 2–11)
  - Liquid leak rate with gravitational correction (Eq. 10)
  - Gas leak rate under isothermal conditions (Eq. 11)
  - Bond index computation from CBL data
  - Sensitivity analysis framework

Reference: https://doi.org/10.30632/PJV66N2-2025a10 (SPWLA)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class FluidType(Enum):
    """Type of leaking fluid."""
    FRESH_WATER = "fresh_water"
    BRINE = "brine"
    OIL = "oil"
    METHANE = "methane"
    CO2_GAS = "co2_gas"
    CO2_SUPERCRITICAL = "co2_supercritical"


@dataclass
class FluidProperties:
    """Physical properties of leaking fluid."""
    density_kg_m3: float
    viscosity_Pa_s: float
    is_gas: bool = False

    @classmethod
    def from_type(cls, fluid: FluidType) -> "FluidProperties":
        """Create FluidProperties from a standard fluid type."""
        props = {
            FluidType.FRESH_WATER: cls(1000.0, 1.0e-3, False),
            FluidType.BRINE: cls(1100.0, 1.2e-3, False),
            FluidType.OIL: cls(850.0, 5.0e-3, False),
            FluidType.METHANE: cls(0.66, 1.1e-5, True),
            FluidType.CO2_GAS: cls(1.87, 1.5e-5, True),
            FluidType.CO2_SUPERCRITICAL: cls(700.0, 5.0e-5, False),
        }
        return props[fluid]


@dataclass
class WellCompletion:
    """Well completion and casing information."""
    casing_od_m: float = 0.2445     # 9-5/8 in
    casing_id_m: float = 0.2205     # Inner diameter
    casing_thickness_m: float = 0.012
    cement_ai_MRayl: float = 5.0     # Nominal cement acoustic impedance

    @property
    def casing_radius_m(self) -> float:
        return self.casing_od_m / 2.0


def microannulus_thickness_from_ai(ai_measured: float,
                                   ai_cement: float,
                                   ai_liquid: float,
                                   casing_thickness_m: float) -> float:
    """
    Estimate microannulus thickness from acoustic impedance (Eq. 1).

    tau = Psi(AI, ac, al, ct)

    The model function fits experimental data relating ultrasonic
    acoustic impedance to microannulus aperture for various cement
    types, casing sizes, and fluid fills (Machicote et al., 2025).

    Parameters
    ----------
    ai_measured : float     Measured acoustic impedance (MRayl)
    ai_cement : float       Nominal cement acoustic impedance (MRayl)
    ai_liquid : float       Liquid in microannulus acoustic impedance (MRayl)
    casing_thickness_m : float

    Returns
    -------
    float : Microannulus thickness in metres (0 if no microannulus).
    """
    if ai_measured >= ai_cement:
        return 0.0  # Good bond, no microannulus

    # Sigmoidal model fitted to experimental data
    # AI drops from ai_cement to ai_liquid as thickness increases
    # Thicker casings are less sensitive (lower frequency resonance)
    freq_factor = 0.012 / max(casing_thickness_m, 0.005)

    if ai_measured <= ai_liquid:
        # Full debonding: return maximum detectable thickness
        return 400e-6  # 400 µm limit of prediction capability

    # Inverse sigmoid mapping
    ai_normalized = (ai_measured - ai_liquid) / (ai_cement - ai_liquid)
    ai_normalized = np.clip(ai_normalized, 0.01, 0.99)

    # Empirical inverse mapping
    tau = -100e-6 * freq_factor * np.log(ai_normalized / (1.0 - ai_normalized))
    return np.clip(tau, 0.0, 400e-6)


def geometric_mean_thickness(thickness_map: np.ndarray) -> float:
    """
    Compute geometric mean of microannulus thickness along its extent.

    The geometric mean mimics the tortuous path of leaking fluid,
    analogous to permeability averaging in heterogeneous porous media
    (Machicote et al., 2025).

    Parameters
    ----------
    thickness_map : np.ndarray
        Microannulus thickness at each depth point (m).

    Returns
    -------
    float : Geometric mean thickness (m).
    """
    positive = thickness_map[thickness_map > 0]
    if len(positive) == 0:
        return 0.0
    return np.exp(np.mean(np.log(positive)))


def omega_function(R_csg: float, tau: float) -> float:
    """
    Compute the Ω function for Hagen-Poiseuille flow in annular geometry (Eq. 9).

    Ω depends on casing radius and microannulus thickness.

    Parameters
    ----------
    R_csg : float  Casing outer radius (m)
    tau : float    Microannulus thickness (m)

    Returns
    -------
    float : Ω value (m⁴)
    """
    R1 = R_csg
    R2 = R_csg + tau

    if tau <= 0:
        return 0.0

    # Annular flow cross section integral
    # Q = (π * ΔP) / (8μL) * Ω
    # Ω = R2⁴ - R1⁴ - (R2² - R1²)² / ln(R2/R1)
    ln_ratio = np.log(R2 / R1)
    if ln_ratio < 1e-12:
        return 0.0

    omega = R2 ** 4 - R1 ** 4 - (R2 ** 2 - R1 ** 2) ** 2 / ln_ratio
    return omega


def liquid_leak_rate(tau: float,
                     R_csg: float,
                     delta_P: float,
                     L: float,
                     fluid: FluidProperties,
                     theta_deg: float = 90.0) -> float:
    """
    Compute liquid leak rate through microannulus using Hagen-Poiseuille (Eq. 10).

    Q = (π / 8μL) * Ω * (ΔP - ρgL·cos(θ))

    Includes gravitational correction for upward flow.

    Parameters
    ----------
    tau : float         Microannulus thickness (m)
    R_csg : float       Casing outer radius (m)
    delta_P : float     Pressure drop P1 - P2 (Pa)
    L : float           Microannulus length (m)
    fluid : FluidProperties
    theta_deg : float   Well inclination from horizontal (90=vertical)

    Returns
    -------
    float : Flow rate Q (m³/s). Positive = upward flow.
    """
    if tau <= 0 or L <= 0:
        return 0.0

    g = 9.81
    theta = np.radians(theta_deg)

    # Gravitational correction (reduces upward flow)
    gravity_dp = fluid.density_kg_m3 * g * L * np.cos(theta)
    effective_dp = delta_P - gravity_dp

    if effective_dp <= 0:
        return 0.0

    omega = omega_function(R_csg, tau)
    Q = (np.pi / (8.0 * fluid.viscosity_Pa_s * L)) * omega * effective_dp
    return Q


def gas_leak_rate(tau: float,
                  R_csg: float,
                  P1: float,
                  P2: float,
                  L: float,
                  fluid: FluidProperties) -> float:
    """
    Compute gas leak rate through microannulus under isothermal conditions (Eq. 11).

    Approximate generalization assuming minor compressibility:
    Q = (π / 16μLP2) * Ω * (P1² - P2²)

    Gravitational effects are neglected for gas phases.

    Parameters
    ----------
    tau : float     Microannulus thickness (m)
    R_csg : float   Casing outer radius (m)
    P1 : float      Inlet pressure (Pa)
    P2 : float      Outlet pressure (Pa)
    L : float       Microannulus length (m)
    fluid : FluidProperties

    Returns
    -------
    float : Flow rate Q at outlet conditions (m³/s).
    """
    if tau <= 0 or L <= 0 or P2 <= 0:
        return 0.0

    omega = omega_function(R_csg, tau)
    Q = (np.pi / (16.0 * fluid.viscosity_Pa_s * L * P2)) * \
        omega * (P1 ** 2 - P2 ** 2)
    return max(Q, 0.0)


def bond_index(cbl_amplitude_mV: float,
               cbl_free_pipe_mV: float,
               cbl_well_bonded_mV: float) -> float:
    """
    Compute bond index (BI) from cement bond log data.

    BI = (attenuation at zone) / (maximum attenuation)
       = log(free_pipe / measured) / log(free_pipe / well_bonded)

    BI = 1 indicates perfect bond; BI < 1 indicates poor bond.

    Parameters
    ----------
    cbl_amplitude_mV : float  Measured CBL amplitude
    cbl_free_pipe_mV : float  Free pipe amplitude (maximum)
    cbl_well_bonded_mV : float  Well-bonded amplitude (minimum)

    Returns
    -------
    float : Bond index (0 to 1).
    """
    if cbl_free_pipe_mV <= cbl_well_bonded_mV:
        return 1.0

    num = np.log(cbl_free_pipe_mV / max(cbl_amplitude_mV, 0.01))
    den = np.log(cbl_free_pipe_mV / max(cbl_well_bonded_mV, 0.01))

    if den <= 0:
        return 0.0

    return np.clip(num / den, 0.0, 1.0)


def sensitivity_analysis(tau_range_um: np.ndarray,
                         R_csg: float,
                         L: float,
                         delta_P_Pa: float,
                         fluids: dict) -> dict:
    """
    Perform sensitivity analysis on leak rate vs. microannulus thickness.

    Parameters
    ----------
    tau_range_um : np.ndarray  Thickness values in micrometres
    R_csg : float              Casing radius (m)
    L : float                  Microannulus length (m)
    delta_P_Pa : float         Pressure drop (Pa)
    fluids : dict              {name: FluidProperties}

    Returns
    -------
    dict with leak rates for each fluid type.
    """
    results = {"tau_um": tau_range_um}
    for name, fluid in fluids.items():
        rates = []
        for tau_um in tau_range_um:
            tau = tau_um * 1e-6
            if fluid.is_gas:
                P1 = delta_P_Pa + 101325.0
                P2 = 101325.0
                q = gas_leak_rate(tau, R_csg, P1, P2, L, fluid)
            else:
                q = liquid_leak_rate(tau, R_csg, delta_P_Pa, L, fluid)
            rates.append(q)
        results[name] = np.array(rates)
    return results


def test_all():
    """Test all functions with synthetic data."""
    print("=" * 70)
    print("Testing: microannuli_leak_rate (Machicote et al., 2025)")
    print("=" * 70)

    well = WellCompletion()

    # Test microannulus thickness from AI
    ai_values = [5.0, 3.5, 2.0, 1.5]
    for ai in ai_values:
        tau = microannulus_thickness_from_ai(
            ai, ai_cement=5.0, ai_liquid=1.5,
            casing_thickness_m=well.casing_thickness_m)
        print(f"  AI={ai:.1f} MRayl → τ={tau*1e6:.0f} µm")
    assert microannulus_thickness_from_ai(5.0, 5.0, 1.5, 0.012) == 0.0

    # Test geometric mean
    thicknesses = np.array([50, 100, 150, 80, 0, 120]) * 1e-6
    geo_mean = geometric_mean_thickness(thicknesses)
    print(f"  Geometric mean thickness: {geo_mean*1e6:.1f} µm")
    assert geo_mean > 0

    # Test Omega function
    omega = omega_function(well.casing_radius_m, 100e-6)
    print(f"  Ω(R={well.casing_radius_m:.3f}m, τ=100µm) = {omega:.3e} m⁴")
    assert omega > 0

    # Test liquid leak rate (water)
    water = FluidProperties.from_type(FluidType.FRESH_WATER)
    Q_water = liquid_leak_rate(
        tau=100e-6, R_csg=well.casing_radius_m,
        delta_P=20e5, L=100.0,  # 20 bar, 100 m
        fluid=water, theta_deg=90.0)
    Q_m3_d = Q_water * 86400.0
    print(f"  Water leak rate (100µm, 20bar, 100m): {Q_m3_d:.4f} m³/d")
    assert Q_water >= 0

    # Test gas leak rate (methane)
    methane = FluidProperties.from_type(FluidType.METHANE)
    Q_gas = gas_leak_rate(
        tau=100e-6, R_csg=well.casing_radius_m,
        P1=200e5, P2=100e5, L=100.0,
        fluid=methane)
    Q_gas_m3_d = Q_gas * 86400.0
    print(f"  Methane leak rate (100µm, ΔP=100bar, 100m): {Q_gas_m3_d:.4f} m³/d")
    assert Q_gas > 0

    # Test bond index
    bi = bond_index(cbl_amplitude_mV=15.0, cbl_free_pipe_mV=80.0,
                    cbl_well_bonded_mV=2.0)
    print(f"  Bond index (CBL=15mV): {bi:.2f}")
    assert 0 <= bi <= 1

    # Test sensitivity analysis
    fluids = {
        "water": FluidProperties.from_type(FluidType.FRESH_WATER),
        "methane": FluidProperties.from_type(FluidType.METHANE),
        "oil": FluidProperties.from_type(FluidType.OIL),
    }
    tau_range = np.array([10, 50, 100, 200, 300, 400])
    sens = sensitivity_analysis(tau_range, well.casing_radius_m,
                                L=50.0, delta_P_Pa=10e5, fluids=fluids)
    print("  Sensitivity analysis (leak rate in m³/d):")
    for name in fluids:
        rates_m3_d = sens[name] * 86400
        print(f"    {name}: {rates_m3_d}")

    # Verify: thicker microannulus = higher leak rate (for each fluid)
    for name in fluids:
        assert np.all(np.diff(sens[name]) >= 0), \
            f"Leak rate should increase with thickness for {name}"

    print("  All tests PASSED.\n")


if __name__ == "__main__":
    test_all()
