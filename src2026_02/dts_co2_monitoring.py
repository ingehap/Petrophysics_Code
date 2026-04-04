"""
Article 2: Real-Time CO2 Injection Monitoring Through Fiber Optics:
Physics-Based Modeling of DTS Data for Time-Lapse Assessment of Fluid Properties.

Authors: Pirrone and Mantegazza (2026)
DOI: 10.30632/PJV67N1-2026a2

Implements the DTS-based fluid mapping workflow for CO2 injection wells,
including Raman scattering temperature determination, fluid mapping via
volumetric heat capacity contrast, and brine displacement tracking.

Key ideas implemented:
    - Raman DTS temperature computation (Stokes/anti-Stokes ratio)
    - Ramey (1962) wellbore temperature model for injection wells
    - Fluid mapping operators (Omega_1..4) for CO2 vs. brine discrimination
    - Volumetric heat capacity model for CO2 and brine
    - Real-time CO2-brine contact tracking

References
----------
Pirrone and Mantegazza (2026), Petrophysics, 67(1), 27-37.
Dakin et al. (1985); Ramey (1962); Witterholt and Tixier (1972).
Smolen and van der Spek (2003); Brown (2016); Kotlar et al. (2021).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# Physical constants
BOLTZMANN = 1.380649e-23   # J/K
HBAR = 1.054571817e-34     # J·s
RAMAN_OMEGA = 2.0e13       # Approximate Raman angular frequency (rad/s)


@dataclass
class DTSSystem:
    """Configuration for a DTS acquisition system.

    Attributes
    ----------
    fiber_length : float
        Total fiber length (m).
    spatial_resolution : float
        Spatial sampling interval (m), typically 0.5-2 m.
    time_interval : float
        Temporal sampling interval (s).
    calibration_temp : float
        Surface calibration temperature (K).
    attenuation_stokes : float
        Attenuation rate for Stokes band (1/m).
    attenuation_anti_stokes : float
        Attenuation rate for anti-Stokes band (1/m).
    """
    fiber_length: float = 3000.0
    spatial_resolution: float = 1.0
    time_interval: float = 60.0
    calibration_temp: float = 293.15
    attenuation_stokes: float = 1.0e-4
    attenuation_anti_stokes: float = 1.2e-4


@dataclass
class WellCompletion:
    """Well completion parameters for thermal simulation.

    Attributes
    ----------
    tubing_inner_radius : float
        Inner radius of tubing (m).
    heat_transfer_coeff : float
        Overall heat transfer coefficient U (W/(m²·K)).
    geothermal_gradient : float
        Geothermal gradient (K/m), positive downward.
    surface_temperature : float
        Surface temperature (K).
    formation_thermal_conductivity : float
        Thermal conductivity of surrounding formation (W/(m·K)).
    """
    tubing_inner_radius: float = 0.031
    heat_transfer_coeff: float = 15.0
    geothermal_gradient: float = 0.03
    surface_temperature: float = 293.15
    formation_thermal_conductivity: float = 2.0


def dts_temperature_from_raman(stokes: np.ndarray,
                               anti_stokes: np.ndarray,
                               z: np.ndarray,
                               system: DTSSystem) -> np.ndarray:
    """Compute temperature profile from Raman DTS Stokes/anti-Stokes ratio.

    Implements Eq. 4 from the paper (Dakin et al., 1985):
        T(z,t) = (ℏω / k_B) / [ln(I_S/I_aS) + Δα·z + C]

    where Δα accounts for differential attenuation between Stokes and
    anti-Stokes bands.

    Parameters
    ----------
    stokes : np.ndarray
        Stokes intensity at each depth.
    anti_stokes : np.ndarray
        Anti-Stokes intensity at each depth.
    z : np.ndarray
        Depth positions along the fiber (m).
    system : DTSSystem
        DTS system configuration with calibration parameters.

    Returns
    -------
    np.ndarray
        Temperature profile T(z) in Kelvin.
    """
    delta_alpha = system.attenuation_stokes - system.attenuation_anti_stokes
    energy_term = HBAR * RAMAN_OMEGA / BOLTZMANN

    ratio = np.clip(stokes / (anti_stokes + 1e-30), 1e-10, None)
    log_ratio = np.log(ratio)

    # Calibration constant from surface measurement
    C = energy_term / system.calibration_temp - log_ratio[0] - delta_alpha * z[0]

    temperature = energy_term / (log_ratio + delta_alpha * z + C)
    return temperature


def geothermal_profile(z: np.ndarray, well: WellCompletion) -> np.ndarray:
    """Compute the undisturbed geothermal temperature profile.

    Parameters
    ----------
    z : np.ndarray
        True vertical depth positions (m).
    well : WellCompletion
        Well completion parameters.

    Returns
    -------
    np.ndarray
        Undisturbed formation temperature at each depth (K).
    """
    return well.surface_temperature + well.geothermal_gradient * z


def injection_temperature_model(z: np.ndarray,
                                t: float,
                                well: WellCompletion,
                                injection_rate: float,
                                fluid_density: float,
                                fluid_cp: float,
                                injection_temp: float) -> np.ndarray:
    """Ramey-type wellbore temperature model for injection (Eq. 5-6).

    T_i(z,t) = T_s(z) + [T_inj - T_s(0)] * exp(-z/A)

    where A is the relaxation distance:
        A = q * ρ * c_p / (2π * r_int * U)

    Parameters
    ----------
    z : np.ndarray
        Depth positions along the well (m).
    t : float
        Time since start of injection (s). Used for time function f(t).
    well : WellCompletion
        Well completion parameters.
    injection_rate : float
        Volumetric injection rate q (m³/s).
    fluid_density : float
        Density of injected fluid (kg/m³).
    fluid_cp : float
        Specific heat capacity of injected fluid (J/(kg·K)).
    injection_temp : float
        Temperature of injected fluid at surface (K).

    Returns
    -------
    np.ndarray
        Modeled temperature profile T_i(z,t) in Kelvin.
    """
    T_s = geothermal_profile(z, well)

    # Relaxation distance A (Eq. 6)
    A = (injection_rate * fluid_density * fluid_cp) / (
        2.0 * np.pi * well.tubing_inner_radius * well.heat_transfer_coeff
    )

    # Time function f(t): simplified dimensionless time correction
    # In the full model this involves Ei functions; here use log approx
    t_dim = well.formation_thermal_conductivity * t / (
        fluid_density * fluid_cp * well.tubing_inner_radius**2 + 1e-30
    )
    f_t = np.log(max(t_dim, 1.0) + 0.5)  # Simplified

    A_eff = A * f_t / (f_t + 1.0)

    T_injection = T_s + (injection_temp - T_s[0]) * np.exp(-z / (A_eff + 1e-10))

    return T_injection


def volumetric_heat_capacity_brine(T: float, P: float,
                                   salinity: float = 0.0) -> float:
    """Volumetric heat capacity of brine s = ρ·c_p (MJ/(K·m³)).

    Brine has a fairly constant value near 4 MJ/(K·m³). Small variations
    with pressure (<0.02 MJ/K/m³ per 3000 psi) and temperature
    (<0.05 MJ/K/m³ per 40°C).

    Parameters
    ----------
    T : float
        Temperature (°C).
    P : float
        Pressure (psi).
    salinity : float
        Brine salinity (fraction, e.g. 0.1 for 10%).

    Returns
    -------
    float
        Volumetric heat capacity in MJ/(K·m³).
    """
    s_base = 4.18  # Pure water at 25°C, 1 atm
    # Temperature correction
    s = s_base - 0.00125 * (T - 25.0)
    # Pressure correction (small)
    s += 6.5e-6 * P
    # Salinity correction
    s *= (1.0 - 0.7 * salinity)
    return s


def volumetric_heat_capacity_co2(T: float, P: float) -> float:
    """Volumetric heat capacity of CO2 s = ρ·c_p (MJ/(K·m³)).

    CO2 heat capacity varies strongly near the critical point (31°C, 1070 psi).
    Far from the critical point: ~0.1 MJ/K/m³ (gas) or ~2 MJ/K/m³ (liquid/SC).
    Near the critical point it can exceed brine values.

    Parameters
    ----------
    T : float
        Temperature (°C).
    P : float
        Pressure (psi).

    Returns
    -------
    float
        Volumetric heat capacity in MJ/(K·m³).
    """
    T_c, P_c = 31.0, 1070.0  # Critical point
    T_r = (T + 273.15) / (T_c + 273.15)
    P_r = P / P_c

    if P < P_c * 0.5:
        # Gas phase, low pressure
        s = 0.05 + 0.05 * P_r
    elif abs(T - T_c) < 10 and abs(P - P_c) < 300:
        # Near critical point: sharp peak
        delta_T = max(abs(T - T_c), 0.5)
        delta_P = max(abs(P - P_c), 50.0)
        s = 2.0 + 3.0 / (delta_T * 0.1 + delta_P * 0.001)
        s = min(s, 8.0)  # Cap at reasonable maximum
    elif P > P_c and T > T_c:
        # Supercritical
        s = 1.5 + 0.5 * P_r / T_r
    else:
        # Liquid phase
        s = 1.8 + 0.2 * P_r
    return s


def fluid_mapping_operator(dts_profiles: np.ndarray,
                           z: np.ndarray,
                           baseline_index: int = 0,
                           stack_count: int = 3) -> np.ndarray:
    """Apply fluid mapping operators Ω1–Ω3 to DTS data (Eqs. 7-9).

    Workflow:
        Ω1: Stack temperature profiles to improve SNR
        Ω2: Subtract pre-injection baseline (relaxation function Ψ)
        Ω3: Standardize along z to map volumetric heat capacity differences

    The output Γ(s,z,t) discriminates CO2 from brine based on their different
    volumetric heat capacities.

    Parameters
    ----------
    dts_profiles : np.ndarray
        Temperature profiles, shape (n_times, n_depths). Each row is a
        temperature trace at a given time.
    z : np.ndarray
        Depth positions, shape (n_depths,).
    baseline_index : int
        Index of the pre-injection baseline profile.
    stack_count : int
        Number of adjacent profiles to stack for SNR improvement (Ω1).

    Returns
    -------
    np.ndarray
        Fluid mapping function Γ(z,t), shape (n_times, n_depths).
        Positive values indicate one fluid, negative the other.
    """
    n_times, n_depths = dts_profiles.shape

    # Ω1: Stacking for SNR improvement
    stacked = np.copy(dts_profiles)
    half = stack_count // 2
    for i in range(n_times):
        lo = max(0, i - half)
        hi = min(n_times, i + half + 1)
        stacked[i] = dts_profiles[lo:hi].mean(axis=0)

    # Ω2: Subtract baseline (relaxation function Ψ)
    baseline = stacked[baseline_index]
    psi = stacked - baseline[None, :]  # Ψ(z,t)

    # Ω3: Standardize along z for fluid mapping
    gamma = np.zeros_like(psi)
    for i in range(n_times):
        if i == baseline_index:
            continue
        profile = psi[i]
        mu = np.mean(profile)
        sigma = np.std(profile)
        if sigma > 1e-10:
            gamma[i] = (profile - mu) / sigma
        else:
            gamma[i] = 0.0

    return gamma


def enforce_physical_constraints(gamma: np.ndarray,
                                 z: np.ndarray) -> np.ndarray:
    """Apply operator Ω4: enforce physical constraints on fluid mapping.

    The CO2-brine contact must move downward over time, and CO2 must be above
    the brine at all times.

    Parameters
    ----------
    gamma : np.ndarray
        Raw fluid mapping function, shape (n_times, n_depths).
    z : np.ndarray
        Depth positions (m).

    Returns
    -------
    np.ndarray
        Constrained fluid map: +1 = CO2, -1 = brine, 0 = uncertain.
    """
    n_times, n_depths = gamma.shape
    fluid_map = np.zeros_like(gamma)
    prev_contact_idx = 0

    for t in range(1, n_times):
        profile = gamma[t]
        if np.all(np.abs(profile) < 1e-10):
            continue

        # Find contact as the transition point
        threshold = 0.0
        crossings = np.where(np.diff(np.sign(profile - threshold)))[0]

        if len(crossings) > 0:
            contact_idx = crossings[0]
            # Enforce monotonically increasing contact depth
            contact_idx = max(contact_idx, prev_contact_idx)
            prev_contact_idx = contact_idx

            fluid_map[t, :contact_idx] = 1.0   # CO2 above
            fluid_map[t, contact_idx:] = -1.0   # Brine below
        else:
            # All one fluid
            if np.mean(profile) > 0:
                fluid_map[t, :] = 1.0
            else:
                fluid_map[t, :] = -1.0

    return fluid_map


def track_co2_brine_contact(fluid_map: np.ndarray,
                            z: np.ndarray,
                            times: np.ndarray) -> tuple:
    """Track the CO2-brine contact depth over time.

    Parameters
    ----------
    fluid_map : np.ndarray
        Constrained fluid map from enforce_physical_constraints.
    z : np.ndarray
        Depth positions (m).
    times : np.ndarray
        Time values (hours).

    Returns
    -------
    contact_depths : np.ndarray
        Depth of CO2-brine contact at each time step.
    displacement_rate : np.ndarray
        Rate of brine displacement (m/hr).
    """
    n_times = fluid_map.shape[0]
    contact_depths = np.full(n_times, np.nan)

    for t in range(n_times):
        co2_mask = fluid_map[t] > 0
        if np.any(co2_mask):
            contact_depths[t] = z[np.where(co2_mask)[0][-1]]

    displacement_rate = np.zeros(n_times)
    for t in range(1, n_times):
        if not np.isnan(contact_depths[t]) and not np.isnan(contact_depths[t-1]):
            dt = times[t] - times[t-1]
            if dt > 0:
                displacement_rate[t] = (
                    (contact_depths[t] - contact_depths[t-1]) / dt
                )

    return contact_depths, displacement_rate
