"""
Genetic Analysis of Overpressure While Drilling Based on Isotope Logging
=========================================================================
Based on: Hu et al., "Genetic Analysis of Overpressure While Drilling
Based on Isotope Logging Technology",
Petrophysics, Vol. 66, No. 2, April 2025, pp. 283–293.

Implements:
  - Eaton and Bowers pore pressure prediction methods
  - Normal compaction trend (NCT) estimation
  - Loading/unloading curve classification
  - Methane carbon isotope (δ¹³C) analysis for overpressure genetic diagnosis
  - Real-time overpressure cause identification

Reference: https://doi.org/10.30632/PJV66N2-2025a7 (SPWLA)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum


class OverpressureCause(Enum):
    """Classification of overpressure genetic causes."""
    UNDERCOMPACTION = "undercompaction"       # Loading mechanism
    FLUID_EXPANSION = "fluid_expansion"       # Unloading mechanism
    PRESSURE_CONDUCTION = "pressure_conduction"  # External source
    NEAR_SOURCE = "near_source"              # Self-sourced
    MIXED = "mixed"
    NORMAL = "normal_pressure"


@dataclass
class WellData:
    """Drilling and logging data at each depth point."""
    depth_m: np.ndarray
    sonic_dt_us_ft: np.ndarray      # Acoustic transit time (μs/ft)
    density_g_cm3: np.ndarray       # Bulk density
    resistivity_ohm_m: np.ndarray   # Deep resistivity
    dc_exponent: np.ndarray         # Rock drillability (dc index)
    delta_13C_permil: Optional[np.ndarray] = None  # Methane δ¹³C (‰ VPDB)
    is_mudstone: Optional[np.ndarray] = None  # Boolean mask for mudstone layers


def normal_compaction_trend(depth_m: np.ndarray,
                            dt_surface: float = 200.0,
                            c: float = 0.0004) -> np.ndarray:
    """
    Compute normal compaction trend (NCT) for sonic transit time.

    dt_n(z) = dt_surface * exp(-c * z)

    Parameters
    ----------
    depth_m : np.ndarray  Depth values
    dt_surface : float    Surface transit time (μs/ft)
    c : float             Compaction coefficient (1/m)

    Returns
    -------
    np.ndarray : Normal transit time at each depth.
    """
    return dt_surface * np.exp(-c * depth_m)


def eaton_pore_pressure(depth_m: np.ndarray,
                        sonic_dt: np.ndarray,
                        dt_normal: np.ndarray,
                        overburden_gradient: float = 1.0,
                        hydrostatic_gradient: float = 0.465,
                        eaton_exponent: float = 3.0) -> np.ndarray:
    """
    Eaton's method for pore pressure prediction from sonic data.

    Pp = OBG - (OBG - Pn) * (dt_n / dt_obs)^n

    Parameters
    ----------
    depth_m : np.ndarray
    sonic_dt : np.ndarray       Observed transit time (μs/ft)
    dt_normal : np.ndarray      Normal compaction transit time
    overburden_gradient : float  Overburden pressure gradient (psi/ft)
    hydrostatic_gradient : float  Normal hydrostatic gradient (psi/ft)
    eaton_exponent : float       Eaton exponent (typically 3.0)

    Returns
    -------
    np.ndarray : Pore pressure gradient (psi/ft equivalent)
    """
    depth_ft = depth_m * 3.28084
    ratio = np.clip(dt_normal / sonic_dt, 0.01, 100.0)
    Pp = overburden_gradient - (overburden_gradient - hydrostatic_gradient) * \
         ratio ** eaton_exponent
    return Pp


def bowers_pore_pressure(depth_m: np.ndarray,
                         sonic_velocity_ft_s: np.ndarray,
                         A: float = 10.0,
                         B: float = 0.7,
                         overburden_psi: Optional[np.ndarray] = None,
                         unloading: bool = False,
                         U: float = 3.0,
                         v_max: float = 14000.0,
                         sigma_max: float = 5000.0) -> np.ndarray:
    """
    Bowers' method for pore pressure from sonic velocity.

    Loading:   V = V0 + A * sigma^B
    Unloading: V = V0 + A * (sigma_max * (sigma/sigma_max)^(1/U))^B

    Parameters
    ----------
    depth_m : np.ndarray
    sonic_velocity_ft_s : np.ndarray  P-wave velocity
    A, B : float  Bowers loading curve parameters
    overburden_psi : np.ndarray  Overburden pressure
    unloading : bool
    U : float  Unloading parameter
    v_max, sigma_max : float  Maximum velocity / effective stress

    Returns
    -------
    np.ndarray : Pore pressure in psi.
    """
    V0 = 5000.0  # Mudline velocity (ft/s)
    depth_ft = depth_m * 3.28084

    if overburden_psi is None:
        overburden_psi = 1.0 * depth_ft  # ~1.0 psi/ft

    if not unloading:
        # Loading curve: sigma_eff = ((V - V0) / A)^(1/B)
        sigma_eff = np.clip((sonic_velocity_ft_s - V0) / A, 0.01, None) ** (1.0 / B)
    else:
        # Unloading curve
        sigma_eff = sigma_max * ((sonic_velocity_ft_s - V0) / (A * sigma_max ** B)) ** U

    Pp = overburden_psi - sigma_eff
    return Pp


def loading_unloading_classification(sonic_dt: np.ndarray,
                                     density: np.ndarray,
                                     depth_m: np.ndarray) -> np.ndarray:
    """
    Classify formation as loading or unloading based on sonic-density crossplot.

    Loading (undercompaction): sonic and density both follow NCT deviation
    Unloading (fluid expansion): sonic deviates but density stays on NCT

    Parameters
    ----------
    sonic_dt : np.ndarray  Sonic transit time
    density : np.ndarray   Bulk density
    depth_m : np.ndarray

    Returns
    -------
    np.ndarray : 0=normal, 1=loading, 2=unloading
    """
    dt_nct = normal_compaction_trend(depth_m)
    # Expected density from NCT
    rho_nct = 1.6 + 0.0003 * depth_m  # Simplified density NCT

    dt_deviation = (sonic_dt - dt_nct) / dt_nct
    rho_deviation = (density - rho_nct) / rho_nct

    classification = np.zeros(len(depth_m), dtype=int)

    for i in range(len(depth_m)):
        if dt_deviation[i] > 0.1:  # Sonic exceeds NCT (overpressured)
            if rho_deviation[i] < -0.03:
                # Both sonic and density deviate: loading (undercompaction)
                classification[i] = 1
            else:
                # Only sonic deviates: unloading (fluid expansion)
                classification[i] = 2

    return classification


def isotope_overpressure_diagnosis(delta_13C: np.ndarray,
                                   depth_m: np.ndarray,
                                   window_size: int = 5) -> List[OverpressureCause]:
    """
    Diagnose overpressure cause from methane carbon isotope data.

    Method proposed by Hu et al. (2025):
    - If δ¹³C increases uniformly or changes slightly with depth,
      overpressure is near-source or self-sourced.
    - If δ¹³C changes significantly with depth, overpressure is
      from an external source (pressure conduction or fluid expansion).

    Parameters
    ----------
    delta_13C : np.ndarray  Methane δ¹³C values (‰ VPDB)
    depth_m : np.ndarray    Corresponding depths
    window_size : int       Sliding window for gradient analysis

    Returns
    -------
    List[OverpressureCause] : Diagnosed cause at each depth.
    """
    n = len(delta_13C)
    causes = [OverpressureCause.NORMAL] * n

    for i in range(window_size, n):
        window = delta_13C[i - window_size:i + 1]
        depth_window = depth_m[i - window_size:i + 1]

        if len(window) < 2:
            continue

        # Gradient of δ¹³C with depth
        dz = depth_window[-1] - depth_window[0]
        if dz < 1.0:
            continue

        gradient = (window[-1] - window[0]) / dz  # ‰ per metre
        variability = np.std(window)

        if abs(gradient) < 0.01 and variability < 1.0:
            # Uniform or slight change → near-source / self-source
            causes[i] = OverpressureCause.NEAR_SOURCE
        elif abs(gradient) > 0.05 or variability > 2.0:
            # Significant change → external source
            if gradient > 0:
                causes[i] = OverpressureCause.FLUID_EXPANSION
            else:
                causes[i] = OverpressureCause.PRESSURE_CONDUCTION
        else:
            causes[i] = OverpressureCause.MIXED

    return causes


def pore_pressure_coefficient(pore_pressure_psi: np.ndarray,
                              depth_m: np.ndarray) -> np.ndarray:
    """
    Compute pore pressure coefficient (ratio to hydrostatic).

    Parameters
    ----------
    pore_pressure_psi : np.ndarray
    depth_m : np.ndarray

    Returns
    -------
    np.ndarray : Pressure coefficient (1.0 = hydrostatic)
    """
    depth_ft = depth_m * 3.28084
    hydrostatic = 0.465 * depth_ft
    return np.where(hydrostatic > 0, pore_pressure_psi / hydrostatic, 1.0)


def test_all():
    """Test all functions with synthetic data."""
    print("=" * 70)
    print("Testing: overpressure_isotope (Hu et al., 2025)")
    print("=" * 70)

    np.random.seed(42)
    n = 100
    depth = np.linspace(2000, 5000, n)

    # Synthetic well data
    dt_normal = normal_compaction_trend(depth)
    # Add overpressure zone at depth 3500-4500m
    sonic_dt = dt_normal.copy()
    sonic_dt[60:90] *= 1.3  # Elevated transit time = overpressure

    density = 2.2 + 0.0001 * depth + 0.02 * np.random.randn(n)

    print(f"  Depth range: {depth.min():.0f} – {depth.max():.0f} m")

    # Eaton method
    Pp = eaton_pore_pressure(depth, sonic_dt, dt_normal)
    print(f"  Eaton Pp range: {Pp.min():.3f} – {Pp.max():.3f} psi/ft")
    assert Pp.max() > 0.465, "Should detect overpressure"

    # Bowers method
    velocity = 1e6 / sonic_dt  # Convert dt to velocity
    Pp_bowers = bowers_pore_pressure(depth, velocity, A=10, B=0.7)
    print(f"  Bowers Pp range: {Pp_bowers.min():.0f} – {Pp_bowers.max():.0f} psi")

    # Loading/unloading classification
    classes = loading_unloading_classification(sonic_dt, density, depth)
    n_loading = np.sum(classes == 1)
    n_unloading = np.sum(classes == 2)
    print(f"  Classification: {n_loading} loading, {n_unloading} unloading points")

    # Isotope diagnosis
    # Near-source: δ¹³C increases smoothly
    delta_13C = -35.0 + 0.002 * depth + 0.5 * np.random.randn(n)
    # Add anomaly in overpressured zone (external source)
    delta_13C[70:85] += np.linspace(0, 8, 15)

    causes = isotope_overpressure_diagnosis(delta_13C, depth)
    cause_counts = {}
    for c in causes:
        cause_counts[c.value] = cause_counts.get(c.value, 0) + 1
    print(f"  Isotope diagnosis: {cause_counts}")

    # Pressure coefficient
    pc = pore_pressure_coefficient(Pp_bowers, depth)
    print(f"  Pressure coefficient range: {pc.min():.2f} – {pc.max():.2f}")

    print("  All tests PASSED.\n")


if __name__ == "__main__":
    test_all()
