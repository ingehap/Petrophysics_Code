"""
What Causes Mud-Logging Mud Gas Response to Vary and Two Techniques to Quantify
================================================================================
Based on: Donovan, W.S. (2024), "What Causes Mud-Logging Mud Gas Response to
Vary and Two Techniques to Quantify Mud Gas," Petrophysics, 65(4), pp. 565-584.
DOI: 10.30632/PJV65N4-2024a10

Implements:
  - Gas marker technique for quantitative mud gas content (SCF/ton)
  - Normalization technique for comparing gas across wells
  - Factors causing mud gas response variation (ROP, mud weight, etc.)
  - Production correlation from mud-log gas content
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DrillingParameters:
    """Drilling parameters that affect mud gas response."""
    depth: np.ndarray
    rop: np.ndarray          # rate of penetration (ft/hr)
    mud_weight: np.ndarray   # mud weight (ppg)
    flow_rate: np.ndarray    # mud flow rate (gpm)
    bit_diameter: float      # bit size (inches)
    annular_volume: float    # annular volume factor (bbl/ft)


@dataclass
class MudGasReading:
    """Raw mud gas readings from the mud logging unit."""
    depth: np.ndarray
    total_gas: np.ndarray    # total hydrocarbon gas (units %)
    c1: np.ndarray           # methane (ppm)
    c2: np.ndarray           # ethane
    c3: np.ndarray           # propane
    c4: np.ndarray           # butane
    c5: np.ndarray           # pentane


def compute_gas_content_scf_per_ton(gas_reading: MudGasReading,
                                    drilling: DrillingParameters,
                                    rock_density: float = 2.5,
                                    gas_trap_efficiency: float = 0.7
                                    ) -> np.ndarray:
    """Compute gas content in SCF per ton of rock using gas markers.

    The paper presents this as a universal technique to quantify mud gas
    by accounting for drilling parameters. The calculation corrects for:
      - Rate of penetration (rock volume per unit time)
      - Mud flow rate (dilution effect)
      - Bit size (rock volume per foot drilled)
      - Gas trap efficiency

    SCF/ton = (gas_ppm * flow_rate) / (ROP * bit_area * density * efficiency)
    """
    # Bit cross-sectional area (ft^2)
    bit_area_ft2 = np.pi * (drilling.bit_diameter / 12 / 2) ** 2

    # Rock mass per foot drilled (tons/ft)
    rock_mass_per_ft = bit_area_ft2 * rock_density * 62.4 / 2000  # convert to tons

    # Gas volume rate (SCF/min from ppm in mud flow)
    total_gas_ppm = gas_reading.c1 + gas_reading.c2 + gas_reading.c3 + \
                    gas_reading.c4 + gas_reading.c5
    gas_volume_scf_per_min = total_gas_ppm * 1e-6 * drilling.flow_rate * 0.1337  # gpm to ft3/min

    # Rock generation rate (tons/min)
    rock_rate_tons_per_min = drilling.rop / 60.0 * rock_mass_per_ft

    # Gas content
    gas_content = gas_volume_scf_per_min / (rock_rate_tons_per_min + 1e-10) / gas_trap_efficiency
    return np.clip(gas_content, 0, None)


def normalize_gas_response(gas_reading: MudGasReading,
                           drilling: DrillingParameters,
                           reference_rop: float = 30.0,
                           reference_flow: float = 500.0,
                           reference_mw: float = 10.0) -> MudGasReading:
    """Normalize mud gas response to reference drilling conditions.

    This is the normalization technique from the paper, which enables
    comparison of gas readings across wells with different drilling
    parameters. Corrects for:
      - ROP: faster drilling generates more gas per unit time but less
        per foot (dilution)
      - Flow rate: higher flow dilutes gas
      - Mud weight: heavier mud suppresses gas liberation
    """
    # Correction factors
    rop_factor = drilling.rop / reference_rop
    flow_factor = drilling.flow_rate / reference_flow
    mw_factor = (drilling.mud_weight / reference_mw) ** 2  # quadratic effect

    correction = rop_factor * flow_factor * mw_factor

    return MudGasReading(
        depth=gas_reading.depth,
        total_gas=gas_reading.total_gas / correction,
        c1=gas_reading.c1 / correction,
        c2=gas_reading.c2 / correction,
        c3=gas_reading.c3 / correction,
        c4=gas_reading.c4 / correction,
        c5=gas_reading.c5 / correction,
    )


def estimate_production_from_gas_content(gas_content_scf_ton: np.ndarray,
                                         net_thickness_ft: float,
                                         drainage_area_acres: float = 40,
                                         recovery_factor: float = 0.6,
                                         rock_density_ton_ft3: float = 0.078
                                         ) -> dict:
    """Estimate production potential from gas content.

    The paper shows that mud-log gas content is a good predictor of
    production performance in coal gas unconventional reservoirs.
    """
    mean_gc = np.mean(gas_content_scf_ton)
    # Volume of rock in drainage area
    rock_volume_ft3 = drainage_area_acres * 43560 * net_thickness_ft
    rock_mass_tons = rock_volume_ft3 * rock_density_ton_ft3
    # Total gas in place
    gip_scf = mean_gc * rock_mass_tons
    # Estimated ultimate recovery
    eur_scf = gip_scf * recovery_factor

    return {
        "mean_gas_content_scf_ton": mean_gc,
        "gas_in_place_mscf": gip_scf / 1000,
        "eur_mscf": eur_scf / 1000,
        "drainage_area_acres": drainage_area_acres,
    }


def gas_response_sensitivity(base_gas: float, rop_change: float = 1.0,
                             flow_change: float = 1.0,
                             mw_change: float = 1.0) -> dict:
    """Analyze sensitivity of gas response to drilling parameter changes.

    The paper explains that the wide variation in mud gas response is
    primarily caused by these three factors.
    """
    # Gas response inversely proportional to ROP * flow, and suppressed by MW
    response = base_gas / (rop_change * flow_change * mw_change ** 2)
    return {
        "base_gas": base_gas,
        "adjusted_gas": response,
        "change_factor": response / base_gas,
        "rop_effect": 1 / rop_change,
        "flow_effect": 1 / flow_change,
        "mw_effect": 1 / mw_change ** 2,
    }


def test_all():
    """Test mud gas quantification techniques."""
    print("=" * 70)
    print("Testing: Mud Gas Quantification Techniques (Donovan, 2024)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    n_pts = 80

    # Generate synthetic well data
    depths = np.linspace(1000, 2000, n_pts)
    drilling = DrillingParameters(
        depth=depths,
        rop=20 + 15 * rng.random(n_pts),
        mud_weight=9.5 + 1.0 * rng.random(n_pts),
        flow_rate=400 + 100 * rng.random(n_pts),
        bit_diameter=8.5,
        annular_volume=0.0459,
    )

    # Gas increases in reservoir zone (depths 1400-1700)
    reservoir_mask = (depths > 1400) & (depths < 1700)
    base_gas = 50 + 20 * rng.random(n_pts)
    base_gas[reservoir_mask] += 200 + 100 * rng.random(reservoir_mask.sum())

    gas_reading = MudGasReading(
        depth=depths,
        total_gas=base_gas / 100,
        c1=base_gas * 0.7,
        c2=base_gas * 0.15,
        c3=base_gas * 0.08,
        c4=base_gas * 0.04,
        c5=base_gas * 0.03,
    )

    # Gas marker technique
    gc = compute_gas_content_scf_per_ton(gas_reading, drilling)
    print(f"  Gas content (SCF/ton):")
    print(f"    Non-reservoir mean: {gc[~reservoir_mask].mean():.1f}")
    print(f"    Reservoir mean:     {gc[reservoir_mask].mean():.1f}")

    # Normalization technique
    normalized = normalize_gas_response(gas_reading, drilling)
    print(f"\n  Normalized total gas:")
    print(f"    Raw mean:        {gas_reading.total_gas.mean():.3f}%")
    print(f"    Normalized mean: {normalized.total_gas.mean():.3f}%")

    # Production estimate
    prod = estimate_production_from_gas_content(
        gc[reservoir_mask], net_thickness_ft=300, drainage_area_acres=40
    )
    print(f"\n  Production estimate:")
    print(f"    Mean gas content: {prod['mean_gas_content_scf_ton']:.1f} SCF/ton")
    print(f"    Gas in place: {prod['gas_in_place_mscf']:.0f} MSCF")
    print(f"    EUR: {prod['eur_mscf']:.0f} MSCF")

    # Sensitivity analysis
    print(f"\n  Sensitivity analysis (base gas = 100 units):")
    for label, rop_c, flow_c, mw_c in [
        ("Double ROP", 2.0, 1.0, 1.0),
        ("Double flow", 1.0, 2.0, 1.0),
        ("10% MW increase", 1.0, 1.0, 1.1),
        ("All combined", 2.0, 2.0, 1.1),
    ]:
        result = gas_response_sensitivity(100, rop_c, flow_c, mw_c)
        print(f"    {label}: gas = {result['adjusted_gas']:.1f} "
              f"({result['change_factor']:.2f}x)")

    print("\n  [PASS] Mud gas quantification tests completed.")
    return True


if __name__ == "__main__":
    test_all()
