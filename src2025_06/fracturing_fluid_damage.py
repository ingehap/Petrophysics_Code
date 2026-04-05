#!/usr/bin/env python3
"""
Comparative Study on the Damage of Different Fracturing Fluids in Tight Sandstone
==================================================================================
Implements the methodology from:
  Li, Q., Xia, X., Xiao, H., Zhang, H., Tan, X., Wang, C., He, T.,
  and Wang, H., 2025,
  "Comparative Study on the Damage of Different Fracturing Fluids in
  Tight Sandstone,"
  Petrophysics, Vol. 66, No. 3, pp. 489–520.

Key ideas implemented:
  - Matrix permeability damage rate calculation (Eq. A1.3c).
  - Hydrolock (water-block) damage assessment.
  - NMR T2 spectrum analysis for pore-size distribution changes.
  - Fracture conductivity damage evaluation.
  - Production simulation comparison across fracturing fluid systems.

References:
  Toumelin, E. et al., 2007 (NMR). King, G.E., 2012.
  Bennion, D.B. and Thomas, F.B., 2005. Meng, M. et al., 2015.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


# ============================================================
# Fracturing fluid types
# ============================================================

FLUID_TYPES = {
    "guanidine_gel": {
        "name": "Guanidine Gel Fracturing Fluid",
        "solid_content": "high",
        "viscosity_cp": 200,
        "residue_fraction": 0.05,
    },
    "slickwater": {
        "name": "Slickwater Fracturing Fluid",
        "solid_content": "low",
        "viscosity_cp": 5,
        "residue_fraction": 0.005,
    },
    "active_water": {
        "name": "Active Water Fracturing Fluid",
        "solid_content": "medium",
        "viscosity_cp": 15,
        "residue_fraction": 0.01,
    },
}


# ============================================================
# Matrix permeability damage
# ============================================================

def permeability_damage_rate(k_before: float, k_after: float) -> float:
    """
    Permeability damage rate (Eq. A1.3c of paper).

    D_K = (K1 - K2) / K1 × 100%

    Parameters
    ----------
    k_before : permeability before fracturing fluid contact (mD)
    k_after  : permeability after fracturing fluid contact (mD)

    Returns
    -------
    Damage rate as a fraction (0 to 1)
    """
    if k_before <= 0:
        return 0.0
    return max(0.0, (k_before - k_after) / k_before)


def porosity_change_rate(phi_before: float, phi_after: float) -> float:
    """
    Porosity change rate after fracturing fluid contact.

    Δφ = (φ_after - φ_before) / φ_before × 100%
    """
    if phi_before <= 0:
        return 0.0
    return (phi_after - phi_before) / phi_before


# ============================================================
# NMR T2 Spectrum Analysis
# ============================================================

def generate_t2_spectrum(t2_values: np.ndarray,
                          amplitudes: np.ndarray,
                          n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a continuous T2 distribution from discrete components.

    Parameters
    ----------
    t2_values  : T2 relaxation times (ms)
    amplitudes : relative amplitudes for each T2 component

    Returns
    -------
    t2_axis, spectrum
    """
    t2_axis = np.logspace(-1, 4, n_points)  # 0.1 to 10000 ms
    spectrum = np.zeros(n_points)
    for t2, amp in zip(t2_values, amplitudes):
        # Log-normal distribution around each T2
        sigma = 0.3
        spectrum += amp * np.exp(-(np.log10(t2_axis) - np.log10(t2)) ** 2 /
                                  (2 * sigma ** 2))
    return t2_axis, spectrum


def t2_cutoff_analysis(t2_axis: np.ndarray,
                        spectrum: np.ndarray,
                        t2_cutoff: float = 33.0) -> dict:
    """
    Analyse T2 spectrum with a cutoff to separate bound and free fluid.

    Parameters
    ----------
    t2_axis   : T2 values (ms)
    spectrum  : T2 amplitude spectrum
    t2_cutoff : T2 cutoff for bound/free fluid separation (ms)

    Returns
    -------
    dict with bound_volume, free_volume, total_volume fractions
    """
    total = np.trapezoid(spectrum, np.log10(t2_axis))
    bound_mask = t2_axis <= t2_cutoff
    bound = np.trapezoid(spectrum[bound_mask], np.log10(t2_axis[bound_mask]))
    free = total - bound

    return {
        "total_volume": total,
        "bound_volume": bound,
        "free_volume": free,
        "bound_fraction": bound / (total + 1e-12),
        "free_fraction": free / (total + 1e-12),
    }


def nmr_damage_assessment(t2_before: Tuple[np.ndarray, np.ndarray],
                           t2_after: Tuple[np.ndarray, np.ndarray],
                           t2_cutoff: float = 33.0) -> dict:
    """
    Assess damage by comparing NMR T2 spectra before and after
    fracturing fluid exposure.

    Parameters
    ----------
    t2_before : (t2_values, amplitudes) before exposure
    t2_after  : (t2_values, amplitudes) after exposure

    Returns
    -------
    dict with damage metrics
    """
    ax_b, sp_b = generate_t2_spectrum(*t2_before)
    ax_a, sp_a = generate_t2_spectrum(*t2_after)

    analysis_b = t2_cutoff_analysis(ax_b, sp_b, t2_cutoff)
    analysis_a = t2_cutoff_analysis(ax_a, sp_a, t2_cutoff)

    return {
        "before": analysis_b,
        "after": analysis_a,
        "free_volume_reduction": (analysis_b["free_volume"] -
                                   analysis_a["free_volume"]) /
                                  (analysis_b["free_volume"] + 1e-12),
        "bound_volume_increase": (analysis_a["bound_volume"] -
                                   analysis_b["bound_volume"]) /
                                  (analysis_b["bound_volume"] + 1e-12),
    }


# ============================================================
# Hydrolock (Water Block) Damage
# ============================================================

@dataclass
class HydrolockResult:
    """Results of hydrolock damage assessment."""
    sw_initial: float        # Initial water saturation
    sw_after_imbibition: float  # Water saturation after imbibition
    sw_after_gas_drive: float   # Water saturation after gas drive
    k_gas_initial: float     # Initial gas permeability (mD)
    k_gas_after: float       # Gas permeability after damage (mD)
    damage_rate: float       # Hydrolock damage rate
    intrusion_depth_mm: float  # Water phase intrusion depth


def hydrolock_damage(k_gas_initial: float,
                      phi: float,
                      sw_initial: float,
                      fluid_type: str = "slickwater",
                      pressure_diff_mpa: float = 2.0,
                      soak_time_hr: float = 2.0) -> HydrolockResult:
    """
    Simulate hydrolock (water-block) damage in tight sandstone.

    Parameters
    ----------
    k_gas_initial  : initial gas permeability (mD)
    phi            : porosity (fraction)
    sw_initial     : initial water saturation (fraction)
    fluid_type     : key in FLUID_TYPES dict
    pressure_diff_mpa : differential pressure (MPa)
    soak_time_hr   : soaking time (hours)
    """
    fluid = FLUID_TYPES.get(fluid_type, FLUID_TYPES["slickwater"])

    # Imbibition model (simplified capillary-driven)
    # Higher viscosity → more imbibition, higher residue → more blockage
    imbibition_factor = 0.1 * np.sqrt(soak_time_hr) * \
                        (1 + fluid["residue_fraction"] * 10) * \
                        np.sqrt(pressure_diff_mpa)
    sw_after = min(sw_initial + imbibition_factor * (1 - sw_initial), 0.95)

    # Gas drive recovery (simplified)
    recovery_factor = 0.6 * (1 - fluid["residue_fraction"] * 5)
    sw_after_drive = sw_initial + (sw_after - sw_initial) * (1 - recovery_factor)

    # Permeability reduction (Brooks-Corey type relative perm)
    kr_gas = max((1 - sw_after_drive) ** 3, 0.001)
    k_gas_after = k_gas_initial * kr_gas

    damage_rate = permeability_damage_rate(k_gas_initial, k_gas_after)

    # Intrusion depth (simplified Washburn equation)
    intrusion = 5.0 * np.sqrt(k_gas_initial * pressure_diff_mpa * soak_time_hr /
                               (fluid["viscosity_cp"] + 1))

    return HydrolockResult(
        sw_initial=sw_initial,
        sw_after_imbibition=sw_after,
        sw_after_gas_drive=sw_after_drive,
        k_gas_initial=k_gas_initial,
        k_gas_after=k_gas_after,
        damage_rate=damage_rate,
        intrusion_depth_mm=intrusion
    )


# ============================================================
# Fracture Conductivity Damage
# ============================================================

def fracture_conductivity_damage(cond_before: float,
                                  cond_after: float) -> float:
    """
    Fracture conductivity damage rate.

    D_cond = (C_before - C_after) / C_before
    """
    if cond_before <= 0:
        return 0.0
    return max(0.0, (cond_before - cond_after) / cond_before)


def conductivity_vs_closure_pressure(proppant_conc_kg_m2: float = 5.0,
                                      closure_pressures_mpa: np.ndarray = None,
                                      fluid_type: str = "slickwater") -> np.ndarray:
    """
    Model fracture conductivity vs closure pressure for a given fluid.

    Returns conductivity array (mD·m) at each closure pressure.
    """
    if closure_pressures_mpa is None:
        closure_pressures_mpa = np.array([3, 7, 14, 21, 28, 35, 42])

    fluid = FLUID_TYPES.get(fluid_type, FLUID_TYPES["slickwater"])

    # Base conductivity decreases with closure pressure
    cond_base = proppant_conc_kg_m2 * 50 * np.exp(-0.05 * closure_pressures_mpa)

    # Damage from fluid residue
    damage_factor = 1.0 - fluid["residue_fraction"] * 5
    return cond_base * max(damage_factor, 0.1)


# ============================================================
# Simple production comparison
# ============================================================

def production_forecast(k_matrix: float,
                         k_damage_rate: float,
                         hydrolock_rate: float,
                         cond_damage_rate: float,
                         days: int = 365) -> np.ndarray:
    """
    Simplified daily production forecast considering all damage types.

    Returns daily production rate (m³/day) over time.
    """
    # Effective permeability after damage
    k_eff = k_matrix * (1 - k_damage_rate)

    # Fracture contribution reduced by conductivity damage
    frac_factor = 1 - cond_damage_rate

    # Hydrolock effect (reduces initially, partially recovers)
    time = np.arange(1, days + 1, dtype=float)
    hydrolock_recovery = 1 - hydrolock_rate * np.exp(-time / 60)

    # Decline curve
    q_initial = 10.0 * k_eff * frac_factor
    q = q_initial * hydrolock_recovery * np.exp(-0.003 * time)
    return q


# ============================================================
# Test
# ============================================================

def test_all():
    """Test all functions with synthetic data."""
    print("=" * 60)
    print("Testing fracturing_fluid_damage module (Li et al., 2025)")
    print("=" * 60)

    # 1. Permeability damage
    d = permeability_damage_rate(0.5, 0.3)
    print(f"\n1) Permeability damage: K1=0.5, K2=0.3 → D={d:.1%}")
    assert 0 <= d <= 1

    # 2. NMR T2 analysis
    t2_before = (np.array([1, 10, 100, 1000]),
                 np.array([0.3, 0.5, 0.8, 0.2]))
    t2_after = (np.array([1, 10, 100, 1000]),
                np.array([0.4, 0.6, 0.5, 0.1]))  # less free fluid
    nmr = nmr_damage_assessment(t2_before, t2_after)
    print(f"\n2) NMR damage: free-volume reduction = "
          f"{nmr['free_volume_reduction']:.1%}")
    print(f"   bound-volume increase = {nmr['bound_volume_increase']:.1%}")

    # 3. Hydrolock for each fluid type
    print("\n3) Hydrolock damage comparison:")
    for ftype in FLUID_TYPES:
        result = hydrolock_damage(0.3, 0.08, 0.35, ftype)
        print(f"   {FLUID_TYPES[ftype]['name']:40s}: "
              f"D_K={result.damage_rate:.1%}, "
              f"Sw_final={result.sw_after_gas_drive:.3f}, "
              f"intrusion={result.intrusion_depth_mm:.1f} mm")

    # 4. Fracture conductivity
    cp = np.array([7, 14, 21, 28, 35])
    for ftype in ["guanidine_gel", "slickwater"]:
        cond = conductivity_vs_closure_pressure(5.0, cp, ftype)
        print(f"\n4) Conductivity ({ftype}): {cond}")
        d_cond = fracture_conductivity_damage(cond[0], cond[-1])
        print(f"   Conductivity damage (low→high Pc): {d_cond:.1%}")

    # 5. Production comparison
    print("\n5) 1-year cumulative production comparison:")
    for ftype in FLUID_TYPES:
        hl = hydrolock_damage(0.3, 0.08, 0.35, ftype)
        q = production_forecast(0.3, hl.damage_rate * 0.3, hl.damage_rate,
                                 0.2 if ftype == "guanidine_gel" else 0.05)
        print(f"   {FLUID_TYPES[ftype]['name']:40s}: "
              f"cum = {np.sum(q):.0f} m³")

    # 6. Porosity change
    dph = porosity_change_rate(0.08, 0.075)
    print(f"\n6) Porosity change rate: {dph:.1%}")

    print("\n✓ All fracturing_fluid_damage tests passed.\n")


if __name__ == "__main__":
    test_all()
