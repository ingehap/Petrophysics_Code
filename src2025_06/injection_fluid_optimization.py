#!/usr/bin/env python3
"""
Optimization Method of Injection Fluids for Energy Storage and Permeability
Enhancement in Tight Oil Reservoirs
============================================================================
Implements the methodology from:
  Xiao, Q., Shao, P., Wang, Z., Shi, L., Su, Z., Dong, N., Wang, H.,
  and Chen, G., 2025,
  "Optimization Method of Injection Fluids Based on Characteristics of
  Reservoir Fracturing for Energy Storage and Permeability Enhancement
  in Tight Oil Reservoirs,"
  Petrophysics, Vol. 66, No. 3, pp. 521–535.

Key ideas implemented:
  - Imbibition theory for fracturing-fluid invasion into tight matrix.
  - High-pressure high-temperature NMR imbibition experiment modelling.
  - Pore-space enlargement and permeability enhancement quantification.
  - Fracturing fluid system comparison and optimisation.
  - Shut-in (soaking) time optimisation for energy storage.

References:
  Liang, T. et al., 2011 (microfracture formation).
  Zhang, X. et al., 2022 (fracture pressure in pressure flooding).
  Liu, Y. et al., 2022 (energy storage mechanisms).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional


# ============================================================
# Reservoir characterisation
# ============================================================

@dataclass
class TightOilReservoir:
    """Tight-oil reservoir properties (Block A, Yanchang Formation style)."""
    permeability_md: float = 0.3       # Matrix permeability (mD)
    porosity: float = 0.08             # Porosity (fraction)
    oil_saturation: float = 0.55       # Initial oil saturation
    pore_throat_um: float = 0.5        # Mean pore-throat radius (μm)
    clay_content: float = 0.12         # Clay mineral fraction
    temperature_c: float = 60.0        # Reservoir temperature (°C)
    pressure_mpa: float = 15.0         # Reservoir pressure (MPa)
    depth_m: float = 1800.0            # Reservoir depth (m)
    oil_viscosity_cp: float = 5.0      # Oil viscosity (cP)
    ift_mn_m: float = 25.0             # Interfacial tension (mN/m)
    contact_angle_deg: float = 30.0    # Contact angle (degrees)


# ============================================================
# Fracturing fluid systems
# ============================================================

@dataclass
class FracturingFluid:
    """Fracturing fluid system properties."""
    name: str
    viscosity_cp: float          # Viscosity at reservoir temperature
    ift_reduction_factor: float  # IFT reduction relative to brine (0-1)
    clay_inhibitor: bool         # Contains clay anti-swelling agent
    surfactant_conc_pct: float   # Surfactant concentration (%)
    ph: float                    # pH value
    density_gcc: float           # Density (g/cc)


DEFAULT_FLUIDS = [
    FracturingFluid("Slickwater", 3.0, 0.6, False, 0.05, 7.0, 1.01),
    FracturingFluid("Surfactant-enhanced", 5.0, 0.3, True, 0.5, 7.5, 1.02),
    FracturingFluid("Nano-emulsion", 8.0, 0.2, True, 1.0, 7.0, 1.03),
    FracturingFluid("Active water", 4.0, 0.4, True, 0.3, 8.0, 1.01),
    FracturingFluid("Optimised fluid", 6.0, 0.15, True, 0.8, 7.2, 1.02),
]


# ============================================================
# Imbibition modelling
# ============================================================

def capillary_pressure(ift_mn_m: float, contact_angle_deg: float,
                        pore_radius_um: float) -> float:
    """
    Young-Laplace capillary pressure (MPa).

    Pc = 2·γ·cos(θ) / r
    """
    gamma = ift_mn_m * 1e-3  # N/m
    theta = np.radians(contact_angle_deg)
    r = pore_radius_um * 1e-6  # m
    pc_pa = 2 * gamma * np.cos(theta) / r
    return pc_pa * 1e-6  # MPa


def imbibition_rate(reservoir: TightOilReservoir,
                     fluid: FracturingFluid,
                     time_hr: np.ndarray) -> np.ndarray:
    """
    Imbibition volume fraction vs time using the Handy (1960) model.

    V_imb ∝ √(k·Pc·φ / μ) · √t

    Parameters
    ----------
    reservoir : reservoir properties
    fluid     : fracturing fluid properties
    time_hr   : time array (hours)

    Returns
    -------
    imbibed_volume_fraction : fraction of pore volume imbibed
    """
    ift_eff = reservoir.ift_mn_m * fluid.ift_reduction_factor
    pc = capillary_pressure(ift_eff, reservoir.contact_angle_deg,
                             reservoir.pore_throat_um)
    pc = max(pc, 0.001)  # ensure positive

    # Handy coefficient
    k_m2 = reservoir.permeability_md * 9.869e-16  # mD to m²
    mu = fluid.viscosity_cp * 1e-3  # cP to Pa·s
    phi = reservoir.porosity

    coeff = np.sqrt(2 * k_m2 * pc * 1e6 * phi / mu)  # m/√s
    time_s = time_hr * 3600

    # Normalise to pore volume
    char_length = 0.05  # characteristic matrix block half-length (m)
    v_frac = coeff * np.sqrt(time_s) / (phi * char_length)
    return np.clip(v_frac, 0, 1)


def imbibition_oil_recovery(reservoir: TightOilReservoir,
                              fluid: FracturingFluid,
                              time_hr: np.ndarray) -> np.ndarray:
    """
    Oil recovery factor from counter-current imbibition.

    RF = V_imb · So / (1 + oil-water mobility ratio adjustment)
    """
    v_imb = imbibition_rate(reservoir, fluid, time_hr)
    mob_ratio = (reservoir.oil_viscosity_cp / fluid.viscosity_cp)
    efficiency = 1.0 / (1.0 + 0.5 * mob_ratio)
    rf = v_imb * reservoir.oil_saturation * efficiency
    return np.clip(rf, 0, reservoir.oil_saturation)


# ============================================================
# NMR-based permeability enhancement assessment
# ============================================================

def nmr_pore_enlargement(t2_before: np.ndarray,
                          amp_before: np.ndarray,
                          imbibition_fraction: float,
                          dissolution_factor: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Model the effect of fracturing fluid on NMR T2 spectrum.
    Imbibition + chemical dissolution shifts pores to larger sizes.

    Parameters
    ----------
    t2_before          : T2 relaxation times (ms)
    amp_before         : amplitudes before treatment
    imbibition_fraction: fraction of pore volume contacted by fluid
    dissolution_factor : mineral dissolution rate

    Returns
    -------
    t2_after, amp_after : modified T2 spectrum
    """
    # Shift T2 to higher values (larger pores) in proportion
    # to dissolution and contact
    shift = 1.0 + dissolution_factor * imbibition_fraction
    t2_after = t2_before * shift

    # Amplitude redistribution: small pores decrease, large pores increase
    amp_after = amp_before.copy()
    mid_idx = len(amp_before) // 2
    transfer = amp_before[:mid_idx] * imbibition_fraction * 0.2
    amp_after[:mid_idx] -= transfer
    amp_after[mid_idx:mid_idx + len(transfer)] += transfer[:len(amp_after) - mid_idx]

    return t2_after, np.clip(amp_after, 0, None)


def permeability_enhancement_factor(phi_before: float, phi_after: float,
                                      k_model: str = "kozeny_carman") -> float:
    """
    Estimate permeability enhancement from porosity change.

    Using Kozeny-Carman: K ∝ φ³/(1-φ)²
    """
    if k_model == "kozeny_carman":
        k_ratio = ((phi_after / phi_before) ** 3 *
                    ((1 - phi_before) / (1 - phi_after)) ** 2)
    else:
        # Simple cubic law
        k_ratio = (phi_after / phi_before) ** 3
    return k_ratio


# ============================================================
# Shut-in (soaking) time optimisation
# ============================================================

def optimal_shutin_time(reservoir: TightOilReservoir,
                         fluid: FracturingFluid,
                         max_time_hr: float = 720,
                         n_points: int = 100) -> Tuple[float, float]:
    """
    Find the optimal shut-in time that maximises the net benefit
    (oil recovery - time cost).

    Returns (optimal_time_hr, max_recovery_factor).
    """
    times = np.linspace(1, max_time_hr, n_points)
    rf = imbibition_oil_recovery(reservoir, fluid, times)

    # Marginal recovery rate
    drf_dt = np.gradient(rf, times)

    # Optimal = where marginal recovery drops below threshold
    threshold = 0.0001  # RF per hour
    optimal_idx = np.argmax(drf_dt < threshold)
    if optimal_idx == 0:
        optimal_idx = len(times) - 1

    return times[optimal_idx], rf[optimal_idx]


# ============================================================
# Fluid system comparison and ranking
# ============================================================

def compare_fluid_systems(reservoir: TightOilReservoir,
                           fluids: List[FracturingFluid],
                           shutin_time_hr: float = 168) -> List[Dict]:
    """
    Compare multiple fracturing fluid systems for a given reservoir.

    Returns list of dicts with performance metrics, sorted best-first.
    """
    results = []
    time = np.array([shutin_time_hr])

    for fluid in fluids:
        rf = imbibition_oil_recovery(reservoir, fluid, time)[0]
        v_imb = imbibition_rate(reservoir, fluid, time)[0]

        # Estimate permeability enhancement
        phi_after = reservoir.porosity * (1 + 0.05 * v_imb)  # 5% max
        k_enhance = permeability_enhancement_factor(reservoir.porosity, phi_after)

        # Clay damage penalty
        clay_penalty = 0.0
        if not fluid.clay_inhibitor and reservoir.clay_content > 0.05:
            clay_penalty = reservoir.clay_content * 0.5

        # Score: recovery + enhancement - clay damage
        score = rf + 0.1 * (k_enhance - 1) - clay_penalty

        results.append({
            "fluid": fluid.name,
            "recovery_factor": rf,
            "imbibition_fraction": v_imb,
            "k_enhancement": k_enhance,
            "clay_penalty": clay_penalty,
            "score": score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ============================================================
# Energy storage assessment
# ============================================================

def energy_storage_pressure(reservoir: TightOilReservoir,
                              injection_volume_m3: float,
                              fracture_half_length_m: float = 100,
                              fracture_height_m: float = 20) -> float:
    """
    Estimate pressure build-up from fracturing fluid injection
    and soaking (energy storage).

    ΔP ≈ V_inj / (c_t · V_stim)

    Parameters
    ----------
    injection_volume_m3     : total injected fluid volume
    fracture_half_length_m  : fracture half-length
    fracture_height_m       : fracture height

    Returns
    -------
    Pressure increase (MPa)
    """
    # Stimulated volume
    penetration_m = 10.0  # matrix penetration depth
    v_stim = 2 * fracture_half_length_m * fracture_height_m * penetration_m
    v_pore = v_stim * reservoir.porosity

    # Total compressibility (oil + rock + water)
    c_t = 1e-4  # 1/MPa (typical tight oil)

    delta_p = injection_volume_m3 / (c_t * v_pore + 1e-6)
    return min(delta_p, 30.0)  # cap at 30 MPa


# ============================================================
# Test
# ============================================================

def test_all():
    """Test all functions with synthetic data."""
    print("=" * 60)
    print("Testing injection_fluid_optimization module (Xiao et al., 2025)")
    print("=" * 60)

    res = TightOilReservoir()
    print(f"\nReservoir: K={res.permeability_md} mD, φ={res.porosity}, "
          f"So={res.oil_saturation}, T={res.temperature_c}°C")

    # 1. Capillary pressure
    pc = capillary_pressure(res.ift_mn_m, res.contact_angle_deg,
                             res.pore_throat_um)
    print(f"\n1) Capillary pressure: {pc:.3f} MPa")
    assert pc > 0

    # 2. Imbibition curves
    times = np.array([1, 6, 24, 48, 168, 336, 720])
    print("\n2) Imbibition comparison (V_imb fraction):")
    for fluid in DEFAULT_FLUIDS[:3]:
        v_imb = imbibition_rate(res, fluid, times)
        print(f"   {fluid.name:25s}: " +
              " ".join(f"{v:.3f}" for v in v_imb))

    # 3. Oil recovery
    print("\n3) Oil recovery factor at 168 hr:")
    for fluid in DEFAULT_FLUIDS:
        rf = imbibition_oil_recovery(res, fluid, np.array([168.0]))
        print(f"   {fluid.name:25s}: RF = {rf[0]:.4f}")

    # 4. Optimal shut-in time
    print("\n4) Optimal shut-in times:")
    for fluid in DEFAULT_FLUIDS[:3]:
        t_opt, rf_opt = optimal_shutin_time(res, fluid)
        print(f"   {fluid.name:25s}: t_opt = {t_opt:.0f} hr, RF = {rf_opt:.4f}")

    # 5. NMR pore enlargement
    t2 = np.array([0.5, 2, 10, 50, 200])
    amp = np.array([0.3, 0.5, 0.4, 0.2, 0.1])
    t2_a, amp_a = nmr_pore_enlargement(t2, amp, 0.5, 0.15)
    print(f"\n5) NMR T2 shift: {t2} → {t2_a}")

    # 6. Permeability enhancement
    k_enh = permeability_enhancement_factor(0.08, 0.085)
    print(f"\n6) K enhancement (φ: 0.08→0.085): {k_enh:.3f}x")
    assert k_enh > 1.0

    # 7. Fluid comparison and ranking
    print("\n7) Fluid system ranking:")
    ranking = compare_fluid_systems(res, DEFAULT_FLUIDS, shutin_time_hr=168)
    for i, r in enumerate(ranking):
        print(f"   #{i+1} {r['fluid']:25s}: score={r['score']:.4f}, "
              f"RF={r['recovery_factor']:.4f}, K_enh={r['k_enhancement']:.3f}")

    # 8. Energy storage
    dp = energy_storage_pressure(res, injection_volume_m3=500)
    print(f"\n8) Energy storage ΔP from 500 m³ injection: {dp:.2f} MPa")

    print("\n✓ All injection_fluid_optimization tests passed.\n")


if __name__ == "__main__":
    test_all()
