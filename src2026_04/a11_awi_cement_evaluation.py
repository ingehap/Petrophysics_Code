"""
New Equipment and Method for Evaluating Anti-Water-Invasion Ability
of Cement Slurry

Reference:
    Zhang, X., Zhang, X., Zhang, T., Li, X., Mei, C., Li, B., Jiang, M.,
    Liu, K., and Bai, Y. (2026). New Equipment and Method for Evaluating
    Anti-Water-Invasion Ability of Cement Slurry.
    Petrophysics, 67(2), 421–435. DOI: 10.30632/PJV67N2-2026a11

Implements:
  - Rapid AWI evaluation: conductivity-jump detection method
  - Water-invasion rate model during cement coagulation
  - Second-interface crack width and hydraulic conductivity model
  - Core permeability effect on water invasion
  - Cement hydration kinetics (setting-time model)
  - AWI performance scoring and comparison (conventional vs. AWI slurry)
  - Ion-loss quantification from XRD/TG analysis proxy
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Cement slurry parameters
# ---------------------------------------------------------------------------

@dataclass
class CementSlurry:
    """Cement slurry properties."""
    name:              str
    density_gcc:       float = 1.90   # slurry density, g/cm³
    water_cement_ratio:float = 0.44
    gel_strength_Pa:   float = 0.0    # static gel strength at time t
    permeability_mD:   float = 0.01   # hardened cement permeability
    is_AWI:            bool  = False   # uses anti-water-invasion material
    AWI_material_pct:  float = 0.0    # % addition of AWI additive


@dataclass
class FormationCore:
    """Core plug properties for water-invasion experiments."""
    permeability_mD:   float   # fluid permeability
    porosity_frac:     float
    length_cm:         float = 10.0
    diameter_cm:       float = 2.54


# ---------------------------------------------------------------------------
# 2. Conductivity-jump detection (rapid AWI evaluation)
# ---------------------------------------------------------------------------

def conductivity_vs_time(t_minutes: np.ndarray,
                          slurry: CementSlurry,
                          T_ambient: float = 50.0) -> np.ndarray:
    """
    Simulated electrical conductivity of cement slurry during coagulation.

    The paper's key observation: conventional slurry shows a sudden
    conductivity increase (percolation onset) when water invasion begins.
    AWI slurry shows the same jump but at a LATER time.

    Model:
        σ(t) = σ_init * exp(-k_set * t) + σ_base

    with a superimposed sigmoid "jump" at the invasion onset time t_inv.

    Parameters
    ----------
    t_minutes : Time array, min
    slurry    : CementSlurry object
    T_ambient : Ambient / wellbore temperature, °C

    Returns
    -------
    sigma_mS  : Conductivity, mS/cm
    """
    # Setting kinetics: faster at higher temperature
    k_set = 0.015 + 0.0005 * T_ambient
    sigma_init = 120.0 if not slurry.is_AWI else 115.0
    sigma_base =  20.0

    sigma = sigma_init * np.exp(-k_set * t_minutes) + sigma_base

    # Invasion onset (conductivity jump): AWI delays invasion
    if slurry.is_AWI:
        t_inv = 80.0 + slurry.AWI_material_pct * 5.0   # delayed onset
    else:
        t_inv = 50.0

    # Sigmoid jump in conductivity when water invades
    jump_amplitude = 40.0 if not slurry.is_AWI else 20.0
    sigma += jump_amplitude / (1.0 + np.exp(-(t_minutes - t_inv) / 3.0))

    return sigma


def detect_conductivity_jump(sigma: np.ndarray,
                              t: np.ndarray,
                              threshold_slope: float = 1.5) -> Optional[float]:
    """
    Detect the time of the sudden conductivity increase (AWI onset time).

    The onset is defined as the first time the slope dσ/dt exceeds a
    threshold (indicating rapid water invasion).

    Parameters
    ----------
    sigma            : Conductivity array, mS/cm
    t                : Time array, min
    threshold_slope  : dσ/dt threshold, mS/(cm·min)

    Returns
    -------
    t_onset : Time of conductivity jump, min  (None if not detected)
    """
    dslope = np.gradient(sigma, t)
    idx    = np.where(dslope > threshold_slope)[0]
    return float(t[idx[0]]) if len(idx) > 0 else None


# ---------------------------------------------------------------------------
# 3. Water invasion rate during coagulation
# ---------------------------------------------------------------------------

def water_invasion_rate(core: FormationCore,
                         slurry: CementSlurry,
                         delta_P_MPa: float,
                         t_min: float) -> float:
    """
    Volume flow rate of formation water invading cement slurry at the
    second interface, using Darcy's law.

        Q = k_core * A * ΔP / (μ * L)

    The AWI slurry reduces effective permeability at the interface.

    Parameters
    ----------
    core        : FormationCore object
    slurry      : CementSlurry object
    delta_P_MPa : Pressure differential across second interface, MPa
    t_min       : Time since cement placement, min

    Returns
    -------
    Q_mL_min : Invasion rate, mL/min
    """
    mu_water_mPas = 1.0
    A_cm2 = np.pi / 4.0 * core.diameter_cm**2

    # Effective permeability at interface: reduced by AWI material
    AWI_reduction = 1.0 - 0.08 * slurry.AWI_material_pct if slurry.is_AWI else 1.0
    k_eff = core.permeability_mD * AWI_reduction

    # Also reduce with gel strength development over time
    gel_buildup = 1.0 / (1.0 + 0.01 * t_min)   # gel reduces flow
    k_eff *= gel_buildup

    # Darcy: Q [cm³/s] = k[D] * A[cm²] * ΔP[atm] / (μ[cP] * L[cm])
    delta_P_atm = delta_P_MPa * 9.8692
    k_Darcy     = k_eff * 9.869e-4   # mD → Darcy
    Q_cm3_s     = k_Darcy * A_cm2 * delta_P_atm / (mu_water_mPas * core.length_cm)
    return Q_cm3_s * 60.0   # → mL/min


# ---------------------------------------------------------------------------
# 4. Second-interface crack width and hydraulic conductivity
# ---------------------------------------------------------------------------

def crack_width_second_interface(delta_P_MPa: float,
                                  E_cement_GPa: float = 8.0,
                                  r_borehole_cm: float = 10.8,
                                  h_annulus_cm: float = 1.5) -> float:
    """
    Estimate crack aperture at the second interface (cement-formation)
    due to pore-pressure-induced bending.

    Simplified Timoshenko beam model:
        w_crack = P * r² / (E * h²)  × scaling

    Parameters
    ----------
    delta_P_MPa    : Formation pore pressure exceeding hydrostatic, MPa
    E_cement_GPa   : Cement Young's modulus
    r_borehole_cm  : Borehole radius, cm
    h_annulus_cm   : Annular cement thickness, cm

    Returns
    -------
    w_mm : Crack width, mm
    """
    P_kPa     = delta_P_MPa * 1000.0
    E_kPa     = E_cement_GPa * 1e6
    r_m       = r_borehole_cm * 0.01
    h_m       = h_annulus_cm  * 0.01
    w_m       = (P_kPa * r_m**2) / (E_kPa * h_m**2) * 0.12
    return w_m * 1000.0   # → mm


def hydraulic_conductivity_crack(w_mm: float,
                                  rho_water: float = 1000.0,
                                  g: float = 9.81,
                                  mu: float = 1e-3) -> float:
    """
    Hydraulic conductivity of a parallel-plate crack (cubic law), m/s.

        K_h = rho * g * w³ / (12 * mu)

    Parameters
    ----------
    w_mm      : Crack aperture, mm
    rho_water : Fluid density, kg/m³
    g         : Gravity, m/s²
    mu        : Dynamic viscosity, Pa·s

    Returns
    -------
    K_h : Hydraulic conductivity, m/s
    """
    w_m = w_mm * 1e-3
    return rho_water * g * w_m**3 / (12.0 * mu)


# ---------------------------------------------------------------------------
# 5. Cement hydration and gel-strength development
# ---------------------------------------------------------------------------

def static_gel_strength(t_min: float,
                         T_C: float = 50.0,
                         is_AWI: bool = False) -> float:
    """
    Static gel strength (SGS) development model, Pa.

        SGS(t) = SGS_max * (1 - exp(-k * t))

    AWI slurry (thixotropic / microexpandable) shows higher SGS_max
    and faster development.

    Parameters
    ----------
    t_min  : Time since cement placement, minutes
    T_C    : Temperature, °C
    is_AWI : True for AWI slurry

    Returns
    -------
    SGS : Pa
    """
    SGS_max = 300.0 if is_AWI else 200.0
    k       = 0.02  * (1 + 0.01 * T_C)
    if is_AWI:
        k *= 1.3   # AWI slurry develops gel faster
    return SGS_max * (1.0 - np.exp(-k * t_min))


def transition_time(target_SGS_Pa: float = 48.0,
                     T_C: float = 50.0,
                     is_AWI: bool = False) -> float:
    """
    Time to reach target static gel strength (min).
    API defines 100→500 Pa transition as 'zero-gel' time.
    AWI slurry reaches 100 Pa sooner (shorter vulnerable window).
    """
    SGS_max = 300.0 if is_AWI else 200.0
    k       = 0.02 * (1 + 0.01 * T_C)
    if is_AWI:
        k *= 1.3
    if target_SGS_Pa >= SGS_max:
        return float("inf")
    t = -np.log(1.0 - target_SGS_Pa / SGS_max) / k
    return t


# ---------------------------------------------------------------------------
# 6. Ion-loss proxy (XRD/TG analysis simulation)
# ---------------------------------------------------------------------------

def ion_loss_fraction(core: FormationCore,
                       slurry: CementSlurry,
                       exposure_time_h: float = 24.0) -> float:
    """
    Proxy for formation-water-ion infiltration into cement (from XRD/TG
    analysis: Cl⁻ concentration in cement stone indicates ion penetration).

    Higher core permeability → more ion loss.
    AWI slurry in low-permeability core → minimal ion loss (paper result).

    Returns
    -------
    f_ion : Fraction of cement mass affected by ion exchange (0–1)
    """
    # Diffusion-limited ion transport into cement
    # f ∝ sqrt(D * t) / L   (Fickian half-space)
    D_eff = 1e-9 * core.permeability_mD / (1.0 + 10.0 * slurry.AWI_material_pct)
    t_s   = exposure_time_h * 3600.0
    L_m   = core.length_cm * 0.01
    f     = np.sqrt(D_eff * t_s) / L_m
    return min(float(f), 1.0)


# ---------------------------------------------------------------------------
# 7. AWI performance score
# ---------------------------------------------------------------------------

def awi_performance_score(slurry: CementSlurry,
                           core: FormationCore,
                           delta_P_MPa: float = 5.0,
                           T_C: float = 50.0) -> Dict:
    """
    Composite AWI performance score combining:
      - Invasion rate at 30 min
      - Transition time to gel strength 100 Pa
      - Crack hydraulic conductivity
      - Ion loss fraction

    Returns
    -------
    dict with individual metrics and composite score (0–100)
    """
    Q30    = water_invasion_rate(core, slurry, delta_P_MPa, t_min=30.0)
    t_gel  = transition_time(100.0, T_C, slurry.is_AWI)
    w_mm   = crack_width_second_interface(delta_P_MPa)
    K_h    = hydraulic_conductivity_crack(w_mm)
    ion_f  = ion_loss_fraction(core, slurry)

    # Normalise (lower invasion rate / conductivity / ion loss = better)
    s_inv  = max(0.0, 1.0 - Q30 / 5.0) * 30.0
    s_gel  = max(0.0, 1.0 - t_gel / 120.0) * 25.0
    s_cond = max(0.0, 1.0 - K_h / 1e-6) * 25.0
    s_ion  = max(0.0, 1.0 - ion_f) * 20.0
    score  = s_inv + s_gel + s_cond + s_ion

    return {
        "invasion_rate_mLmin":   round(Q30, 4),
        "gel_transition_min":    round(t_gel, 1),
        "crack_width_mm":        round(w_mm, 4),
        "hydraulic_cond_ms":     float(f"{K_h:.2e}"),
        "ion_loss_fraction":     round(ion_f, 4),
        "AWI_score":             round(score, 1),
    }


# ---------------------------------------------------------------------------
# 8. Example workflow
# ---------------------------------------------------------------------------

def example_workflow():
    print("=" * 60)
    print("Anti-Water-Invasion (AWI) Cement Slurry Evaluation")
    print("Ref: Zhang et al., Petrophysics 67(2) 2026")
    print("=" * 60)

    # Conventional cement in high-permeability core
    conv = CementSlurry("Conventional", is_AWI=False, AWI_material_pct=0.0)
    core_hi = FormationCore(permeability_mD=50.0, porosity_frac=0.22)

    # AWI cement in low-permeability core
    awi  = CementSlurry("AWI Cement",  is_AWI=True, AWI_material_pct=5.0,
                         permeability_mD=0.5)
    core_lo = FormationCore(permeability_mD=2.0, porosity_frac=0.15)

    t = np.linspace(0, 180, 500)

    # Conductivity curves
    sigma_conv = conductivity_vs_time(t, conv)
    sigma_awi  = conductivity_vs_time(t, awi)
    t_conv = detect_conductivity_jump(sigma_conv, t)
    t_awi  = detect_conductivity_jump(sigma_awi, t)

    print(f"\nConductivity jump (water-invasion onset):")
    print(f"  Conventional slurry : {t_conv:.1f} min")
    print(f"  AWI slurry          : {t_awi:.1f} min  "
          f"(delayed by {t_awi - t_conv:.1f} min)")

    # Gel strength
    print(f"\nGel strength development (at 30 min, 50°C):")
    print(f"  Conventional: {static_gel_strength(30, is_AWI=False):.1f} Pa")
    print(f"  AWI cement  : {static_gel_strength(30, is_AWI=True):.1f} Pa")

    t_gel_conv = transition_time(100.0, is_AWI=False)
    t_gel_awi  = transition_time(100.0, is_AWI=True)
    print(f"\nTime to reach 100 Pa SGS:")
    print(f"  Conventional: {t_gel_conv:.1f} min")
    print(f"  AWI cement  : {t_gel_awi:.1f} min")

    # Performance scores
    score_conv = awi_performance_score(conv, core_hi, delta_P_MPa=5.0)
    score_awi  = awi_performance_score(awi,  core_lo, delta_P_MPa=5.0)

    print(f"\nAWI Performance Scores:")
    print(f"  {'Metric':<28s} {'Conventional':>14s} {'AWI Cement':>12s}")
    print("-" * 58)
    for key in ["invasion_rate_mLmin", "gel_transition_min",
                "crack_width_mm", "hydraulic_cond_ms",
                "ion_loss_fraction", "AWI_score"]:
        c = score_conv[key]
        a = score_awi[key]
        print(f"  {key:<28s} {str(c):>14s} {str(a):>12s}")

    return score_conv, score_awi


if __name__ == "__main__":
    example_workflow()
