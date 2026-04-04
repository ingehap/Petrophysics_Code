"""
Improving Fracture Complexity of Deep Shale Based on Induced Stress
Difference: A Case Study of the Qiongzhusi Shale

Reference:
    Ci, J. (2026). Improving Fracture Complexity of Deep Shale Based on
    Induced Stress Difference: A Case Study of the Qiongzhusi Shale.
    Petrophysics, 67(2), 374–385. DOI: 10.30632/PJV67N2-2026a8

Implements:
  - 2-D plane-strain induced stress field around a hydraulic fracture
    using the displacement-discontinuity method (Crouch & Starfield 1983)
  - "Strip" distribution of σ_induced,min and "X"-shaped σ_induced,max
  - Induced stress difference field: Δσ = σH_induced - σh_induced
  - Zonal distribution pattern vs. injection volume
  - Pumping-rate and injection-volume optimisation for Well ZY-1
  - Double-fracturing design: secondary fracture deflection criterion
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Formation and fracture parameters
# ---------------------------------------------------------------------------

@dataclass
class FormationParams:
    """Geomechanical parameters for the Qiongzhusi shale."""
    E_GPa:      float = 35.0    # Young's modulus
    nu:         float = 0.22    # Poisson's ratio
    sigma_H:    float = 120.0   # max horizontal principal stress, MPa
    sigma_h:    float = 90.0    # min horizontal principal stress, MPa
    sigma_v:    float = 105.0   # vertical stress, MPa
    T_MPa:      float = 6.0     # tensile strength
    Pp_MPa:     float = 50.0    # pore pressure

    @property
    def stress_difference(self) -> float:
        return self.sigma_H - self.sigma_h


@dataclass
class FractureParams:
    """Half-length and net-pressure of the primary hydraulic fracture."""
    half_length_m: float = 200.0   # fracture half-length
    net_pressure:  float = 10.0    # net pressure (= Pf - sigma_h), MPa
    height_m:      float = 30.0    # fracture height


# ---------------------------------------------------------------------------
# 2. Displacement-discontinuity induced stress (2D plane-strain)
# ---------------------------------------------------------------------------

def induced_stress_field(xf: float, net_pressure: float,
                          X: np.ndarray, Y: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Induced stress components (σxx, σyy) around a single tensile hydraulic
    fracture of half-length xf and net pressure p, using the analytical
    solution for a pressurised crack in 2D plane strain
    (Sneddon 1946; used as the basis for the displacement-discontinuity
    model in the paper).

    The fracture lies along y = 0, -xf ≤ x ≤ xf, opened by net_pressure.

    Parameters
    ----------
    xf          : Fracture half-length, m
    net_pressure: Net pressure (Pf – sigma_h), MPa
    X, Y        : Meshgrid arrays, m

    Returns
    -------
    sigma_xx, sigma_yy : Induced stress components, MPa
    """
    p = net_pressure
    a = xf
    r1 = np.sqrt((X + a)**2 + Y**2) + 1e-6
    r2 = np.sqrt((X - a)**2 + Y**2) + 1e-6
    r  = np.sqrt(X**2 + Y**2) + 1e-6

    # Simplified far-field induced stress from Sneddon (1946):
    # σyy induced by pressurised crack opening
    theta1 = np.arctan2(Y, X + a)
    theta2 = np.arctan2(Y, X - a)

    # σxx (minimum horizontal direction stress perturbation → "strip" pattern)
    sigma_xx = -p * (a / r) * np.cos(0.5 * (theta1 + theta2)) * (
        1.0 - (a / r) * np.sin(0.5 * (theta1 - theta2)) * np.sin(1.5 * (theta1 + theta2))
    )

    # σyy (maximum horizontal direction → "X"-shaped pattern)
    sigma_yy = p * (a / r) * np.cos(0.5 * (theta1 + theta2)) * (
        1.0 + (a / r) * np.sin(0.5 * (theta1 - theta2)) * np.sin(1.5 * (theta1 + theta2))
    )

    return sigma_xx, sigma_yy


def induced_stress_difference(sigma_H_induced: np.ndarray,
                               sigma_h_induced: np.ndarray) -> np.ndarray:
    """
    Induced stress difference field Δσ_induced = σH_ind - σh_ind.
    Zones where Δσ_induced reduces the in-situ difference (σH - σh)
    are favourable for complex fracture network formation.
    """
    return sigma_H_induced - sigma_h_induced


def effective_stress_difference(formation: FormationParams,
                                 sigma_H_induced: np.ndarray,
                                 sigma_h_induced: np.ndarray) -> np.ndarray:
    """
    Total horizontal stress difference after fracturing (in-situ + induced).
    Lower values → higher likelihood of fracture complexity.
    """
    return (formation.sigma_H + sigma_H_induced) - (formation.sigma_h + sigma_h_induced)


# ---------------------------------------------------------------------------
# 3. Injection-volume scaling (Δσ weakens with volume)
# ---------------------------------------------------------------------------

def stress_diff_vs_volume(fracture: FractureParams,
                           formation: FormationParams,
                           Q_m3: np.ndarray,
                           Q_rate_m3min: float,
                           x_obs: float, y_obs: float = 10.0) -> np.ndarray:
    """
    Induced stress difference at an observation point as a function of
    injected volume (zonal weakening trend in paper Figs).

    Half-length grows as ~ Q^0.5 (PKN-type) and net-pressure decreases
    slightly with volume due to fluid efficiency.

    Parameters
    ----------
    Q_m3          : Array of injected volumes, m³
    Q_rate_m3min  : Pumping rate, m³/min (affects net pressure via friction)
    x_obs, y_obs  : Observation point coordinates, m

    Returns
    -------
    delta_sigma : Δσ_induced at observation point, MPa
    """
    k_length = 0.15   # half-length growth coefficient  (m / sqrt(m³))
    p0       = fracture.net_pressure

    results = []
    for Q in Q_m3:
        xf   = k_length * np.sqrt(Q)                    # growing half-length
        # Net pressure slightly reduces with volume (fluid efficiency)
        p_eff = p0 * (1.0 - 0.05 * np.log1p(Q / 100.0))
        p_eff = max(p_eff, 0.5)
        X     = np.array([[x_obs]])
        Y     = np.array([[y_obs]])
        sxx, syy = induced_stress_field(xf, p_eff, X, Y)
        diff  = float(abs(syy - sxx)[0, 0])
        results.append(diff)
    return np.array(results)


# ---------------------------------------------------------------------------
# 4. Pumping-rate optimisation criterion
# ---------------------------------------------------------------------------

def optimal_pumping_rate(formation: FormationParams,
                          fracture: FractureParams,
                          candidate_rates: np.ndarray,
                          target_zone_radius: float = 150.0) -> Dict:
    """
    Evaluate the induced stress difference at the target zone for a set of
    pumping rates.  Based on the paper's analysis of Well ZY-1, the
    optimum is ~18 m³/min.

    The net pressure increases with rate via wellbore friction:
        p_net(Q) ≈ p_net_base + alpha * Q^0.8   (empirical Pipe flow)

    Returns
    -------
    dict with 'rates', 'delta_sigma', 'recommended_rate'
    """
    alpha = 0.02   # friction coefficient (MPa / (m³/min)^0.8)
    delta_sigma_arr = []

    for Q in candidate_rates:
        p_net = fracture.net_pressure + alpha * (Q ** 0.8)
        X     = np.array([[target_zone_radius]])
        Y     = np.array([[20.0]])
        sxx, syy = induced_stress_field(fracture.half_length_m, p_net, X, Y)
        delta_sigma_arr.append(float(abs(syy - sxx)[0, 0]))

    delta_sigma_arr = np.array(delta_sigma_arr)
    # Optimum: Δσ_induced reduces the in-situ difference most
    best_idx = int(np.argmax(delta_sigma_arr))

    return {
        "rates":            candidate_rates,
        "delta_sigma":      delta_sigma_arr,
        "recommended_rate": float(candidate_rates[best_idx]),
        "max_delta_sigma":  float(delta_sigma_arr[best_idx]),
    }


# ---------------------------------------------------------------------------
# 5. Double-fracturing: secondary fracture deflection criterion
# ---------------------------------------------------------------------------

def secondary_fracture_condition(formation: FormationParams,
                                  sigma_H_induced: np.ndarray,
                                  sigma_h_induced: np.ndarray) -> np.ndarray:
    """
    Boolean mask: True where the secondary fracture is expected to deflect
    and potentially connect natural fractures, forming complex network.

    Criterion: (σH + σH_ind) - (σh + σh_ind) < formation.T_MPa
    i.e., the local stress difference drops below tensile strength.
    """
    eff_diff = effective_stress_difference(formation, sigma_H_induced, sigma_h_induced)
    return eff_diff < formation.T_MPa


# ---------------------------------------------------------------------------
# 6. Example workflow
# ---------------------------------------------------------------------------

def example_workflow():
    print("=" * 60)
    print("Induced Stress Difference & Fracture Complexity (Shale)")
    print("Ref: Ci, Petrophysics 67(2) 2026")
    print("=" * 60)

    formation = FormationParams()
    fracture  = FractureParams(half_length_m=200.0, net_pressure=12.0)

    # 2-D stress field
    x_arr = np.linspace(-400, 400, 80)
    y_arr = np.linspace(-300, 300, 60)
    X, Y  = np.meshgrid(x_arr, y_arr)

    sxx, syy = induced_stress_field(fracture.half_length_m,
                                     fracture.net_pressure, X, Y)
    delta_sigma = induced_stress_difference(syy, sxx)  # σH_dir - σh_dir

    print(f"\nInduced stress at fracture tip (±200 m, y=10 m):")
    for x_obs in [0, 100, 200, 300, 400]:
        xi = np.argmin(np.abs(x_arr - x_obs))
        yi = np.argmin(np.abs(y_arr - 10))
        print(f"  x={x_obs:4d} m  Δσ_induced = {delta_sigma[yi,xi]:+.2f} MPa")

    # Volume sweep (1,000 – 3,000 m³)
    volumes = np.linspace(500, 3000, 20)
    ds_vs_vol = stress_diff_vs_volume(fracture, formation, volumes,
                                       Q_rate_m3min=18.0,
                                       x_obs=150.0, y_obs=10.0)
    peak_vol = volumes[np.argmax(ds_vs_vol)]
    print(f"\nOptimal injection volume (max Δσ at 150 m): {peak_vol:.0f} m³")

    # Pumping rate optimisation
    rates = np.arange(8, 25, 2, dtype=float)
    opt   = optimal_pumping_rate(formation, fracture, rates)
    print(f"Recommended pumping rate: {opt['recommended_rate']:.0f} m³/min")

    # Secondary fracture condition map
    cond = secondary_fracture_condition(formation, syy, sxx)
    pct_complex = cond.mean() * 100
    print(f"Domain with complex fracture potential: {pct_complex:.1f}%")

    return delta_sigma, opt


if __name__ == "__main__":
    example_workflow()
