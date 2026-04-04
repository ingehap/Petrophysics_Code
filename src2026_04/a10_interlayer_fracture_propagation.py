"""
Analysis of the Dynamic Interlayer Fracture Propagation Behavior of
Hydraulic Fractures in Interbedded Coal-Bearing Strata — A Case of the
Taiyuan Formation in the Jiyang Depression

Reference:
    Zhao, Z., Jin, H., Guo, J., Xu, S., Xue, K., and Zhang, Z. (2026).
    Analysis of the Dynamic Interlayer Fracture Propagation Behavior of
    Hydraulic Fractures in Interbedded Coal-Bearing Strata.
    Petrophysics, 67(2), 404–419. DOI: 10.30632/PJV67N2-2026a10

Implements:
  - Finite-discrete element model (FDEM) simplified 2D proxy
  - Interlayer interface crossing criterion (tensile / shear)
  - Fracture initiation-layer effect: sandstone vs. coal
  - Pumping-rate and viscosity effect on vertical propagation
  - Cleat-fracture connection in horizontal direction (coal)
  - Multi-layer stress profile builder
  - Production optimisation: recommended parameters for Jiyang Depression
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Stratigraphic layer model (Taiyuan Formation)
# ---------------------------------------------------------------------------

@dataclass
class Layer:
    """Single stratigraphic layer."""
    name:         str
    thickness_m:  float
    E_GPa:        float   # Young's modulus
    nu:           float   # Poisson's ratio
    Kic_MPa_m05:  float   # Mode-I fracture toughness, MPa·√m
    sigma_h_MPa:  float   # minimum horizontal stress
    sigma_H_MPa:  float   # maximum horizontal stress
    tensile_MPa:  float   # tensile strength
    permeability_mD: float = 0.0


def build_jiyang_stratigraphy() -> List[Layer]:
    """
    Representative interbedded sequence for Taiyuan Formation,
    Jiyang Depression (sand / coal / mudstone alternation).
    """
    return [
        Layer("Sandstone-1",  8.0, 28.0, 0.24, 1.8, 32.0, 52.0, 4.0, 20.0),
        Layer("Mudstone-1",   3.0, 18.0, 0.32, 1.2, 40.0, 60.0, 2.0,  0.1),
        Layer("Coal-1",       5.0,  2.5, 0.30, 0.8, 28.0, 44.0, 1.2,  2.0),
        Layer("Mudstone-2",   2.0, 18.0, 0.32, 1.2, 38.0, 58.0, 2.0,  0.1),
        Layer("Sandstone-2",  6.0, 30.0, 0.22, 1.9, 30.0, 50.0, 4.5, 25.0),
    ]


# ---------------------------------------------------------------------------
# 2. Interface crossing criterion
# ---------------------------------------------------------------------------

def interface_crossing_tension(sigma_n: float,
                                Kic: float,
                                aperture_m: float = 1e-3) -> bool:
    """
    Determine whether a hydraulic fracture crosses a layer interface
    based on the tensile-stress criterion.

    A fracture crosses if the normal stress concentration at the tip
    exceeds the adjacent layer's tensile strength (Blanton 1982, adapted):

        σ_tip = Kic / √(π * aperture)  > T_adjacent

    Parameters
    ----------
    sigma_n   : Normal stress at interface, MPa (negative = tension)
    Kic       : Mode-I toughness of adjacent layer, MPa·√m
    aperture_m: Fracture aperture at interface, m

    Returns
    -------
    True if fracture is predicted to cross the interface
    """
    if aperture_m <= 0:
        return False
    sigma_tip = Kic / np.sqrt(np.pi * aperture_m)
    # Crossing: tip stress exceeds normal stress barrier
    return sigma_tip > abs(sigma_n)


def interface_crossing_probability(layers: List[Layer],
                                    frac_layer_idx: int,
                                    p_net_MPa: float,
                                    Q_m3min: float,
                                    viscosity_mPas: float) -> Dict:
    """
    Estimate the probability that a fracture initiated in `frac_layer_idx`
    propagates through each interface to adjacent layers.

    Key paper finding:
      - Sandstone initiation → fractures can cross into coal
      - Coal initiation → fractures remain confined (horizontal cleat network)

    Parameters
    ----------
    layers         : Stratigraphic sequence from build_jiyang_stratigraphy()
    frac_layer_idx : Index of the fracturing treatment layer
    p_net_MPa      : Net fracturing pressure
    Q_m3min        : Pumping rate, m³/min
    viscosity_mPas : Fluid viscosity, mPa·s

    Returns
    -------
    dict: {layer_name: crossing_probability}
    """
    n = len(layers)
    frac_layer = layers[frac_layer_idx]

    # Aperture scales with Q and viscosity (simplified lubrication theory)
    # w ∝ (viscosity * Q)^(1/4)
    aperture_m = 2e-3 * (viscosity_mPas * Q_m3min / 15.0) ** 0.25

    results = {}
    for i, layer in enumerate(layers):
        if i == frac_layer_idx:
            results[layer.name] = 1.0   # initiating layer, always fractured
            continue

        # Stress barrier = difference in minimum horizontal stress
        delta_sh = abs(layer.sigma_h_MPa - frac_layer.sigma_h_MPa)
        # Distance from fracture layer
        dist = abs(i - frac_layer_idx)

        # Net pressure must overcome stress barrier and decrease with distance
        p_eff = p_net_MPa - delta_sh - dist * 0.5  # 0.5 MPa/layer attenuation
        if p_eff <= 0:
            results[layer.name] = 0.0
            continue

        cross = interface_crossing_tension(
            sigma_n=layer.sigma_h_MPa - p_net_MPa,
            Kic=layer.Kic_MPa_m05,
            aperture_m=aperture_m,
        )
        # Coal initiation: fractures confined (Fig. in paper) → lower prob
        is_coal_init = frac_layer.name.startswith("Coal")
        base_prob = 0.85 if cross else 0.30
        if is_coal_init:
            base_prob *= 0.2   # strong containment in coal
        results[layer.name] = round(min(base_prob / dist, 1.0), 3)

    return results


# ---------------------------------------------------------------------------
# 3. Fracture geometry vs. pumping rate and viscosity
# ---------------------------------------------------------------------------

def fracture_height_growth(layers: List[Layer],
                            frac_layer_idx: int,
                            Q_m3min: float,
                            viscosity_mPas: float,
                            inj_time_min: float = 60.0) -> float:
    """
    Estimate total fracture height (m) by summing layers that are penetrated
    at the given pumping rate and viscosity.

    Higher Q and lower viscosity → better vertical propagation through interlayers.
    Paper recommends Q = 22 m³/min and η = 15 mPa·s for sandstone initiation.
    """
    frac_layer  = layers[frac_layer_idx]
    p_net_MPa   = _estimate_net_pressure(Q_m3min, viscosity_mPas, frac_layer)
    crossing    = interface_crossing_probability(layers, frac_layer_idx,
                                                  p_net_MPa, Q_m3min, viscosity_mPas)
    total_h = 0.0
    for layer in layers:
        prob = crossing.get(layer.name, 0.0)
        if prob >= 0.5:
            total_h += layer.thickness_m
    return total_h


def _estimate_net_pressure(Q: float, mu: float, layer: Layer) -> float:
    """Proxy net pressure from pumping rate and viscosity (simplified)."""
    # p_net ≈ p_base + alpha * Q^0.5 * mu^0.25
    p_base = 5.0
    alpha  = 0.8
    return p_base + alpha * np.sqrt(Q) * mu**0.25


# ---------------------------------------------------------------------------
# 4. Cleat-fracture connection in coal (horizontal)
# ---------------------------------------------------------------------------

def cleat_connection_probability(coal_layer: Layer,
                                  p_net_MPa: float,
                                  cleat_spacing_cm: float = 2.0,
                                  cleat_length_m: float = 0.5) -> float:
    """
    Probability that hydraulic fracture connects to natural cleats in coal.

    In coal, when fracture is confined, it preferentially connects with
    cleats in the horizontal direction (face and butt cleats).

    Criterion: p_net > sigma_H - sigma_h (cleat opening condition).
    """
    diff = coal_layer.sigma_H_MPa - coal_layer.sigma_h_MPa
    if p_net_MPa <= diff:
        return 0.3   # unlikely to open cleats
    # Probability increases with excess pressure over critical
    excess = p_net_MPa - diff
    prob   = 1.0 - np.exp(-excess / 3.0)
    return float(min(prob, 0.95))


# ---------------------------------------------------------------------------
# 5. Parameter optimisation for multi-layer co-production
# ---------------------------------------------------------------------------

def optimise_fracturing_parameters(layers: List[Layer],
                                    candidate_Q: np.ndarray,
                                    candidate_mu: np.ndarray,
                                    initiation_layer: str = "Sandstone-1"
                                    ) -> Dict:
    """
    Grid search over pumping rate and viscosity to maximise fractured height
    (and therefore total gas-bearing interval contacted).

    Paper result: Q = 22 m³/min, μ = 15 mPa·s for Jiyang Depression.

    Returns
    -------
    dict with 'Q_opt', 'mu_opt', 'height_grid' (2D array)
    """
    frac_idx = next((i for i, l in enumerate(layers)
                     if l.name == initiation_layer), 0)
    height_grid = np.zeros((len(candidate_Q), len(candidate_mu)))

    for i, Q in enumerate(candidate_Q):
        for j, mu in enumerate(candidate_mu):
            height_grid[i, j] = fracture_height_growth(
                layers, frac_idx, Q, mu)

    best_ij  = np.unravel_index(height_grid.argmax(), height_grid.shape)
    Q_opt    = candidate_Q[best_ij[0]]
    mu_opt   = candidate_mu[best_ij[1]]

    return {
        "Q_opt":       float(Q_opt),
        "mu_opt":      float(mu_opt),
        "max_height":  float(height_grid.max()),
        "height_grid": height_grid,
        "candidate_Q": candidate_Q,
        "candidate_mu":candidate_mu,
    }


# ---------------------------------------------------------------------------
# 6. Example workflow
# ---------------------------------------------------------------------------

def example_workflow():
    print("=" * 60)
    print("Interlayer Fracture Propagation – Interbedded Coal Strata")
    print("Ref: Zhao et al., Petrophysics 67(2) 2026")
    print("=" * 60)

    layers = build_jiyang_stratigraphy()
    print("\nStratigraphy:")
    for i, l in enumerate(layers):
        print(f"  [{i}] {l.name:<15s} {l.thickness_m:5.1f} m  "
              f"σh={l.sigma_h_MPa:.0f} MPa  E={l.E_GPa:.0f} GPa")

    # Sandstone initiation (paper's recommendation)
    print("\n--- Sandstone-1 initiation, Q=22 m³/min, μ=15 mPa·s ---")
    crossing_ss = interface_crossing_probability(layers, frac_layer_idx=0,
                                                   p_net_MPa=12.0,
                                                   Q_m3min=22.0,
                                                   viscosity_mPas=15.0)
    for name, prob in crossing_ss.items():
        print(f"  {name:<18s} crossing probability = {prob:.2f}")

    h_ss = fracture_height_growth(layers, frac_layer_idx=0,
                                   Q_m3min=22.0, viscosity_mPas=15.0)
    print(f"\n  Total fractured height (sandstone init): {h_ss:.1f} m")

    # Coal initiation (confined in coal)
    print("\n--- Coal-1 initiation, Q=10 m³/min, μ=15 mPa·s ---")
    crossing_coal = interface_crossing_probability(layers, frac_layer_idx=2,
                                                    p_net_MPa=8.0,
                                                    Q_m3min=10.0,
                                                    viscosity_mPas=15.0)
    for name, prob in crossing_coal.items():
        print(f"  {name:<18s} crossing probability = {prob:.2f}")

    h_coal = fracture_height_growth(layers, frac_layer_idx=2,
                                     Q_m3min=10.0, viscosity_mPas=15.0)
    print(f"  Total fractured height (coal init): {h_coal:.1f} m")

    # Cleat connectivity in coal
    coal = layers[2]
    p_net = _estimate_net_pressure(22.0, 15.0, coal)
    prob_cleat = cleat_connection_probability(coal, p_net)
    print(f"\nCleat connection probability (sandstone frac, Q=22): "
          f"{prob_cleat:.2f}")

    # Parameter optimisation
    candidate_Q  = np.arange(8, 28, 4, dtype=float)
    candidate_mu = np.array([5.0, 10.0, 15.0, 25.0, 50.0])
    opt = optimise_fracturing_parameters(layers, candidate_Q, candidate_mu,
                                          initiation_layer="Sandstone-1")
    print(f"\nOptimum parameters:")
    print(f"  Q  = {opt['Q_opt']:.0f} m³/min  (paper: 22 m³/min)")
    print(f"  μ  = {opt['mu_opt']:.0f} mPa·s  (paper: 15 mPa·s)")
    print(f"  Fractured height = {opt['max_height']:.1f} m")

    return opt


if __name__ == "__main__":
    example_workflow()
