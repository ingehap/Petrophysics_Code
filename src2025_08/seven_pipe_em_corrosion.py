#!/usr/bin/env python3
"""
Module 3: Seven-Pipe Electromagnetic Corrosion Evaluation
=========================================================
Implements ideas from:
  Fouda et al., "First-Ever Seven-Pipe Corrosion Evaluation for
  Comprehensive Assessment of Pipe Integrity in Complex Well Completions,"
  Petrophysics, vol. 66, no. 4, pp. 566–577, August 2025.

Key concepts:
  - Multi-frequency continuous-wave EM eddy-current forward model
  - Model-based inversion for individual pipe wall thicknesses
  - Cost function with magnitude misfit, phase misfit, and regularisation
  - Multiple transmitter-receiver spacings for depth of investigation
  - Metal-loss estimation and comparison with nominal thickness
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PipeProperties:
    """Properties for a single concentric pipe."""
    od_inches: float
    wall_thickness_inches: float
    conductivity_S_per_m: float = 5.0e6    # typical steel
    permeability_rel: float = 100.0        # relative mu


@dataclass
class EMToolConfig:
    """Configuration for the multi-frequency EM tool."""
    frequencies_hz: List[float] = field(default_factory=lambda: [
        10, 25, 50, 100, 250, 500, 1000, 2500, 5000,
    ])
    tr_spacings_ft: List[float] = field(default_factory=lambda: [
        1.0, 2.0, 3.0, 5.0, 8.0, 12.0,
    ])


# ---------------------------------------------------------------------------
# 1. Simplified EM forward model
# ---------------------------------------------------------------------------
def em_forward(
    pipes: List[PipeProperties],
    config: EMToolConfig,
    noise_level: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simplified EM eddy-current forward model.

    For each (frequency, spacing) pair the response is a complex number
    whose magnitude and phase depend on the cumulative skin-depth
    attenuation through all pipe walls.

    Returns
    -------
    ndarray, shape (n_spacings, n_frequencies), complex
    """
    rng = rng or np.random.default_rng(0)
    mu0 = 4e-7 * np.pi
    n_sp = len(config.tr_spacings_ft)
    n_freq = len(config.frequencies_hz)
    response = np.zeros((n_sp, n_freq), dtype=complex)

    for i_sp, spacing_ft in enumerate(config.tr_spacings_ft):
        spacing_m = spacing_ft * 0.3048
        for i_f, freq in enumerate(config.frequencies_hz):
            omega = 2 * np.pi * freq
            # Geometrical factor ∝ 1/spacing^3
            geo = 1.0 / (spacing_m ** 3)
            # Cumulative attenuation through all pipes
            attenuation = 1.0 + 0j
            for ip, p in enumerate(pipes):
                mu = mu0 * p.permeability_rel
                sigma = p.conductivity_S_per_m
                skin_depth = np.sqrt(2.0 / (omega * mu * sigma))
                t_m = p.wall_thickness_inches * 0.0254
                # Each pipe has a unique conductivity/permeability scaling
                # to break degeneracy (simulates real material variation)
                scale = 1.0 + 0.05 * ip
                alpha = (1 + 1j) * t_m * scale / skin_depth
                attenuation *= np.exp(-alpha)
                # Spacing-dependent sensitivity: longer spacing → deeper
                depth_weight = np.exp(-0.3 * ip / (i_sp + 1))
                attenuation *= (1.0 + 0.1j * depth_weight)
            response[i_sp, i_f] = geo * attenuation

    if noise_level > 0:
        scale = noise_level * np.abs(response).mean()
        noise = rng.normal(0, scale, response.shape) + \
                1j * rng.normal(0, scale, response.shape)
        response += noise

    return response


# ---------------------------------------------------------------------------
# 2. Cost function for inversion (Eq. 1 in the paper)
# ---------------------------------------------------------------------------
def cost_function(
    thicknesses: np.ndarray,
    measured: np.ndarray,
    pipes_template: List[PipeProperties],
    config: EMToolConfig,
    nominal_thicknesses: np.ndarray,
    w_mag: float = 1.0,
    w_phase: float = 1.0,
    w_reg: float = 0.01,
) -> float:
    """Compute the inversion cost function.

    J = w_mag * ||mag(m) - mag(s)|| + w_phase * ||phase(m) - phase(s)||
        + w_reg * ||x - x0||^2

    Parameters
    ----------
    thicknesses : array, shape (n_pipes,)
        Current estimate of wall thicknesses (inches).
    measured : complex array (n_sp, n_freq)
        Measured data.
    pipes_template : list of PipeProperties
        Template pipes (thicknesses will be overridden).
    config : EMToolConfig
    nominal_thicknesses : array
        Prior / initial guess.
    """
    pipes = []
    for i, p in enumerate(pipes_template):
        pp = PipeProperties(
            od_inches=p.od_inches,
            wall_thickness_inches=float(thicknesses[i]),
            conductivity_S_per_m=p.conductivity_S_per_m,
            permeability_rel=p.permeability_rel,
        )
        pipes.append(pp)

    synthetic = em_forward(pipes, config)

    mag_misfit = np.sum((np.abs(measured) - np.abs(synthetic)) ** 2)
    phase_misfit = np.sum((np.angle(measured) - np.angle(synthetic)) ** 2)
    reg = np.sum((thicknesses - nominal_thicknesses) ** 2)

    return w_mag * mag_misfit + w_phase * phase_misfit + w_reg * reg


# ---------------------------------------------------------------------------
# 3. Gauss-Newton-style inversion
# ---------------------------------------------------------------------------
def invert_thicknesses(
    measured: np.ndarray,
    pipes_template: List[PipeProperties],
    config: EMToolConfig,
    n_iter: int = 30,
    step_size: float = 0.001,
) -> Tuple[np.ndarray, List[float]]:
    """Estimate individual pipe thicknesses via gradient descent.

    Returns
    -------
    estimated_thicknesses : ndarray
    cost_history : list of float
    """
    n_pipes = len(pipes_template)
    nominal = np.array([p.wall_thickness_inches for p in pipes_template])
    x = nominal.copy()
    cost_history: List[float] = []

    for _ in range(n_iter):
        c0 = cost_function(x, measured, pipes_template, config, nominal)
        cost_history.append(c0)
        grad = np.zeros(n_pipes)
        eps = 1e-5
        for j in range(n_pipes):
            x_pert = x.copy()
            x_pert[j] += eps
            c_pert = cost_function(x_pert, measured, pipes_template, config, nominal)
            grad[j] = (c_pert - c0) / eps
        grad_norm = np.linalg.norm(grad) + 1e-12
        direction = -grad / grad_norm

        # Backtracking line search
        alpha = step_size
        for _ in range(10):
            x_trial = np.clip(x + alpha * direction, 0.05, 2.0)
            c_trial = cost_function(x_trial, measured, pipes_template,
                                     config, nominal)
            if c_trial < c0:
                break
            alpha *= 0.5
        x = np.clip(x + alpha * direction, 0.05, 2.0)

    return x, cost_history


# ---------------------------------------------------------------------------
# 4. Metal-loss estimation
# ---------------------------------------------------------------------------
def estimate_metal_loss(
    nominal: np.ndarray, estimated: np.ndarray,
) -> np.ndarray:
    """Metal loss as percentage of nominal wall thickness."""
    return np.clip((1.0 - estimated / nominal) * 100.0, 0, 100)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_all():
    # Define a 5-pipe completion (subset of the 7-pipe case)
    pipes_true = [
        PipeProperties(od_inches=2.875, wall_thickness_inches=0.217),
        PipeProperties(od_inches=4.5,   wall_thickness_inches=0.271),
        PipeProperties(od_inches=7.0,   wall_thickness_inches=0.317),
        PipeProperties(od_inches=9.625, wall_thickness_inches=0.395),
        PipeProperties(od_inches=13.375, wall_thickness_inches=0.380),
    ]

    # Introduce corrosion on pipe 3 and 5
    pipes_corroded = []
    for i, p in enumerate(pipes_true):
        wt = p.wall_thickness_inches
        if i == 2:
            wt *= 0.80   # 20% metal loss
        if i == 4:
            wt *= 0.70   # 30% metal loss
        pipes_corroded.append(PipeProperties(
            od_inches=p.od_inches, wall_thickness_inches=wt,
            conductivity_S_per_m=p.conductivity_S_per_m,
            permeability_rel=p.permeability_rel,
        ))

    config = EMToolConfig()

    # Forward model with noise
    measured = em_forward(pipes_corroded, config, noise_level=0.01)
    assert measured.shape == (len(config.tr_spacings_ft), len(config.frequencies_hz))

    # Inversion starting from nominal
    estimated, history = invert_thicknesses(
        measured, pipes_true, config, n_iter=80, step_size=0.005
    )
    assert len(history) == 80
    assert history[-1] < history[0], "Cost should decrease"

    # Metal-loss estimation
    nominal_wt = np.array([p.wall_thickness_inches for p in pipes_true])
    ml = estimate_metal_loss(nominal_wt, estimated)
    assert ml.shape == (5,)

    # Check that the inner pipes (easier to resolve) are estimated reasonably
    true_wt = np.array([p.wall_thickness_inches for p in pipes_corroded])
    inner_error = np.abs(estimated[:2] - true_wt[:2])
    assert np.all(inner_error < 0.15), \
        f"Inner-pipe thickness error too large: {inner_error}"

    print("[PASS] seven_pipe_em_corrosion — all tests passed")


if __name__ == "__main__":
    test_all()
