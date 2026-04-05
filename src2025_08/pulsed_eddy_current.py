#!/usr/bin/env python3
"""
Module 7: Pulsed Eddy Current (PEC) Casing Break Detection
===========================================================
Implements ideas from:
  Jawed et al., "Pulsed Eddy Current Logging Technology for Through-
  Tubing Casing Break Detection up to the Third Tubular,"
  Petrophysics, vol. 66, no. 4, pp. 631–646, August 2025.

Key concepts:
  - Pulsed eddy-current (PEC) time-domain EM signal simulation
  - Time-transient decay analysis for multi-pipe inspection
  - Casing-break signature detection in the transient signal
  - Through-tubing evaluation (tubing shielding effect)
  - Time-lapse comparison for monitoring thermal well integrity
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TubularLayer:
    """One concentric tubular (tubing or casing)."""
    name: str
    od_inches: float
    wall_thickness_inches: float
    conductivity_S_per_m: float = 5e6
    mu_rel: float = 100.0
    has_break: bool = False
    break_depth_ft: Optional[float] = None


# ---------------------------------------------------------------------------
# 1. PEC transient signal simulation
# ---------------------------------------------------------------------------
def pec_transient(
    layers: List[TubularLayer],
    time_ms: np.ndarray,
    pulse_amplitude: float = 1.0,
    noise_std: float = 0.0,
    rng=None,
) -> np.ndarray:
    """Simulate the PEC time-transient decay signal.

    Each tubular layer contributes an exponential decay component
    whose time constant depends on the wall thickness and EM properties.
    A break in a pipe layer eliminates or reduces its contribution.

    Parameters
    ----------
    layers : list of TubularLayer
    time_ms : ndarray
        Time axis in milliseconds after the pulse.

    Returns
    -------
    ndarray — transient signal amplitude.
    """
    rng = rng or np.random.default_rng(0)
    mu0 = 4e-7 * np.pi
    signal = np.zeros_like(time_ms)

    for layer in layers:
        mu = mu0 * layer.mu_rel
        sigma = layer.conductivity_S_per_m
        t_m = layer.wall_thickness_inches * 0.0254
        # Characteristic decay time (simplified diffusion time constant)
        tau_ms = mu * sigma * t_m ** 2 * 1e3  # convert to ms
        amplitude = pulse_amplitude * t_m / (t_m + 0.01)

        if layer.has_break:
            # Break effectively eliminates this layer's contribution
            amplitude *= 0.05   # residual from partial contact
        signal += amplitude * np.exp(-time_ms / (tau_ms + 1e-12))

    if noise_std > 0:
        signal += rng.normal(0, noise_std, signal.shape)

    return signal


# ---------------------------------------------------------------------------
# 2. Multi-depth PEC log (VDL-like)
# ---------------------------------------------------------------------------
def pec_log(
    layers_by_depth: List[List[TubularLayer]],
    time_ms: np.ndarray,
    noise_std: float = 0.001,
) -> np.ndarray:
    """Generate a PEC log (depth × time channels) for a series of depths.

    Parameters
    ----------
    layers_by_depth : list of list of TubularLayer
        For each depth station, the list of tubular layers.

    Returns
    -------
    ndarray, shape (n_depths, n_time_channels)
    """
    n_depths = len(layers_by_depth)
    n_channels = len(time_ms)
    log = np.zeros((n_depths, n_channels))
    rng = np.random.default_rng(42)
    for i, layers in enumerate(layers_by_depth):
        log[i, :] = pec_transient(layers, time_ms, noise_std=noise_std, rng=rng)
    return log


# ---------------------------------------------------------------------------
# 3. Break detection via transient analysis
# ---------------------------------------------------------------------------
def detect_breaks(
    pec_log_data: np.ndarray,
    time_ms: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    threshold_sigma: float = 3.0,
) -> List[int]:
    """Detect casing breaks from a PEC log by comparing to baseline.

    A break causes a significant reduction in the late-time channels
    (which are sensitive to outer pipes).

    Returns
    -------
    list of int — depth indices with detected breaks.
    """
    if baseline is None:
        # Use the median as a pseudo-baseline
        baseline = np.median(pec_log_data, axis=0)

    # Focus on late-time channels (sensitive to outer pipes)
    n_channels = pec_log_data.shape[1]
    late_start = n_channels // 2
    late_data = pec_log_data[:, late_start:]
    late_base = baseline[late_start:]

    # Compute residual energy
    residual = np.sum((late_data - late_base) ** 2, axis=1)
    mu, sigma = np.mean(residual), np.std(residual)

    break_indices = []
    for i in range(len(residual)):
        if residual[i] > mu + threshold_sigma * sigma:
            break_indices.append(i)

    return break_indices


# ---------------------------------------------------------------------------
# 4. Time-lapse comparison
# ---------------------------------------------------------------------------
def time_lapse_difference(
    log_baseline: np.ndarray,
    log_monitor: np.ndarray,
) -> np.ndarray:
    """Compute normalised time-lapse difference between two PEC logs."""
    denom = np.abs(log_baseline) + 1e-12
    return (log_monitor - log_baseline) / denom


# ---------------------------------------------------------------------------
# 5. Estimate which pipe has the break
# ---------------------------------------------------------------------------
def identify_break_pipe(
    transient: np.ndarray,
    transient_nominal: np.ndarray,
    time_ms: np.ndarray,
    n_pipes: int = 3,
) -> int:
    """Estimate which pipe has the break based on time-domain analysis.

    The relative difference (normalised by nominal signal) grows at
    later times when outer pipes are affected, because the outer-pipe
    contribution dominates the late-time decay.

    Returns
    -------
    int — estimated pipe index (0 = innermost).
    """
    # Relative difference
    rel_diff = np.abs(transient - transient_nominal) / (np.abs(transient_nominal) + 1e-12)

    # Divide time axis into n_pipes segments (log-spaced)
    log_t = np.log10(time_ms + 1e-6)
    t_min, t_max = log_t[0], log_t[-1]
    segment_energy = np.zeros(n_pipes)

    for k in range(n_pipes):
        lo = t_min + k * (t_max - t_min) / n_pipes
        hi = t_min + (k + 1) * (t_max - t_min) / n_pipes
        mask = (log_t >= lo) & (log_t < hi)
        if np.any(mask):
            segment_energy[k] = np.mean(rel_diff[mask])

    # The segment with the highest relative difference indicates the pipe
    pipe_idx = int(np.argmax(segment_energy))
    return pipe_idx


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_all():
    time_ms = np.logspace(-2, 2, 200)  # 0.01 to 100 ms

    # Define 3-pipe completion
    tubing = TubularLayer("tubing", 2.875, 0.217)
    casing1 = TubularLayer("casing_1", 7.0, 0.317)
    casing2 = TubularLayer("casing_2", 9.625, 0.395)

    # Nominal signal
    sig_nom = pec_transient([tubing, casing1, casing2], time_ms)
    assert sig_nom.shape == time_ms.shape

    # Signal with break in casing_2
    casing2_broken = TubularLayer("casing_2", 9.625, 0.395, has_break=True)
    sig_break = pec_transient([tubing, casing1, casing2_broken], time_ms)

    # Break should reduce late-time signal
    late = time_ms > 10
    assert np.mean(sig_break[late]) < np.mean(sig_nom[late]), \
        "Break should reduce late-time signal"

    # Build a PEC log with a break at depth 50
    n_depths = 100
    layers_list = []
    for d in range(n_depths):
        if 48 <= d <= 52:
            layers_list.append([tubing, casing1, casing2_broken])
        else:
            layers_list.append([tubing, casing1, casing2])

    log_data = pec_log(layers_list, time_ms, noise_std=0.0005)
    assert log_data.shape == (n_depths, len(time_ms))

    # Detect breaks
    breaks = detect_breaks(log_data, time_ms)
    assert len(breaks) >= 1, "Should detect the break zone"
    assert any(48 <= b <= 52 for b in breaks), \
        f"Break should be detected near depth 50, got {breaks}"

    # Time-lapse
    log_baseline = pec_log(
        [[tubing, casing1, casing2]] * n_depths, time_ms)
    diff = time_lapse_difference(log_baseline, log_data)
    assert diff.shape == log_data.shape

    # Pipe identification
    pipe_id = identify_break_pipe(sig_break, sig_nom, time_ms, n_pipes=3)
    assert pipe_id >= 1, f"Break is in outer pipe but identified as pipe {pipe_id}"

    print("[PASS] pulsed_eddy_current — all tests passed")


if __name__ == "__main__":
    test_all()
