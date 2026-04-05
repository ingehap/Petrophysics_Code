#!/usr/bin/env python3
"""
Module 2: Fiber-Optic Sensing for Well Integrity Diagnosis
==========================================================
Implements ideas from:
  Bazaid et al., "Pioneering Well Logging: The Role of Fiber Optics
  in Modern Monitoring for Well Integrity Diagnosis,"
  Petrophysics, vol. 66, no. 4, pp. 555–565, August 2025.

Key concepts:
  - Distributed Temperature Sensing (DTS) simulation & leak detection
  - Distributed Acoustic Sensing (DAS) simulation & event detection
  - SNR improvement via spatial/temporal stacking
  - Comparison with single-point conventional measurements
  - Diagnostic-time reduction estimation (≈85 % faster)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class LeakEvent:
    """Detected leak from DTS or DAS analysis."""
    depth_ft: float
    severity: float       # 0–1 normalised
    method: str           # 'DTS' or 'DAS'
    description: str = ""


# ---------------------------------------------------------------------------
# 1. DTS simulation
# ---------------------------------------------------------------------------
def simulate_dts(
    depth_ft: np.ndarray,
    geothermal_gradient: float = 0.015,   # °F per ft
    surface_temp: float = 70.0,           # °F
    leak_depths: Optional[List[float]] = None,
    leak_amplitudes: Optional[List[float]] = None,
    noise_std: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a synthetic DTS temperature profile with optional leaks.

    Parameters
    ----------
    depth_ft : ndarray
        Depth array (ft).
    geothermal_gradient : float
        Temperature increase per foot of depth (°F/ft).
    surface_temp : float
        Surface temperature (°F).
    leak_depths : list of float, optional
        Depths at which leaks inject/remove heat.
    leak_amplitudes : list of float, optional
        Temperature anomaly (°F) at each leak location.
    noise_std : float
        Gaussian noise standard deviation (°F).
    rng : numpy Generator, optional

    Returns
    -------
    ndarray
        Simulated DTS temperature at each depth.
    """
    rng = rng or np.random.default_rng(0)
    temp = surface_temp + geothermal_gradient * depth_ft
    if leak_depths and leak_amplitudes:
        for ld, la in zip(leak_depths, leak_amplitudes):
            # Gaussian anomaly centred at leak depth
            temp += la * np.exp(-0.5 * ((depth_ft - ld) / 20.0) ** 2)
    temp += rng.normal(0, noise_std, size=depth_ft.shape)
    return temp


# ---------------------------------------------------------------------------
# 2. DAS simulation
# ---------------------------------------------------------------------------
def simulate_das(
    depth_ft: np.ndarray,
    time_s: np.ndarray,
    event_depths: Optional[List[float]] = None,
    event_times: Optional[List[float]] = None,
    event_amplitudes: Optional[List[float]] = None,
    noise_std: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a synthetic DAS waterfall (depth × time) image.

    Parameters
    ----------
    depth_ft, time_s : ndarray
        1-D depth and time axes.
    event_depths, event_times, event_amplitudes : lists
        Each acoustic event is a localised Gaussian blob.
    noise_std : float

    Returns
    -------
    ndarray, shape (len(depth_ft), len(time_s))
        Simulated DAS amplitude.
    """
    rng = rng or np.random.default_rng(1)
    D, T = np.meshgrid(depth_ft, time_s, indexing="ij")
    das = rng.normal(0, noise_std, size=D.shape)

    if event_depths and event_times and event_amplitudes:
        for ed, et, ea in zip(event_depths, event_times, event_amplitudes):
            das += ea * np.exp(
                -0.5 * (((D - ed) / 15.0) ** 2 + ((T - et) / 0.5) ** 2)
            )
    return das


# ---------------------------------------------------------------------------
# 3. SNR improvement via stacking
# ---------------------------------------------------------------------------
def temporal_stack(das: np.ndarray, n_stack: int = 10) -> np.ndarray:
    """Stack DAS traces in the time dimension to improve SNR.

    The SNR improves approximately as sqrt(n_stack).
    """
    n_depth, n_time = das.shape
    n_out = n_time // n_stack
    stacked = np.zeros((n_depth, n_out))
    for i in range(n_out):
        stacked[:, i] = das[:, i * n_stack:(i + 1) * n_stack].mean(axis=1)
    return stacked


def compute_snr(signal: np.ndarray, noise_std: float) -> float:
    """Compute signal-to-noise ratio in dB."""
    sig_power = np.mean(signal ** 2)
    noise_power = noise_std ** 2
    if noise_power == 0:
        return float('inf')
    return 10.0 * np.log10(sig_power / noise_power)


# ---------------------------------------------------------------------------
# 4. Leak detection from DTS
# ---------------------------------------------------------------------------
def detect_leaks_dts(
    depth_ft: np.ndarray,
    temperature: np.ndarray,
    geothermal_gradient: float = 0.015,
    surface_temp: float = 70.0,
    threshold_sigma: float = 3.0,
) -> List[LeakEvent]:
    """Detect leaks as temperature anomalies relative to expected geotherm.

    Parameters
    ----------
    threshold_sigma : float
        Number of standard deviations for anomaly detection.

    Returns
    -------
    list of LeakEvent
    """
    expected = surface_temp + geothermal_gradient * depth_ft
    residual = temperature - expected
    sigma = np.std(residual)
    anomalies = np.abs(residual) > threshold_sigma * sigma

    events: List[LeakEvent] = []
    # group contiguous anomaly regions
    in_anomaly = False
    start = 0
    for i in range(len(anomalies)):
        if anomalies[i] and not in_anomaly:
            in_anomaly = True
            start = i
        elif not anomalies[i] and in_anomaly:
            in_anomaly = False
            idx_max = start + np.argmax(np.abs(residual[start:i]))
            events.append(LeakEvent(
                depth_ft=float(depth_ft[idx_max]),
                severity=float(np.abs(residual[idx_max]) / sigma),
                method="DTS",
                description=f"Temp anomaly {residual[idx_max]:+.2f} °F",
            ))
    return events


# ---------------------------------------------------------------------------
# 5. Acoustic event detection from DAS
# ---------------------------------------------------------------------------
def detect_events_das(
    depth_ft: np.ndarray,
    time_s: np.ndarray,
    das: np.ndarray,
    threshold_sigma: float = 4.0,
) -> List[LeakEvent]:
    """Detect high-energy acoustic events in a DAS waterfall image."""
    energy = np.mean(das ** 2, axis=1)
    mu, sigma = np.mean(energy), np.std(energy)
    anomalies = energy > mu + threshold_sigma * sigma

    events: List[LeakEvent] = []
    in_anomaly = False
    start = 0
    for i in range(len(anomalies)):
        if anomalies[i] and not in_anomaly:
            in_anomaly = True
            start = i
        elif not anomalies[i] and in_anomaly:
            in_anomaly = False
            idx_max = start + np.argmax(energy[start:i])
            events.append(LeakEvent(
                depth_ft=float(depth_ft[idx_max]),
                severity=float((energy[idx_max] - mu) / sigma),
                method="DAS",
                description="High acoustic energy event",
            ))
    return events


# ---------------------------------------------------------------------------
# 6. Diagnostic time comparison
# ---------------------------------------------------------------------------
def estimate_diagnostic_times(
    logging_interval_ft: float = 15000.0,
    speed_ft_per_min: float = 30.0,
    station_spacing_ft: float = 10.0,
    station_dwell_min: float = 1.0,
    n_conditions: int = 3,
) -> dict:
    """Compare conventional single-point vs. fiber-optic acquisition times.

    Returns
    -------
    dict with keys 'conventional_hours', 'fiber_hours', 'reduction_pct'.
    """
    # Conventional: temp down-pass + acoustic up-pass, per condition
    temp_pass = logging_interval_ft / speed_ft_per_min / 60.0
    acoustic_pass = (logging_interval_ft / station_spacing_ft) * station_dwell_min / 60.0
    conv_per_cond = temp_pass + acoustic_pass
    conv_total = conv_per_cond * n_conditions

    # Fiber optic: single RIH + short acquisition per condition
    rih_time = logging_interval_ft / speed_ft_per_min / 60.0
    fiber_acq_per_cond = 2.0   # ~2 hrs per condition
    fiber_total = rih_time + fiber_acq_per_cond * n_conditions

    reduction = (1.0 - fiber_total / conv_total) * 100.0
    return {
        "conventional_hours": round(conv_total, 1),
        "fiber_hours": round(fiber_total, 1),
        "reduction_pct": round(reduction, 1),
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_all():
    depth = np.linspace(0, 10000, 2000)
    time_s = np.linspace(0, 10, 500)

    # DTS
    temp = simulate_dts(depth, leak_depths=[3000, 7000],
                        leak_amplitudes=[5.0, -3.0])
    assert temp.shape == depth.shape
    leaks = detect_leaks_dts(depth, temp)
    assert len(leaks) >= 1, "Should detect at least one DTS leak"
    assert any(abs(l.depth_ft - 3000) < 200 for l in leaks), \
        "Should find leak near 3000 ft"

    # DAS
    das = simulate_das(depth, time_s,
                       event_depths=[5000], event_times=[5.0],
                       event_amplitudes=[1.0])
    assert das.shape == (len(depth), len(time_s))

    # Stacking
    stacked = temporal_stack(das, n_stack=10)
    assert stacked.shape[1] == len(time_s) // 10

    # DAS event detection
    events = detect_events_das(depth, time_s, das)
    assert len(events) >= 1

    # SNR
    snr = compute_snr(das[:, 250], 0.05)
    assert snr > 0

    # Diagnostic time
    times = estimate_diagnostic_times()
    assert times["reduction_pct"] > 50, "Fiber optics should reduce time by >50%"

    print("[PASS] fiber_optics_sensing — all tests passed")


if __name__ == "__main__":
    test_all()
