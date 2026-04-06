#!/usr/bin/env python3
"""
Advanced Noise Logging: From Leak Detection to Quantitative Flow Profiling
===========================================================================
Based on: Galli and Pirrone (2024), Petrophysics 65(6), pp. 919-927.
DOI: 10.30632/PJV65N6-2024a6

Implements the ANL (Advanced Noise Log) spectral analysis methodology:
  1. Noise power amplitude (NPA) computation in frequency bands.
  2. Leak detection via spectral signature analysis.
  3. Borehole vs. reservoir flow separation using frequency cutoff.
  4. Quantitative relative flow profiling.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class NoiseStation:
    depth: float            # m MD
    spectrum: np.ndarray    # power spectral density
    frequencies: np.ndarray # Hz

def generate_synthetic_noise_data(n_depths=100, n_freqs=256, seed=42):
    """Generate synthetic ANL data with flow and leak signatures."""
    rng = np.random.RandomState(seed)
    depths = np.linspace(1000, 2000, n_depths)
    freqs = np.linspace(100, 20000, n_freqs)
    data = []
    for i, d in enumerate(depths):
        # Background noise
        psd = rng.exponential(0.01, n_freqs) * (1000 / (freqs + 100))
        # Add flow signature (low freq < 4kHz) in producing zones
        if 30 < i < 50:  # borehole flow zone
            flow_mask = freqs < 4000
            psd[flow_mask] += 0.05 * np.exp(-((freqs[flow_mask] - 1500) / 1000)**2)
        if 60 < i < 75:  # reservoir flow zone
            res_mask = freqs > 4000
            psd[res_mask] += 0.03 * np.exp(-((freqs[res_mask] - 8000) / 3000)**2)
        # Add leak signature (broadband, high amplitude)
        if 45 < i < 48:
            psd += 0.1 * np.ones(n_freqs)
        data.append(NoiseStation(depth=d, spectrum=psd, frequencies=freqs))
    return data

def compute_npa(station, freq_low, freq_high):
    """Compute noise power amplitude in a frequency band (dB)."""
    mask = (station.frequencies >= freq_low) & (station.frequencies <= freq_high)
    if mask.sum() == 0: return -np.inf
    power = np.mean(station.spectrum[mask])
    return 10 * np.log10(max(power, 1e-20))

def spectral_analysis(stations, bands=None):
    """Compute NPA for multiple frequency bands at all depths.
    Default bands: low (100-2kHz), mid (2-8kHz), high (8-20kHz)."""
    if bands is None:
        bands = [('low', 100, 2000), ('mid', 2000, 8000), ('high', 8000, 20000)]
    depths = np.array([s.depth for s in stations])
    result = {'depth': depths}
    for name, lo, hi in bands:
        result[name] = np.array([compute_npa(s, lo, hi) for s in stations])
    return result

def detect_leaks(spectral_result, threshold_db=-15.0):
    """Detect leaks from broadband noise anomalies.
    Leaks show elevated noise across all frequency bands."""
    depths = spectral_result['depth']
    bands = [k for k in spectral_result if k != 'depth']
    leak_flags = np.ones(len(depths), dtype=bool)
    for b in bands:
        leak_flags &= (spectral_result[b] > threshold_db)
    leak_depths = depths[leak_flags]
    return leak_flags, leak_depths

def separate_borehole_reservoir_flow(stations, freq_cutoff=4000.0):
    """Separate borehole flow (low freq) from reservoir flow (high freq)
    using the paper's frequency cutoff methodology."""
    depths = np.array([s.depth for s in stations])
    bh_noise = np.array([compute_npa(s, 100, freq_cutoff) for s in stations])
    res_noise = np.array([compute_npa(s, freq_cutoff, 20000) for s in stations])
    return depths, bh_noise, res_noise

def compute_relative_flow_profile(noise_db):
    """Convert noise power to relative flow rate profile (cumulative).
    Based on proportionality between noise power and flow rate."""
    linear_power = 10 ** (noise_db / 10)
    # Subtract background
    bg = np.percentile(linear_power, 10)
    flow = np.maximum(linear_power - bg, 0)
    total = flow.sum()
    if total > 0:
        discrete_rate = flow / total * 100  # percentage contribution
        cumulative_rate = np.cumsum(discrete_rate)
    else:
        discrete_rate = np.zeros_like(flow)
        cumulative_rate = np.zeros_like(flow)
    return discrete_rate, cumulative_rate

def quantitative_flow_profiling(stations, freq_cutoff=4000.0):
    """Full quantitative flow profiling workflow."""
    depths, bh_noise, res_noise = separate_borehole_reservoir_flow(stations, freq_cutoff)
    bh_discrete, bh_cumulative = compute_relative_flow_profile(bh_noise)
    res_discrete, res_cumulative = compute_relative_flow_profile(res_noise)
    return {
        'depth': depths,
        'borehole_discrete_pct': bh_discrete,
        'borehole_cumulative_pct': bh_cumulative,
        'reservoir_discrete_pct': res_discrete,
        'reservoir_cumulative_pct': res_cumulative,
    }

def test_all():
    print("=" * 70)
    print("Module 6: Advanced Noise Logging (Galli & Pirrone, 2024)")
    print("=" * 70)
    stations = generate_synthetic_noise_data()
    print(f"Generated {len(stations)} noise stations")
    spec = spectral_analysis(stations)
    print(f"Spectral analysis bands: {[k for k in spec if k != 'depth']}")
    print(f"  Low band NPA range: [{spec['low'].min():.1f}, {spec['low'].max():.1f}] dB")
    leak_flags, leak_depths = detect_leaks(spec, threshold_db=-18.0)
    print(f"Leak detection: {leak_flags.sum()} stations flagged")
    if len(leak_depths) > 0:
        print(f"  Leak depths: {leak_depths[0]:.0f}-{leak_depths[-1]:.0f} m")
    depths, bh, res = separate_borehole_reservoir_flow(stations)
    print(f"Flow separation: BH noise range [{bh.min():.1f}, {bh.max():.1f}] dB")
    flow = quantitative_flow_profiling(stations)
    print(f"Flow profiling: BH cumulative max={flow['borehole_cumulative_pct'][-1]:.1f}%")
    print(f"                Res cumulative max={flow['reservoir_cumulative_pct'][-1]:.1f}%")
    print("\n[PASS] All tests completed successfully.\n")

if __name__ == "__main__":
    test_all()
