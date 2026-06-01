"""
Innovative Application of Acoustic Emission Monitoring of Multiphase Flow
in Intelligent Completions

Reference:
    Zeghlache, M. L., Aidagulov, G., and Sindt, O. (2026). Innovative
    Application of Acoustic Emission Monitoring of Multiphase Flow in
    Intelligent Completions. Petrophysics, 67(3), 619-631.
    DOI: 10.30632/PJV67N3-2026a10

The paper applies acoustic emission (AE) monitoring to multiphase flow in
intelligent completions, testing single- and two-phase flows at various rates
and correlating the acoustic energy with the flow properties / flow rates,
with potential for early detection of water and gas breakthrough.

This module implements the standard AE flow-monitoring building blocks:
    - AE feature extraction from a waveform (RMS, acoustic energy,
      ring-down count / hit rate, peak amplitude).
    - An acoustic-energy vs flow-rate power-law calibration (energy ~ a Q^b)
      and its inverse (rate from energy).
    - A two-phase (water/gas) discriminator from the AE spectral centroid /
      energy partition, and a breakthrough detector that flags a sudden
      change in the AE signature.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. AE feature extraction
# ---------------------------------------------------------------------------

def ae_rms(waveform: np.ndarray) -> float:
    """Root-mean-square amplitude of an AE waveform."""
    w = np.asarray(waveform, float)
    return float(np.sqrt(np.mean(w ** 2)))


def acoustic_energy(waveform: np.ndarray, dt: float = 1.0) -> float:
    """
    Acoustic energy = integral of the squared signal:

        E = sum(w^2) * dt
    """
    w = np.asarray(waveform, float)
    return float(np.sum(w ** 2) * dt)


def ring_down_count(waveform: np.ndarray, threshold: float) -> int:
    """
    Ring-down count = number of positive-going threshold crossings (a classic
    AE hit-activity measure).
    """
    w = np.asarray(waveform, float)
    above = w > threshold
    return int(np.sum(above[1:] & ~above[:-1]))


def spectral_centroid(waveform: np.ndarray, fs: float) -> float:
    """
    Spectral centroid (Hz) of the AE waveform - the amplitude-weighted mean
    frequency, used to separate higher-frequency gas-dominated turbulence from
    lower-frequency liquid flow.
    """
    w = np.asarray(waveform, float)
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(w.size, d=1.0 / fs)
    s = spec.sum()
    return float(np.sum(freqs * spec) / s) if s > 0 else 0.0


@dataclass
class AEFeatures:
    rms: float
    energy: float
    counts: int
    centroid: float


def extract_features(waveform: np.ndarray, fs: float,
                     dt: float = 1.0, threshold: float = None) -> AEFeatures:
    """Extract the standard AE feature set from a waveform."""
    if threshold is None:
        threshold = 2.0 * ae_rms(waveform)
    return AEFeatures(
        rms=ae_rms(waveform),
        energy=acoustic_energy(waveform, dt),
        counts=ring_down_count(waveform, threshold),
        centroid=spectral_centroid(waveform, fs),
    )


# ---------------------------------------------------------------------------
# 2. Acoustic-energy vs flow-rate calibration
# ---------------------------------------------------------------------------

def fit_energy_rate(rates: Sequence[float], energies: Sequence[float]
                    ) -> Tuple[float, float]:
    """
    Fit a power-law calibration  E = a * Q^b  in log-log space.

    Returns (a, b).
    """
    q = np.log(np.asarray(rates, float))
    e = np.log(np.asarray(energies, float))
    b, loga = np.polyfit(q, e, 1)
    return float(math.exp(loga)), float(b)


def rate_from_energy(energy: float, a: float, b: float) -> float:
    """Invert the calibration: Q = (E / a)^(1/b)."""
    if a <= 0.0 or b == 0.0:
        return float("nan")
    return (energy / a) ** (1.0 / b)


# ---------------------------------------------------------------------------
# 3. Phase discrimination and breakthrough detection
# ---------------------------------------------------------------------------

def classify_phase(features: AEFeatures,
                   centroid_gas_hz: float = 5e4) -> str:
    """
    Classify the dominant flowing phase from the AE signature.

    Gas-bearing turbulent flow produces a higher-frequency, higher-energy AE
    signature than single-phase liquid; the spectral centroid is the primary
    discriminator.
    """
    if features.centroid >= centroid_gas_hz:
        return "gas / two-phase (gas)"
    if features.counts > 0 and features.rms > 0:
        return "liquid (single or water-bearing)"
    return "quiescent"


def detect_breakthrough(energy_series: Sequence[float],
                        rel_jump: float = 0.5) -> List[int]:
    """
    Flag indices where the acoustic energy jumps by more than `rel_jump`
    relative to the running mean of the preceding samples - an early indicator
    of water or gas breakthrough into the completion.

    Returns the list of flagged sample indices.
    """
    e = np.asarray(energy_series, float)
    flags = []
    for i in range(1, e.size):
        base = e[:i].mean()
        if base > 0 and (e[i] - base) / base > rel_jump:
            flags.append(i)
    return flags


# ---------------------------------------------------------------------------
# 4. Convenience: full workflow example
# ---------------------------------------------------------------------------

def example_workflow():
    """Run a complete example and print key results."""
    print("=" * 64)
    print("Acoustic Emission Monitoring of Multiphase Flow")
    print("Ref: Zeghlache, Aidagulov & Sindt, Petrophysics 67(3) 2026")
    print("=" * 64)

    fs = 1e6  # 1 MHz sampling
    rng = np.random.default_rng(7)
    n = 4096
    t = np.arange(n) / fs

    # Synthetic waveforms: liquid (low-freq) vs gas (high-freq, energetic).
    liquid = 0.5 * np.sin(2 * np.pi * 8e3 * t) + 0.1 * rng.standard_normal(n)
    gas = 1.2 * np.sin(2 * np.pi * 8e4 * t) + 0.4 * rng.standard_normal(n)

    fl = extract_features(liquid, fs)
    fg = extract_features(gas, fs)
    print(f"\nLiquid AE: energy={fl.energy:8.1f}  centroid={fl.centroid:8.0f} Hz"
          f"  -> {classify_phase(fl)}")
    print(f"Gas    AE: energy={fg.energy:8.1f}  centroid={fg.centroid:8.0f} Hz"
          f"  -> {classify_phase(fg)}")

    # Energy-rate calibration from a flow-loop sweep.
    rates = [50, 100, 200, 400, 800]            # e.g. bbl/d
    energies = [120, 480, 1900, 7600, 30000]
    a, b = fit_energy_rate(rates, energies)
    print(f"\nEnergy-rate calibration:  E = {a:.2f} * Q^{b:.2f}")
    print(f"  rate from E=5000 -> {rate_from_energy(5000, a, b):.0f} (units of Q)")

    # Breakthrough detection on an energy time series.
    series = [100, 110, 105, 120, 600, 650, 700]
    print(f"\nBreakthrough flags at indices: {detect_breakthrough(series)}")

    return fg


if __name__ == "__main__":
    example_workflow()
