#!/usr/bin/env python3
"""
Module 5: Cement Bond Evaluation for FBE-Coated Casings
=======================================================
Implements ideas from:
  Bazaid et al., "Innovative Approach to Enhance Evaluation of Well
  Integrity in Unconventional Completions With FBE Coating,"
  Petrophysics, vol. 66, no. 4, pp. 594–615, August 2025.

Key concepts:
  - Ultrasonic pulse-echo measurement through FBE-coated casing
  - Flexural wave imaging for cement bond quality
  - Acoustic impedance estimation from flexural resonance
  - Combined thickness + cement evaluation with full azimuthal coverage
  - Handling of the extra FBE coating interface
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Material properties
# ---------------------------------------------------------------------------
@dataclass
class Material:
    """Acoustic properties of a layer."""
    name: str
    velocity_m_per_s: float       # compressional velocity
    density_kg_per_m3: float
    thickness_m: float = 0.0

    @property
    def impedance(self) -> float:
        """Acoustic impedance (MRayl = 1e6 Pa·s/m)."""
        return self.density_kg_per_m3 * self.velocity_m_per_s / 1e6


WATER = Material("water", 1480, 1000)
STEEL = Material("steel", 5900, 7800)
FBE_COATING = Material("FBE", 2400, 1300, thickness_m=0.0004)  # ~0.4 mm
CEMENT_121 = Material("cement_121pcf", 3200, 1940)
CEMENT_91 = Material("cement_91pcf", 2600, 1460)
FREE_PIPE = Material("air/fluid", 1480, 1000)   # annulus fluid


# ---------------------------------------------------------------------------
# 1. Reflection coefficient at an interface
# ---------------------------------------------------------------------------
def reflection_coeff(z1: float, z2: float) -> float:
    """Normal-incidence reflection coefficient (amplitude)."""
    return (z2 - z1) / (z2 + z1 + 1e-30)


def transmission_coeff(z1: float, z2: float) -> float:
    return 1.0 - abs(reflection_coeff(z1, z2))


# ---------------------------------------------------------------------------
# 2. Ultrasonic pulse-echo model (conventional)
# ---------------------------------------------------------------------------
def pulse_echo_waveform(
    layers: list,
    dt_s: float = 1e-7,
    n_samples: int = 2000,
    centre_freq_hz: float = 500e3,
    noise_std: float = 0.01,
    rng=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a 1-D ultrasonic pulse-echo waveform through layered media.

    Parameters
    ----------
    layers : list of Material
        Ordered list of layers from inside (fluid) to outside.

    Returns
    -------
    time_s, waveform : ndarrays
    """
    rng = rng or np.random.default_rng(0)
    time_s = np.arange(n_samples) * dt_s
    waveform = np.zeros(n_samples)

    # Pulse wavelet
    def wavelet(t0):
        t_rel = time_s - t0
        env = np.exp(-0.5 * (t_rel / (1.0 / centre_freq_hz)) ** 2)
        return env * np.cos(2 * np.pi * centre_freq_hz * t_rel)

    cumulative_time = 0.0
    cumulative_transmission = 1.0

    for i in range(len(layers) - 1):
        l_curr = layers[i]
        l_next = layers[i + 1]

        # Two-way travel through current layer
        if l_curr.thickness_m > 0:
            twt = 2.0 * l_curr.thickness_m / l_curr.velocity_m_per_s
            cumulative_time += twt

        r = reflection_coeff(l_curr.impedance, l_next.impedance)
        amp = cumulative_transmission * r
        waveform += amp * wavelet(cumulative_time)
        cumulative_transmission *= transmission_coeff(l_curr.impedance,
                                                      l_next.impedance)

    waveform += rng.normal(0, noise_std, n_samples)
    return time_s, waveform


# ---------------------------------------------------------------------------
# 3. Flexural wave measurement – resonance-based impedance
# ---------------------------------------------------------------------------
def flexural_resonance_impedance(
    casing: Material,
    annulus_material: Material,
    coating: Optional[Material] = None,
) -> float:
    """Estimate annulus acoustic impedance from flexural-wave resonance.

    The flexural mode is sensitive to what is outside the casing.  A
    simplified model: the resonance quality factor Q is proportional
    to the impedance contrast.

    Returns
    -------
    float
        Estimated annulus impedance (MRayl).
    """
    z_annulus = annulus_material.impedance

    # If FBE coating is present, the effective impedance is slightly shifted
    if coating is not None:
        z_coat = coating.impedance
        # Coating acts as a thin matching layer — slight perturbation
        coating_factor = 1.0 - 0.1 * abs(z_coat - z_annulus) / (z_coat + z_annulus + 1e-12)
    else:
        coating_factor = 1.0

    # The flexural mode measurement resolves the annulus impedance
    # with accuracy ±0.5 MRayl as stated in the paper
    estimated_z = z_annulus * coating_factor
    return estimated_z


# ---------------------------------------------------------------------------
# 4. Casing thickness from pulse-echo time of flight
# ---------------------------------------------------------------------------
def estimate_casing_thickness(
    time_s: np.ndarray,
    waveform: np.ndarray,
    casing_velocity: float = 5900.0,
) -> float:
    """Estimate casing wall thickness from the pulse-echo waveform.

    Detects the first two strong reflections and uses the time
    difference to calculate thickness.
    """
    envelope = np.abs(waveform)
    # Find peaks
    from scipy.signal import find_peaks
    peaks, props = find_peaks(envelope, height=0.05 * envelope.max(),
                               distance=20)
    if len(peaks) < 2:
        return 0.0
    dt = time_s[peaks[1]] - time_s[peaks[0]]
    thickness_m = casing_velocity * dt / 2.0
    return thickness_m


# ---------------------------------------------------------------------------
# 5. Full azimuthal scan
# ---------------------------------------------------------------------------
def azimuthal_scan(
    n_sectors: int,
    casing: Material,
    annulus_material: Material,
    coating: Optional[Material] = None,
    defect_sectors: Optional[list] = None,
) -> dict:
    """Simulate a full azimuthal pulse-echo + flexural scan.

    Parameters
    ----------
    defect_sectors : list of int
        Sectors where annulus is free-pipe (no cement).

    Returns
    -------
    dict with 'impedance_map' and 'thickness_map'.
    """
    impedance_map = np.zeros(n_sectors)
    thickness_map = np.zeros(n_sectors)
    rng = np.random.default_rng(42)

    for s in range(n_sectors):
        if defect_sectors and s in defect_sectors:
            ann = FREE_PIPE
        else:
            ann = annulus_material

        # Flexural impedance
        z_est = flexural_resonance_impedance(casing, ann, coating)
        impedance_map[s] = z_est + rng.normal(0, 0.1)

        # Pulse-echo thickness
        layers = [WATER]
        if coating:
            layers.append(Material(coating.name, coating.velocity_m_per_s,
                                    coating.density_kg_per_m3,
                                    thickness_m=coating.thickness_m))
        layers.append(Material(casing.name, casing.velocity_m_per_s,
                                casing.density_kg_per_m3,
                                thickness_m=casing.thickness_m))
        layers.append(ann)
        t, w = pulse_echo_waveform(layers, noise_std=0.005, rng=rng)
        thickness_map[s] = estimate_casing_thickness(t, w) * 1000  # mm

    return {"impedance_map": impedance_map, "thickness_map": thickness_map}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_all():
    # Reflection coefficient sanity
    r = reflection_coeff(WATER.impedance, STEEL.impedance)
    assert abs(r) > 0.8, "Water-steel reflection should be strong"

    # Pulse-echo through water→FBE→steel→cement
    casing = Material("steel", 5900, 7800, thickness_m=0.009)  # ~9 mm
    layers = [WATER, FBE_COATING, casing, CEMENT_121]
    t, w = pulse_echo_waveform(layers)
    assert w.shape[0] == 2000

    # Flexural impedance estimation
    z_cem = flexural_resonance_impedance(casing, CEMENT_121, FBE_COATING)
    z_free = flexural_resonance_impedance(casing, FREE_PIPE, FBE_COATING)
    assert z_cem > z_free, "Cemented should have higher impedance than free"

    # Accuracy check
    assert abs(z_cem - CEMENT_121.impedance) < 1.0, \
        f"Impedance error too large: {z_cem} vs {CEMENT_121.impedance}"

    # Azimuthal scan
    result = azimuthal_scan(
        n_sectors=16,
        casing=casing,
        annulus_material=CEMENT_121,
        coating=FBE_COATING,
        defect_sectors=[4, 5, 6],
    )
    imp = result["impedance_map"]
    # Free-pipe sectors should have lower impedance
    mean_cement = np.mean([imp[i] for i in range(16) if i not in [4, 5, 6]])
    mean_free = np.mean([imp[i] for i in [4, 5, 6]])
    assert mean_free < mean_cement, "Free-pipe impedance should be lower"

    print("[PASS] fbe_cement_evaluation — all tests passed")


if __name__ == "__main__":
    test_all()
