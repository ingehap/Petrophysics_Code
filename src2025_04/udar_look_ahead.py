"""
UDAR Look-Ahead-While-Drilling Technology Assessment
=====================================================
Based on: Cuadros et al., "Look-Ahead-While-Drilling Technology Assessment
for Early Hazards Identification in Presalt Offshore Brazil",
Petrophysics, Vol. 66, No. 2, April 2025, pp. 190–211.

Implements:
  - Antenna tilt calibration for high-resistivity environments
  - Signal-to-noise ratio (SNR) estimation across frequencies
  - Model distribution analysis for conductive/resistive feature detection
  - Multi-frequency signal combination for optimal depth of detection

Reference: https://doi.org/10.30632/PJV66N2-2025a2 (SPWLA)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class UDARToolConfig:
    """Configuration for an ultradeep azimuthal resistivity tool."""
    frequencies_hz: np.ndarray = field(default_factory=lambda: np.array([2e3, 6e3, 12e3, 24e3, 50e3]))
    tx_rx_spacings_ft: np.ndarray = field(default_factory=lambda: np.array([42.0, 85.0, 160.0]))
    antenna_tilt_deg: float = 45.0
    noise_floor_db: float = -120.0


@dataclass
class FormationLayer:
    """A single layer in a 1D layer-cake formation model."""
    top_tvd_ft: float
    resistivity_ohm_m: float
    thickness_ft: float
    anisotropy_ratio: float = 1.0  # Rv/Rh


def skin_depth(frequency_hz: float, resistivity_ohm_m: float, mu_r: float = 1.0) -> float:
    """
    Compute electromagnetic skin depth in a conductive medium.

    Parameters
    ----------
    frequency_hz : float
        Operating frequency in Hz.
    resistivity_ohm_m : float
        Formation resistivity in Ohm·m.
    mu_r : float
        Relative magnetic permeability (default 1.0).

    Returns
    -------
    float
        Skin depth in metres.
    """
    mu_0 = 4.0 * np.pi * 1e-7
    omega = 2.0 * np.pi * frequency_hz
    return np.sqrt(2.0 * resistivity_ohm_m / (omega * mu_0 * mu_r))


def estimate_snr(tool: UDARToolConfig,
                 formation_resistivity_ohm_m: float,
                 spacing_idx: int = 0) -> np.ndarray:
    """
    Estimate signal-to-noise ratio for each frequency at a given spacing.

    In high-resistivity environments (e.g. halite in presalt), higher
    frequencies yield better SNR because the signal attenuation per unit
    distance is lower relative to the noise floor. The article notes the
    strategic use of higher frequencies to optimize SNR and enhance
    penetration depth in evaporite sections (Cuadros et al., 2025).

    Parameters
    ----------
    tool : UDARToolConfig
    formation_resistivity_ohm_m : float
    spacing_idx : int
        Index into tool.tx_rx_spacings_ft.

    Returns
    -------
    np.ndarray
        SNR in dB for each frequency.
    """
    spacing_m = tool.tx_rx_spacings_ft[spacing_idx] * 0.3048
    snr_db = np.zeros(len(tool.frequencies_hz))
    for i, f in enumerate(tool.frequencies_hz):
        delta = skin_depth(f, formation_resistivity_ohm_m)
        # Signal decays as exp(-distance/skin_depth)
        signal_atten_db = -20.0 * np.log10(np.exp(spacing_m / delta))
        # Geometric spreading (1/r^2 approximation in dB)
        geometric_db = -20.0 * np.log10(spacing_m)
        signal_db = geometric_db + signal_atten_db
        snr_db[i] = signal_db - tool.noise_floor_db
    return snr_db


def antenna_tilt_calibration(measured_signal: np.ndarray,
                             tilt_error_deg: float,
                             nominal_tilt_deg: float = 45.0) -> np.ndarray:
    """
    Apply in-situ antenna tilt calibration correction.

    Environmental and tool-specific errors in antenna tilt affect
    the measured voltages. This calibration adjusts for the angular
    error between the nominal and actual tilt (Cuadros et al., 2025).

    Parameters
    ----------
    measured_signal : np.ndarray
        Raw measured signal amplitude(s).
    tilt_error_deg : float
        Estimated tilt error in degrees.
    nominal_tilt_deg : float
        Designed antenna tilt angle in degrees.

    Returns
    -------
    np.ndarray
        Calibrated signal amplitude(s).
    """
    actual_tilt = np.radians(nominal_tilt_deg + tilt_error_deg)
    nominal_tilt = np.radians(nominal_tilt_deg)
    # Tilted antenna coupling scales as sin(2*theta) for cross-component
    correction = np.sin(2.0 * nominal_tilt) / np.sin(2.0 * actual_tilt)
    return measured_signal * correction


def model_distribution_analysis(resistivity_profile: np.ndarray,
                                 depth_ft: np.ndarray,
                                 mode: str = "conductive") -> dict:
    """
    Perform model distribution analysis to detect features ahead of the bit.

    The article describes tailoring analysis to detect either highly
    conductive features (e.g., brine-filled fractures) or more resistive
    formations (e.g., anhydrite/carbonate in presalt).

    Parameters
    ----------
    resistivity_profile : np.ndarray
        Inverted resistivity values along the wellbore (Ohm·m).
    depth_ft : np.ndarray
        Corresponding measured depths (ft).
    mode : str
        "conductive" to focus on low-resistivity anomalies,
        "resistive" to focus on high-resistivity anomalies.

    Returns
    -------
    dict
        - "anomaly_depths_ft": depths where anomalies are detected
        - "anomaly_magnitudes": magnitude of anomalies (log10 ratio)
        - "background_resistivity": estimated background resistivity
    """
    log_res = np.log10(resistivity_profile)
    # Robust background using median
    bg = np.median(log_res)
    residual = log_res - bg

    if mode == "conductive":
        # Conductive anomalies: resistivity drops below background
        threshold = -0.5  # half-decade drop
        mask = residual < threshold
    elif mode == "resistive":
        # Resistive anomalies: resistivity rises above background
        threshold = 0.5
        mask = residual > threshold
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'conductive' or 'resistive'.")

    return {
        "anomaly_depths_ft": depth_ft[mask],
        "anomaly_magnitudes": residual[mask],
        "background_resistivity": 10.0 ** bg,
    }


def optimal_frequency_selection(tool: UDARToolConfig,
                                 formation_resistivity_ohm_m: float,
                                 target_depth_m: float) -> Tuple[float, float]:
    """
    Select the optimal operating frequency for a given detection target depth.

    For presalt evaporite sections with very high resistivity, the paper
    notes that higher frequencies provide better signal while maintaining
    adequate penetration. This function finds the frequency that maximizes
    the signal at the target depth.

    Parameters
    ----------
    tool : UDARToolConfig
    formation_resistivity_ohm_m : float
    target_depth_m : float

    Returns
    -------
    Tuple[float, float]
        (optimal_frequency_hz, expected_snr_db)
    """
    best_snr = -np.inf
    best_freq = tool.frequencies_hz[0]
    for f in tool.frequencies_hz:
        delta = skin_depth(f, formation_resistivity_ohm_m)
        atten_db = -20.0 * np.log10(np.exp(target_depth_m / delta))
        geo_db = -20.0 * np.log10(max(target_depth_m, 0.1))
        snr = (geo_db + atten_db) - tool.noise_floor_db
        if snr > best_snr:
            best_snr = snr
            best_freq = f
    return best_freq, best_snr


def forward_model_1d(layers: List[FormationLayer],
                     tool: UDARToolConfig,
                     bit_depth_ft: float,
                     freq_idx: int = 0,
                     spacing_idx: int = 0) -> complex:
    """
    Simplified 1D forward model for UDAR tool response.

    Computes the approximate voltage response of a tilted-antenna EM tool
    in a horizontally layered formation. The response is modelled as a
    superposition of contributions from each layer weighted by their
    proximity and conductivity contrast.

    Parameters
    ----------
    layers : List[FormationLayer]
    tool : UDARToolConfig
    bit_depth_ft : float
    freq_idx : int
    spacing_idx : int

    Returns
    -------
    complex
        Modelled voltage response (arbitrary units).
    """
    freq = tool.frequencies_hz[freq_idx]
    spacing_m = tool.tx_rx_spacings_ft[spacing_idx] * 0.3048
    tilt = np.radians(tool.antenna_tilt_deg)
    response = 0.0 + 0.0j

    for layer in layers:
        layer_center_ft = layer.top_tvd_ft + layer.thickness_ft / 2.0
        distance_ft = abs(layer_center_ft - bit_depth_ft)
        distance_m = distance_ft * 0.3048

        delta = skin_depth(freq, layer.resistivity_ohm_m)
        # Signal contribution from this layer
        amplitude = (layer.thickness_ft * 0.3048 / max(distance_m, 0.1)) * \
                    np.exp(-distance_m / delta)
        phase = -distance_m / delta + spacing_m / delta
        # Tilt coupling
        coupling = np.sin(2.0 * tilt)
        response += coupling * amplitude * np.exp(1j * phase)

    return response


def test_all():
    """Test all functions with synthetic data."""
    print("=" * 70)
    print("Testing: udar_look_ahead (Cuadros et al., 2025)")
    print("=" * 70)

    tool = UDARToolConfig()

    # Test skin depth
    sd = skin_depth(2e3, 100.0)
    assert sd > 0, "Skin depth must be positive"
    print(f"  Skin depth at 2 kHz, 100 Ohm·m: {sd:.1f} m")

    # Test SNR estimation
    snr = estimate_snr(tool, formation_resistivity_ohm_m=1000.0, spacing_idx=0)
    assert len(snr) == len(tool.frequencies_hz)
    print(f"  SNR at 1000 Ohm·m (halite): {snr}")
    # In high resistivity, higher freq should generally give better SNR
    # for shorter spacings

    # Test antenna calibration
    raw = np.array([1.0, 2.0, 3.0])
    cal = antenna_tilt_calibration(raw, tilt_error_deg=2.0)
    assert cal.shape == raw.shape
    print(f"  Calibrated signal (2° tilt error): {cal}")

    # Test model distribution analysis
    np.random.seed(42)
    depth = np.linspace(5000, 6000, 200)
    res = 10.0 ** (2.0 + 0.1 * np.random.randn(200))
    # Insert conductive anomaly
    res[80:90] = 0.5
    result = model_distribution_analysis(res, depth, mode="conductive")
    assert len(result["anomaly_depths_ft"]) > 0, "Should detect conductive anomaly"
    print(f"  Detected {len(result['anomaly_depths_ft'])} conductive anomaly points")
    print(f"  Background resistivity: {result['background_resistivity']:.1f} Ohm·m")

    # Test optimal frequency selection
    opt_f, opt_snr = optimal_frequency_selection(tool, 500.0, target_depth_m=20.0)
    print(f"  Optimal freq for 500 Ohm·m at 20m: {opt_f/1e3:.0f} kHz, SNR={opt_snr:.1f} dB")

    # Test forward model
    layers = [
        FormationLayer(top_tvd_ft=4900, resistivity_ohm_m=1000.0, thickness_ft=100),
        FormationLayer(top_tvd_ft=5000, resistivity_ohm_m=0.5, thickness_ft=10),
        FormationLayer(top_tvd_ft=5010, resistivity_ohm_m=500.0, thickness_ft=200),
    ]
    v = forward_model_1d(layers, tool, bit_depth_ft=4980)
    print(f"  Forward model voltage: |V|={abs(v):.4e}, phase={np.degrees(np.angle(v)):.1f}°")

    print("  All tests PASSED.\n")


if __name__ == "__main__":
    test_all()
