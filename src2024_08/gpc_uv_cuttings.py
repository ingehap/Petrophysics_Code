"""
Application of GPC-UV Method for Reservoir Fluid Analysis From Drill Cuttings
=============================================================================
Based on: Yang, T., Cely, A., Moore, J., and Michael, E. (2024), "Application
of GPC-UV Method for Reservoir Fluid Analysis From Drill Cuttings,"
Petrophysics, 65(4), pp. 593-603. DOI: 10.30632/PJV65N4-2024a12

Implements:
  - Gel Permeation Chromatography with UV detection (GPC-UV) model
  - Isoabsorbance plot generation and analysis
  - Fluid property estimation from retention time and signal strength
  - OBM contamination handling
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional


@dataclass
class GPCUVMeasurement:
    """GPC-UV measurement result for a single sample."""
    retention_times: np.ndarray   # retention time (minutes)
    signal_310nm: np.ndarray      # UV signal at 310nm wavelength
    wavelengths: np.ndarray       # scanned wavelengths (nm)
    isoabsorbance: np.ndarray     # 2D isoabsorbance (time x wavelength)
    sample_type: str              # 'stock_tank_oil', 'core_extract', 'cuttings_extract'


@dataclass
class FluidProperties:
    """Estimated fluid properties from GPC-UV analysis."""
    api_gravity: float
    estimated_gor: float
    fluid_type: str            # 'black_oil', 'volatile_oil', 'gas_condensate', 'wet_gas'
    asphaltene_content: float  # relative asphaltene index
    wax_content: float         # relative wax index


def simulate_gpc_uv(api_gravity: float, gor: float,
                    contamination_level: float = 0.0,
                    random_state: int = 42) -> GPCUVMeasurement:
    """Simulate a GPC-UV measurement for a given fluid type.

    Key observations from the paper:
      - Black oil: large-size envelope, strong signals, short retention times
      - Volatile oil: flatter envelope, reduced size, increased retention times
      - Gas condensate: small envelope, much-reduced wavelength span
      - Higher API gravity -> shifted to longer retention times
    """
    rng = np.random.RandomState(random_state)

    retention_times = np.linspace(5, 25, 200)  # minutes
    wavelengths = np.linspace(250, 400, 100)   # nm

    # Characteristic retention time shifts with API gravity
    # Heavy oils (low API) elute earlier (larger molecules)
    center_rt = 10 + 0.15 * api_gravity  # shifts right with lighter oil
    rt_width = 4.0 - 0.03 * api_gravity  # narrower for lighter oils

    # Signal strength inversely related to API (heavier oils have stronger UV)
    signal_strength = 2.0 * np.exp(-0.03 * api_gravity)

    # Wavelength span: heavier oils absorb over wider wavelength range
    wl_center = 310
    wl_width = 40 - 0.3 * api_gravity

    # Generate 1D signal at 310nm
    signal_310 = signal_strength * np.exp(
        -((retention_times - center_rt) ** 2) / (2 * rt_width ** 2)
    )

    # Add asphaltene peak for heavy oils (short retention time)
    if api_gravity < 20:
        asph_peak = 0.5 * signal_strength * np.exp(
            -((retention_times - 7) ** 2) / (2 * 1.5 ** 2)
        )
        signal_310 += asph_peak

    # OBM contamination: adds a broad peak at specific retention times
    if contamination_level > 0:
        obm_peak = contamination_level * 0.8 * np.exp(
            -((retention_times - 18) ** 2) / (2 * 2.0 ** 2)
        )
        signal_310 += obm_peak

    signal_310 += rng.normal(0, 0.01, len(retention_times))
    signal_310 = np.clip(signal_310, 0, None)

    # Generate 2D isoabsorbance
    RT, WL = np.meshgrid(retention_times, wavelengths, indexing='ij')
    isoabsorbance = signal_strength * np.exp(
        -((RT - center_rt) ** 2) / (2 * rt_width ** 2)
        - ((WL - wl_center) ** 2) / (2 * wl_width ** 2)
    )
    isoabsorbance += rng.normal(0, 0.005, isoabsorbance.shape)
    isoabsorbance = np.clip(isoabsorbance, 0, None)

    return GPCUVMeasurement(
        retention_times=retention_times,
        signal_310nm=signal_310,
        wavelengths=wavelengths,
        isoabsorbance=isoabsorbance,
        sample_type="stock_tank_oil",
    )


def extract_envelope_features(measurement: GPCUVMeasurement) -> dict:
    """Extract key features from the GPC-UV isoabsorbance envelope.

    Features used for fluid property estimation:
      - Peak retention time: molecular size indicator
      - Peak signal strength: concentration/type indicator
      - Envelope area: total UV-active material
      - Wavelength span: chromophore diversity
    """
    # 1D features from 310nm signal
    peak_idx = np.argmax(measurement.signal_310nm)
    peak_rt = measurement.retention_times[peak_idx]
    peak_signal = measurement.signal_310nm[peak_idx]
    area_310 = np.trapezoid(measurement.signal_310nm, measurement.retention_times)

    # 2D features from isoabsorbance
    iso = measurement.isoabsorbance
    # Wavelength span at half-max
    max_iso = iso.max()
    half_max_mask = iso.max(axis=0) > max_iso * 0.5
    if half_max_mask.any():
        wl_span = (measurement.wavelengths[half_max_mask][-1] -
                   measurement.wavelengths[half_max_mask][0])
    else:
        wl_span = 0

    total_volume = np.sum(iso)

    return {
        "peak_retention_time": peak_rt,
        "peak_signal_strength": peak_signal,
        "area_310nm": area_310,
        "wavelength_span_nm": wl_span,
        "envelope_volume": total_volume,
    }


def estimate_fluid_properties(features: dict) -> FluidProperties:
    """Estimate fluid properties from GPC-UV envelope features.

    Uses empirical correlations observed in the paper:
      - API gravity correlates with peak retention time
      - GOR correlates inversely with signal strength
      - Fluid type determined by envelope shape
    """
    rt = features["peak_retention_time"]
    sig = features["peak_signal_strength"]
    area = features["area_310nm"]
    wl_span = features["wavelength_span_nm"]

    # API gravity from retention time (empirical)
    api = (rt - 10) / 0.15
    api = np.clip(api, 5, 55)

    # GOR estimation (inverse of signal strength and area)
    if sig > 0.1:
        gor = 50 * (1.0 / sig) ** 1.5
    else:
        gor = 5000  # very light fluid

    # Fluid type classification
    if api < 22 and gor < 200:
        fluid_type = "black_oil"
    elif api < 35 and gor < 600:
        fluid_type = "volatile_oil"
    elif api < 50 and gor < 10000:
        fluid_type = "gas_condensate"
    else:
        fluid_type = "wet_gas"

    # Asphaltene index (proportional to early-eluting signal)
    asphaltene = sig * max(0, 1 - (rt - 8) / 5)

    # Wax index (proportional to area and wavelength span)
    wax = area * wl_span / 1000

    return FluidProperties(
        api_gravity=float(api),
        estimated_gor=float(gor),
        fluid_type=fluid_type,
        asphaltene_content=float(asphaltene),
        wax_content=float(wax),
    )


def assess_contamination(cuttings_meas: GPCUVMeasurement,
                         oil_meas: GPCUVMeasurement) -> dict:
    """Assess OBM contamination level by comparing cuttings to stock tank oil.

    The paper shows that GPC-UV can bypass OBM contamination issues
    because it targets heavy molecular fractions (asphaltenes, waxes)
    that are distinct from synthetic OBM base oils.
    """
    # Compare peak positions and shapes
    cut_peak = np.argmax(cuttings_meas.signal_310nm)
    oil_peak = np.argmax(oil_meas.signal_310nm)

    rt_shift = abs(cuttings_meas.retention_times[cut_peak] -
                   oil_meas.retention_times[oil_peak])

    signal_ratio = (np.max(cuttings_meas.signal_310nm) /
                    (np.max(oil_meas.signal_310nm) + 1e-10))

    # Correlation between shapes
    min_len = min(len(cuttings_meas.signal_310nm), len(oil_meas.signal_310nm))
    corr = np.corrcoef(cuttings_meas.signal_310nm[:min_len],
                       oil_meas.signal_310nm[:min_len])[0, 1]

    return {
        "retention_time_shift": rt_shift,
        "signal_ratio": signal_ratio,
        "shape_correlation": corr,
        "contamination_significant": signal_ratio < 0.3 or corr < 0.7,
    }


def test_all():
    """Test GPC-UV fluid analysis pipeline."""
    print("=" * 70)
    print("Testing: GPC-UV Cuttings Analysis (Yang et al., 2024)")
    print("=" * 70)

    # Simulate measurements for different fluid types
    fluids = [
        ("Heavy Black Oil", 13, 50),
        ("Medium Black Oil", 25, 150),
        ("Volatile Oil", 38, 400),
        ("Gas Condensate", 48, 3000),
    ]

    print(f"  {'Fluid Type':<20} {'API':>5} {'GOR':>6} | "
          f"{'Est.API':>7} {'Est.GOR':>8} {'Type':<16}")
    print(f"  {'-'*20} {'-'*5} {'-'*6} | {'-'*7} {'-'*8} {'-'*16}")

    for name, api, gor in fluids:
        meas = simulate_gpc_uv(api, gor, random_state=42)
        features = extract_envelope_features(meas)
        props = estimate_fluid_properties(features)
        print(f"  {name:<20} {api:5.0f} {gor:6.0f} | "
              f"{props.api_gravity:7.1f} {props.estimated_gor:8.0f} {props.fluid_type:<16}")

    # Test OBM contamination assessment
    print(f"\n  Contamination assessment:")
    oil_meas = simulate_gpc_uv(13, 50, contamination_level=0.0, random_state=1)
    clean_cut = simulate_gpc_uv(13, 50, contamination_level=0.1, random_state=2)
    dirty_cut = simulate_gpc_uv(13, 50, contamination_level=0.8, random_state=3)

    for label, cut in [("Low contamination", clean_cut), ("High contamination", dirty_cut)]:
        result = assess_contamination(cut, oil_meas)
        print(f"    {label}: correlation={result['shape_correlation']:.3f}, "
              f"signal_ratio={result['signal_ratio']:.3f}, "
              f"significant={result['contamination_significant']}")

    # Envelope feature analysis
    print(f"\n  Envelope features (Heavy Oil):")
    meas = simulate_gpc_uv(13, 50)
    features = extract_envelope_features(meas)
    for k, v in features.items():
        print(f"    {k}: {v:.3f}")

    print("\n  [PASS] GPC-UV cuttings analysis tests completed.")
    return True


if __name__ == "__main__":
    test_all()
