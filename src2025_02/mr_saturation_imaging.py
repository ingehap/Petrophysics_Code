#!/usr/bin/env python3
"""
Direct Hydrocarbon Saturation Imaging with 13C Magnetic Resonance.

Reference: Ansaribaranghar et al., 2025, Petrophysics 66(1), 169-182. DOI:10.30632/PJV66N1-2025a12

Implements:
  - 1D hybrid SE-SPI MRI profile simulation
  - 13C direct oil saturation profiling
  - 1H total fluid profiling
  - Water saturation by subtraction (1H - 13C)
  - Capillary end-effect detection from profiles
  - Dean-Stark validation
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class ImagingParams:
    """MRI imaging parameters."""
    field_T: float = 3.1     # magnetic field [T]
    n_pixels: int = 64       # spatial resolution
    core_length_mm: float = 50.0
    C13_sensitivity: float = 0.011  # relative to 1H (~1.1% natural abundance * γ ratio)

def simulate_saturation_profile(
    n_pixels, Sw_inlet, Sw_outlet, profile_type='linear',
    cee_length_frac=0.15,
):
    """Simulate a 1D water saturation profile along core.

    Args:
        n_pixels: Number of pixels.
        Sw_inlet: Sw at inlet.
        Sw_outlet: Sw at outlet (before CEE).
        profile_type: 'linear', 'uniform', or 'cee' (capillary end effect).
        cee_length_frac: Fraction of core affected by CEE.

    Returns:
        Sw profile array.
    """
    x = np.linspace(0, 1, n_pixels)
    if profile_type == 'uniform':
        Sw = np.full(n_pixels, (Sw_inlet + Sw_outlet) / 2)
    elif profile_type == 'linear':
        Sw = Sw_inlet + (Sw_outlet - Sw_inlet) * x
    elif profile_type == 'cee':
        Sw_base = Sw_inlet + (Sw_outlet - Sw_inlet) * x
        cee_start = 1.0 - cee_length_frac
        cee_mask = x > cee_start
        cee_buildup = (x[cee_mask] - cee_start) / cee_length_frac
        Sw[cee_mask] = Sw_base[cee_mask] + 0.2 * cee_buildup ** 2
    else:
        Sw = np.full(n_pixels, Sw_inlet)
    return np.clip(Sw, 0, 1)

def C13_oil_profile(So_profile, porosity_profile, sensitivity=0.011):
    """13C signal profile proportional to oil content.

    Signal ∝ So * φ * sensitivity_factor
    """
    return So_profile * porosity_profile * sensitivity

def H1_total_profile(Sw_profile, So_profile, porosity_profile):
    """1H signal profile (total fluid: oil + water).

    Signal ∝ (Sw + So) * φ  (both phases contain 1H)
    """
    return (Sw_profile + So_profile) * porosity_profile

def water_profile_by_subtraction(H1_profile, C13_profile, sensitivity_ratio=0.011):
    """Water saturation profile from 1H - scaled 13C.

    Since 13C is oil-only: water_signal = 1H_signal - 13C_signal/sensitivity
    """
    oil_signal = C13_profile / sensitivity_ratio
    return H1_profile - oil_signal

def detect_capillary_end_effect(So_profile, threshold_gradient=0.02):
    """Detect CEE from oil saturation profile near outlet.

    CEE appears as an increase in oil (decrease in water) near the outlet end.

    Args:
        So_profile: Oil saturation profile.
        threshold_gradient: Gradient threshold for CEE detection.

    Returns:
        CEE detection results.
    """
    grad = np.gradient(So_profile)
    outlet_region = grad[-len(grad)//4:]  # last quarter of core

    has_cee = np.any(outlet_region < -threshold_gradient)
    cee_magnitude = abs(outlet_region.min()) if has_cee else 0

    return dict(
        has_cee=has_cee,
        cee_magnitude=cee_magnitude,
        outlet_So=So_profile[-1],
        bulk_So=np.mean(So_profile[len(So_profile)//4:-len(So_profile)//4]),
    )

def dean_stark_validation(Sw_profile_mean, Sw_dean_stark):
    """Compare profile-averaged Sw with Dean-Stark measurement."""
    diff = abs(Sw_profile_mean - Sw_dean_stark)
    return dict(Sw_MRI=Sw_profile_mean, Sw_DS=Sw_dean_stark, diff_su=diff*100,
                passes=diff < 0.01)  # < 1 s.u. as in paper

def oil_wet_cee_profile(n_pixels, Sw_bulk=0.30, cee_frac=0.2):
    """Generate oil-wet CEE profile where oil accumulates at outlet.

    In oil-wet samples, 13C profiles clearly reveal hydrocarbon CEE.
    """
    x = np.linspace(0, 1, n_pixels)
    So = np.full(n_pixels, 1 - Sw_bulk)
    # Oil-wet CEE: oil builds up at outlet
    cee_mask = x > (1 - cee_frac)
    So[cee_mask] += 0.15 * ((x[cee_mask] - (1-cee_frac)) / cee_frac) ** 1.5
    return np.clip(So, 0, 1)

if __name__ == "__main__":
    params = ImagingParams()
    n = params.n_pixels
    phi = np.full(n, 0.22)

    # Water-wet drainage profile
    Sw = simulate_saturation_profile(n, 0.25, 0.30, 'linear')
    So = 1 - Sw

    c13 = C13_oil_profile(So, phi, params.C13_sensitivity)
    h1 = H1_total_profile(Sw, So, phi)
    w_sub = water_profile_by_subtraction(h1, c13, params.C13_sensitivity)

    cee = detect_capillary_end_effect(So)
    ds = dean_stark_validation(Sw.mean(), 0.275)

    # Oil-wet sample with CEE
    So_ow = oil_wet_cee_profile(n)
    cee_ow = detect_capillary_end_effect(So_ow)

    print(f"13C MR Imaging — {params.field_T} T, {n} pixels, L={params.core_length_mm} mm")
    print(f"Water-wet: Sw_mean={Sw.mean():.3f}, CEE={cee['has_cee']}")
    print(f"Dean-Stark validation: diff={ds['diff_su']:.1f} s.u., passes={ds['passes']}")
    print(f"Oil-wet CEE: detected={cee_ow['has_cee']}, magnitude={cee_ow['cee_magnitude']:.3f}")
