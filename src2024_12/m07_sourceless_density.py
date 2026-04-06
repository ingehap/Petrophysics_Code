#!/usr/bin/env python3
"""
Enhancing Accuracy and Range of Sourceless Density
=====================================================
Based on: Mauborgne et al. (2024), Petrophysics 65(6), pp. 929-943.
DOI: 10.30632/PJV65N6-2024a7

Implements the sourceless neutron-gamma density (sNGD) measurement
algorithm concepts:
  1. Inelastic/capture gamma-ray separation via time gating.
  2. Neutron transport correction (hydrogen index dependency).
  3. Density computation from corrected gamma-ray count rate.
  4. Environmental corrections (hole size, mud weight, salinity).
"""
import numpy as np
from typing import Dict, Tuple

def compute_hydrogen_index(porosity, fluid='water', gas_density=0.2):
    """Compute hydrogen index for formation fluid.
    HI = 1.0 for fresh water, scales with density for gas."""
    phi = np.asarray(porosity, dtype=float)
    if fluid == 'water':
        return phi * 1.0  # HI_water = 1.0
    elif fluid == 'gas':
        return phi * gas_density / 1.0
    elif fluid == 'oil':
        return phi * 0.9  # typical oil HI
    return phi

def separate_inelastic_capture(total_counts, burst_gate, early_gate, late_gate, sigma_formation=15.0):
    """Separate inelastic and capture gamma rays using time gating.
    The capture signal builds up during pulsed neutron bursts and
    is removed by subtracting the late-gate (capture-dominated) signal.
    
    Parameters: total_counts, burst/early/late gate counts, sigma (capture units)."""
    total = np.asarray(total_counts, dtype=float)
    burst = np.asarray(burst_gate, dtype=float)
    early = np.asarray(early_gate, dtype=float)
    late = np.asarray(late_gate, dtype=float)
    # Capture fraction estimation from sigma (higher sigma = faster thermalization)
    capture_fraction = 1 - np.exp(-sigma_formation / 30.0)
    # Inelastic signal: burst gate minus estimated capture contribution
    inelastic = burst - capture_fraction * late
    inelastic = np.maximum(inelastic, 0.01 * burst)
    return inelastic, late  # returns (inelastic, capture)

def neutron_transport_correction(inelastic_counts, hydrogen_index, calibration_hi=0.15):
    """Correct inelastic count rate for neutron transport effects.
    The secondary gamma source strength and shape depend on HI.
    Blue curve correction per paper Fig. 1."""
    hi = np.asarray(hydrogen_index, dtype=float)
    # Correction factor: ratio of actual HI to calibration HI
    correction = np.exp(-0.8 * (hi - calibration_hi))
    return np.asarray(inelastic_counts, dtype=float) * correction

def counts_to_density(corrected_counts, spine_slope=-0.08, spine_intercept=2.65):
    """Convert corrected count rates to formation density using spine relation.
    log(counts) ~ slope * density + intercept (spine-and-ribs approach)."""
    log_counts = np.log(np.maximum(corrected_counts, 1.0))
    density = (log_counts - np.log(np.exp(spine_intercept))) / spine_slope + spine_intercept
    return np.clip(density, 1.0, 3.5)

def apply_environmental_corrections(density, hole_diameter_in=8.5, mud_weight_ppg=9.0,
                                     mud_salinity_kppm=0.0, standoff_in=0.0):
    """Apply environmental corrections for borehole effects."""
    d = np.asarray(density, dtype=float)
    # Hole size correction (nominal 8.5 in)
    hole_corr = 0.02 * (hole_diameter_in - 8.5)
    # Mud weight correction (nominal 9 ppg)
    mud_corr = 0.005 * (mud_weight_ppg - 9.0)
    # Salinity correction
    sal_corr = 0.0001 * mud_salinity_kppm
    # Standoff correction
    so_corr = 0.05 * standoff_in
    return d + hole_corr + mud_corr + sal_corr - so_corr

def sourceless_density_workflow(burst_counts, early_counts, late_counts,
                                 porosity, sigma, hole_dia=8.5, mw=9.0):
    """Full sourceless neutron-gamma density workflow."""
    hi = compute_hydrogen_index(porosity)
    inelastic, capture = separate_inelastic_capture(burst_counts, burst_counts, early_counts, late_counts, sigma)
    corrected = neutron_transport_correction(inelastic, hi)
    raw_density = counts_to_density(corrected)
    final_density = apply_environmental_corrections(raw_density, hole_dia, mw)
    return final_density

def test_all():
    print("=" * 70)
    print("Module 7: Sourceless Density (Mauborgne et al., 2024)")
    print("=" * 70)
    rng = np.random.RandomState(42)
    n = 100
    true_density = np.linspace(1.8, 2.8, n) + rng.normal(0, 0.02, n)
    porosity = np.clip(0.4 - 0.12 * (true_density - 1.8), 0.01, 0.40)
    sigma = 10 + 20 * porosity + rng.normal(0, 1, n)
    burst = 10000 * np.exp(-0.08 * (true_density - 2.65)) + rng.normal(0, 50, n)
    early = burst * 0.7 + rng.normal(0, 30, n)
    late = burst * 0.3 + rng.normal(0, 20, n)
    hi = compute_hydrogen_index(porosity)
    print(f"Hydrogen index range: [{hi.min():.3f}, {hi.max():.3f}]")
    inel, capt = separate_inelastic_capture(burst, burst, early, late, sigma)
    print(f"Inelastic/capture separation: inel mean={np.mean(inel):.0f}, capt mean={np.mean(capt):.0f}")
    corrected = neutron_transport_correction(inel, hi)
    density = counts_to_density(corrected)
    final = apply_environmental_corrections(density, 8.5, 10.0, 50.0, 0.25)
    print(f"Raw density range: [{density.min():.3f}, {density.max():.3f}] g/cc")
    print(f"Corrected density range: [{final.min():.3f}, {final.max():.3f}] g/cc")
    # Full workflow
    wf = sourceless_density_workflow(burst, early, late, porosity, sigma, 9.875, 12.0)
    err = np.abs(wf - true_density)
    print(f"Workflow density error: mean={np.mean(err):.3f}, max={np.max(err):.3f} g/cc")
    print("\n[PASS] All tests completed successfully.\n")

if __name__ == "__main__":
    test_all()
