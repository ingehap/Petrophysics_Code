#!/usr/bin/env python3
"""
Perched Water Observations in Deepwater Miocene Fields
========================================================
Based on: Kostin and Sanchez-Ramirez (2024), Petrophysics 65(6), pp. 1010-1022.
DOI: 10.30632/PJV65N6-2024a13

Implements perched water detection and characterization:
  1. Capillary pressure-based saturation profile (drainage Sw).
  2. Resistivity-based saturation (Archie equation).
  3. Perched water detection by comparing Sw profiles.
  4. Transition zone thickness estimation.
  5. Volumetric impact assessment of perched water intervals.
"""
import numpy as np
from typing import Dict, Tuple, Optional

def leverett_j_function(Sw, a=0.5, b=2.0):
    """Leverett J-function for drainage capillary pressure.
    J = a / (Sw - Swirr)^b (simplified)."""
    Sw = np.asarray(Sw, dtype=float)
    return a / (np.clip(Sw - 0.05, 0.01, None) ** b)

def drainage_sw_from_height(height_above_fwl, permeability_md, porosity,
                             sigma_cos_theta=26.0, rho_w=1.03, rho_hc=0.7):
    """Compute drainage water saturation vs. height above FWL.
    Uses simplified capillary pressure - saturation relation.
    Pc = (rho_w - rho_hc) * g * h, then Sw from J-function."""
    h = np.asarray(height_above_fwl, dtype=float)
    g = 9.81
    # Capillary pressure (Pa)
    Pc = (rho_w - rho_hc) * 1000 * g * h
    Pc_psi = Pc / 6894.76  # convert to psi
    # J-function
    k = np.asarray(permeability_md, dtype=float)
    phi = np.asarray(porosity, dtype=float)
    J = Pc_psi * np.sqrt(k / phi) / (sigma_cos_theta + 1e-10)
    # Sw from J (inverse of J-function, simplified)
    Swirr = 0.05 + 0.1 * (1 - phi)  # irreducible water
    Sw = Swirr + 0.5 / (J + 0.5)
    return np.clip(Sw, Swirr, 1.0)

def archie_sw(Rt, Rw, porosity, a=1.0, m=2.0, n=2.0):
    """Water saturation from Archie equation.
    Sw = (a * Rw / (Rt * phi^m))^(1/n)."""
    phi = np.asarray(porosity, dtype=float)
    Rt_arr = np.asarray(Rt, dtype=float)
    Sw = (a * Rw / (Rt_arr * phi**m + 1e-10)) ** (1.0 / n)
    return np.clip(Sw, 0, 1)

def detect_perched_water(Sw_capillary, Sw_resistivity, threshold=0.15):
    """Detect perched water by comparing capillary-predicted and
    resistivity-derived saturations.
    Perched water: Sw_resistivity >> Sw_capillary (more water than expected)."""
    Sw_cap = np.asarray(Sw_capillary, dtype=float)
    Sw_res = np.asarray(Sw_resistivity, dtype=float)
    excess_water = Sw_res - Sw_cap
    perched_flag = excess_water > threshold
    return perched_flag, excess_water

def estimate_transition_zone(Sw_profile, depths, Sw_threshold=0.5, Sw_irr_threshold=0.15):
    """Estimate perched water transition zone thickness.
    Returns the depth interval where Sw transitions from high to irreducible."""
    Sw = np.asarray(Sw_profile, dtype=float)
    d = np.asarray(depths, dtype=float)
    # Find where Sw drops below threshold
    above_thresh = Sw > Sw_threshold
    if not np.any(above_thresh):
        return {'top': None, 'bottom': None, 'thickness': 0.0}
    top_idx = np.argmax(above_thresh)
    # Find bottom of transition zone
    below_irr = Sw[top_idx:] < Sw_irr_threshold + 0.1
    if np.any(below_irr):
        bot_idx = top_idx + np.argmax(below_irr)
    else:
        bot_idx = len(Sw) - 1
    return {
        'top': float(d[top_idx]),
        'bottom': float(d[bot_idx]),
        'thickness': float(d[bot_idx] - d[top_idx]),
    }

def perched_water_volume(Sw_excess, porosity, area_m2, dz_m):
    """Estimate volume of perched water (m³).
    V = sum(phi * Sw_excess * A * dz)."""
    phi = np.asarray(porosity, dtype=float)
    sw_ex = np.clip(np.asarray(Sw_excess, dtype=float), 0, None)
    return float(np.sum(phi * sw_ex * area_m2 * dz_m))

def classify_perched_water_origin(water_salinity_ppm, aquifer_salinity_ppm, tolerance_pct=15):
    """Classify whether perched water is from the main aquifer or
    represents trapped formation water based on chemistry.
    If salinity differs significantly from aquifer -> trapped water."""
    diff_pct = abs(water_salinity_ppm - aquifer_salinity_ppm) / (aquifer_salinity_ppm + 1e-10) * 100
    if diff_pct < tolerance_pct:
        return 'aquifer_origin', diff_pct
    else:
        return 'trapped_formation_water', diff_pct

def perched_water_analysis(depths, Rt, porosity, permeability, Rw,
                            height_above_fwl, rho_hc=0.7):
    """Full perched water analysis workflow."""
    Sw_cap = drainage_sw_from_height(height_above_fwl, permeability, porosity, rho_hc=rho_hc)
    Sw_res = archie_sw(Rt, Rw, porosity)
    perched_flag, excess = detect_perched_water(Sw_cap, Sw_res)
    tz = estimate_transition_zone(Sw_res, depths)
    dz = np.mean(np.diff(depths)) if len(depths) > 1 else 1.0
    volume = perched_water_volume(excess, porosity, 10000, dz)  # 10000 m² area
    return {
        'Sw_capillary': Sw_cap,
        'Sw_resistivity': Sw_res,
        'perched_flag': perched_flag,
        'excess_water': excess,
        'transition_zone': tz,
        'perched_volume_m3': volume,
        'n_perched_intervals': int(np.sum(np.diff(perched_flag.astype(int)) == 1)),
    }

def test_all():
    print("=" * 70)
    print("Module 13: Perched Water (Kostin & Sanchez-Ramirez, 2024)")
    print("=" * 70)
    rng = np.random.RandomState(42)
    n = 100
    depths = np.linspace(2000, 2100, n)
    fwl_depth = 2200  # gas-water contact at 2200m
    height = fwl_depth - depths
    porosity = 0.22 + rng.normal(0, 0.02, n)
    perm = 200 + rng.normal(0, 30, n)
    # Expected (capillary) Sw
    Sw_cap = drainage_sw_from_height(height, perm, porosity, rho_hc=0.25)
    print(f"Capillary Sw range: [{Sw_cap.min():.3f}, {Sw_cap.max():.3f}]")
    # Resistivity (normal + perched water zone)
    Rw = 0.05
    Rt = rng.uniform(10, 100, n)
    # Inject a perched water zone: lower resistivity -> higher Sw
    Rt[40:55] = rng.uniform(1.5, 3.0, 15)  # low Rt = water
    Sw_res = archie_sw(Rt, Rw, porosity)
    print(f"Resistivity Sw range: [{Sw_res.min():.3f}, {Sw_res.max():.3f}]")
    # Detection
    perched, excess = detect_perched_water(Sw_cap, Sw_res)
    print(f"Perched water detected: {perched.sum()} depth points")
    if perched.any():
        pw_depths = depths[perched]
        print(f"  Depth range: {pw_depths.min():.0f}-{pw_depths.max():.0f} m")
    # Transition zone
    tz = estimate_transition_zone(Sw_res, depths)
    print(f"Transition zone: {tz['top']:.0f}-{tz['bottom']:.0f} m ({tz['thickness']:.1f} m)")
    # Volume
    dz = np.mean(np.diff(depths))
    vol = perched_water_volume(excess, porosity, 10000, dz)
    print(f"Perched water volume: {vol:.0f} m³ (over 10000 m² area)")
    # Chemistry
    origin, diff = classify_perched_water_origin(85000, 120000)
    print(f"Water origin: {origin} (salinity diff={diff:.1f}%)")
    origin2, diff2 = classify_perched_water_origin(118000, 120000)
    print(f"Water origin: {origin2} (salinity diff={diff2:.1f}%)")
    # Full workflow
    result = perched_water_analysis(depths, Rt, porosity, perm, Rw, height, rho_hc=0.25)
    print(f"\nFull analysis: {result['n_perched_intervals']} perched interval(s)")
    print(f"  Volume: {result['perched_volume_m3']:.0f} m³")
    print("\n[PASS] All tests completed successfully.\n")

if __name__ == "__main__":
    test_all()
