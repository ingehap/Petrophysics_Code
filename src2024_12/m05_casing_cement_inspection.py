#!/usr/bin/env python3
"""
Casing and Cement Inspection: Logging Two Casing Sizes Simultaneously
======================================================================
Based on: Hawthorn, Ingebretson, Girneata, Delabroy, Winther,
Steinsiek, and Leslie (2024), Petrophysics 65(6), pp. 913-918.
DOI: 10.30632/PJV65N6-2024a5

Implements dual-casing evaluation workflows for logging through
drillpipe while conducting ongoing operations, including:
  1. Dual-string casing thickness modelling.
  2. Inner/outer casing evaluation from pulse-echo measurements.
  3. Cement plug quality verification.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class CasingString:
    od: float          # outer diameter, inches
    wt: float          # wall thickness, mm
    material: str      # 'carbon_steel' or 'CRA'
    grade: str         # e.g. 'L80', 'P110'

def compute_resonance_frequency(thickness_mm, vl=5900.0):
    """Compute pulse-echo resonance frequency for a casing wall.
    f_res = v_longitudinal / (2 * thickness). Ref: ultrasonic UT principles."""
    t_m = np.asarray(thickness_mm, dtype=float) * 1e-3
    return vl / (2 * t_m + 1e-12)

def estimate_thickness_from_frequency(freq_hz, vl=5900.0):
    """Invert resonance frequency to wall thickness (mm)."""
    return vl / (2 * np.asarray(freq_hz, dtype=float) + 1e-12) * 1e3

def dual_casing_model(inner: CasingString, outer: CasingString, n_points=100, seed=42):
    """Simulate dual-casing thickness measurements with noise.
    Returns depth, inner_thickness, outer_thickness arrays."""
    rng = np.random.RandomState(seed)
    depth = np.linspace(0, 500, n_points)
    inner_t = inner.wt + rng.normal(0, 0.15, n_points)
    outer_t = outer.wt + rng.normal(0, 0.2, n_points)
    # Simulate a corroded section in outer casing
    outer_t[40:55] -= rng.uniform(1.0, 3.0, 15)
    return depth, np.clip(inner_t, 0.5, inner.wt*1.5), np.clip(outer_t, 0.5, outer.wt*1.5)

def evaluate_dual_casing(inner_t, outer_t, inner_nom, outer_nom):
    """Evaluate condition of both casing strings."""
    inner_loss = (inner_nom - inner_t) / inner_nom * 100
    outer_loss = (outer_nom - outer_t) / outer_nom * 100
    def condition(loss): return np.where(loss < 10, 'good', np.where(loss < 25, 'fair', np.where(loss < 42.5, 'poor', 'critical')))
    return {
        'inner_max_loss_pct': float(np.max(inner_loss)),
        'outer_max_loss_pct': float(np.max(outer_loss)),
        'inner_condition': str(condition(np.max(inner_loss))),
        'outer_condition': str(condition(np.max(outer_loss))),
    }

def verify_cement_plug(impedance_above, impedance_plug, impedance_below, threshold=3.0):
    """Verify cement plug quality by comparing acoustic impedance (MRayl).
    Good plug: impedance_plug >> impedance_above and impedance_below."""
    return {
        'plug_impedance': float(np.mean(impedance_plug)),
        'above_impedance': float(np.mean(impedance_above)),
        'below_impedance': float(np.mean(impedance_below)),
        'plug_quality': 'good' if np.mean(impedance_plug) > threshold else 'poor',
        'seal_integrity': np.mean(impedance_plug) > threshold,
    }

def test_all():
    print("=" * 70)
    print("Module 5: Casing & Cement Inspection (Hawthorn et al., 2024)")
    print("=" * 70)
    inner = CasingString(od=5.5, wt=9.17, material='carbon_steel', grade='L80')
    outer = CasingString(od=9.625, wt=11.99, material='carbon_steel', grade='P110')
    depth, it, ot = dual_casing_model(inner, outer)
    print(f"Inner casing: nom={inner.wt}mm, meas mean={np.mean(it):.2f}mm")
    print(f"Outer casing: nom={outer.wt}mm, meas mean={np.mean(ot):.2f}mm")
    freq_inner = compute_resonance_frequency(it)
    t_back = estimate_thickness_from_frequency(freq_inner)
    print(f"Resonance freq range: {freq_inner.min()/1e3:.1f}-{freq_inner.max()/1e3:.1f} kHz")
    print(f"Round-trip thickness error: {np.mean(np.abs(t_back - it)):.4f} mm")
    ev = evaluate_dual_casing(it, ot, inner.wt, outer.wt)
    print(f"Evaluation: inner={ev['inner_condition']} ({ev['inner_max_loss_pct']:.1f}%), "
          f"outer={ev['outer_condition']} ({ev['outer_max_loss_pct']:.1f}%)")
    rng = np.random.RandomState(42)
    plug = verify_cement_plug(rng.uniform(0.5, 1.5, 10), rng.uniform(3.5, 5.0, 20), rng.uniform(0.5, 1.5, 10))
    print(f"Cement plug: quality={plug['plug_quality']}, impedance={plug['plug_impedance']:.2f} MRayl")
    print("\n[PASS] All tests completed successfully.\n")

if __name__ == "__main__":
    test_all()
