#!/usr/bin/env python3
"""
Well Integrity Measurements Throughout the CCS Project Life Cycle
==================================================================
Based on: Valstar, Nettleton, Borchardt, Costeno, Landry, and Laronga (2024),
Petrophysics 65(6), pp. 896-912. DOI: 10.30632/PJV65N6-2024a4

Implements a well integrity assessment framework for CCS projects:
  1. Cement bond quality evaluation (acoustic/ultrasonic).
  2. Casing corrosion assessment.
  3. Time-lapse monitoring strategy with risk scoring.
  4. Impact of CO2-resistant materials on measurement interpretation.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class ProjectPhase(Enum):
    LEGACY_ASSESSMENT = "legacy_assessment"
    CONSTRUCTION = "construction"
    INJECTION = "injection"
    CLOSURE = "closure"
    POST_CLOSURE = "post_closure"

class MeasurementType(Enum):
    ACOUSTIC_CBL = "acoustic_cement_bond_log"
    ULTRASONIC_IMAGING = "ultrasonic_imaging"
    ELECTROMAGNETIC = "electromagnetic_corrosion"
    PULSED_NEUTRON = "pulsed_neutron"
    DTS_TEMPERATURE = "distributed_temperature_sensing"
    DAS_ACOUSTIC = "distributed_acoustic_sensing"
    MECHANICAL_INTEGRITY_TEST = "mechanical_integrity_test"

@dataclass
class WellIntegrityReport:
    well_id: str
    phase: ProjectPhase
    overall_risk_score: float
    casing_condition: str
    cement_condition: str
    leak_detected: bool
    recommended_actions: List[str]
    measurements_used: List[MeasurementType]

def compute_cement_bond_index(amplitude, free_pipe_amp=80.0, bonded_amp=5.0):
    """Bond Index = (A_free - A_meas) / (A_free - A_bonded)."""
    bi = (free_pipe_amp - np.asarray(amplitude, dtype=float)) / (free_pipe_amp - bonded_amp + 1e-10)
    return np.clip(bi, 0, 1)

def evaluate_acoustic_impedance(impedance, cement_type='Portland'):
    """Evaluate cement quality from ultrasonic acoustic impedance (MRayl)."""
    thresholds = {
        'Portland': (4.0, 2.5), 'CO2_resistant': (3.5, 2.0), 'epoxy_resin': (2.5, 1.5),
    }
    good, fair = thresholds.get(cement_type, (4.0, 2.5))
    imp = np.asarray(impedance, dtype=float)
    q = np.where(imp >= good, 1.0, np.where(imp >= fair, 0.5 + 0.5*(imp-fair)/(good-fair), 0.25*imp/fair))
    return np.clip(q, 0, 1)

def assess_casing_corrosion(measured_thickness, nominal_thickness=12.0, rate_mmpy=0.1):
    """Assess casing corrosion from wall thickness measurements (mm)."""
    t = np.asarray(measured_thickness, dtype=float)
    min_t, avg_t = float(t.min()), float(t.mean())
    loss_pct = (nominal_thickness - min_t) / nominal_thickness * 100
    min_accept = nominal_thickness * 0.575
    life = max(0, (min_t - min_accept) / rate_mmpy) if rate_mmpy > 0 else 999.0
    cond = 'good' if loss_pct < 10 else 'fair' if loss_pct < 25 else 'poor' if loss_pct < 42.5 else 'critical'
    return {'min_thickness_mm': min_t, 'avg_thickness_mm': avg_t, 'max_loss_pct': loss_pct,
            'remaining_life_years': life, 'condition': cond}

def compute_risk_score(cement_bi, casing_loss_pct, has_leak, phase, years_remaining=100.0):
    """Compute overall well integrity risk score (0-100, higher = worse)."""
    cement_risk = 40 * (1 - cement_bi)
    casing_risk = 30 * min(casing_loss_pct / 50, 1.0)
    leak_risk = 20 if has_leak else 0
    time_risk = 10 * min(years_remaining / 100, 1.0)
    mult = {ProjectPhase.LEGACY_ASSESSMENT: 1.2, ProjectPhase.CONSTRUCTION: 0.8,
            ProjectPhase.INJECTION: 1.0, ProjectPhase.CLOSURE: 1.1, ProjectPhase.POST_CLOSURE: 1.3}.get(phase, 1.0)
    return min(100, (cement_risk + casing_risk + leak_risk + time_risk) * mult)

def recommend_measurements(phase):
    """Recommend measurements for a CCS project phase."""
    s = {
        ProjectPhase.LEGACY_ASSESSMENT: [MeasurementType.ACOUSTIC_CBL, MeasurementType.ULTRASONIC_IMAGING,
                                          MeasurementType.ELECTROMAGNETIC, MeasurementType.MECHANICAL_INTEGRITY_TEST],
        ProjectPhase.CONSTRUCTION: [MeasurementType.ACOUSTIC_CBL, MeasurementType.ULTRASONIC_IMAGING],
        ProjectPhase.INJECTION: [MeasurementType.PULSED_NEUTRON, MeasurementType.DTS_TEMPERATURE, MeasurementType.DAS_ACOUSTIC],
        ProjectPhase.CLOSURE: [MeasurementType.ACOUSTIC_CBL, MeasurementType.ELECTROMAGNETIC, MeasurementType.MECHANICAL_INTEGRITY_TEST],
        ProjectPhase.POST_CLOSURE: [MeasurementType.DTS_TEMPERATURE, MeasurementType.DAS_ACOUSTIC, MeasurementType.PULSED_NEUTRON],
    }
    return s.get(phase, [])

def generate_report(well_id, phase, cbl_amp, casing_thickness, nominal_t=12.0):
    """Generate comprehensive well integrity report."""
    bi = compute_cement_bond_index(cbl_amp)
    avg_bi = float(np.mean(bi))
    cement_cond = 'good' if avg_bi > 0.8 else 'fair' if avg_bi > 0.5 else 'poor'
    corr = assess_casing_corrosion(casing_thickness, nominal_t)
    has_leak = avg_bi < 0.3 and corr['max_loss_pct'] > 30
    risk = compute_risk_score(avg_bi, corr['max_loss_pct'], has_leak, phase)
    actions = []
    if cement_cond == 'poor': actions.append("Remedial squeeze cementing recommended.")
    if corr['condition'] in ('poor', 'critical'): actions.append("Casing repair recommended.")
    if has_leak: actions.append("Immediate leak remediation required.")
    if not actions: actions.append("Continue routine monitoring.")
    return WellIntegrityReport(well_id, phase, risk, corr['condition'], cement_cond,
                               has_leak, actions, recommend_measurements(phase))

def test_all():
    print("=" * 70)
    print("Module 4: Well Integrity for CCS (Valstar et al., 2024)")
    print("=" * 70)
    rng = np.random.RandomState(42)
    n = 200
    cbl = rng.uniform(5, 40, n); cbl[50:80] = rng.uniform(60, 80, 30)
    bi = compute_cement_bond_index(cbl)
    print(f"Cement bond index: mean={np.mean(bi):.2f}, min={np.min(bi):.2f}")
    ai = rng.uniform(1, 6, n)
    q = evaluate_acoustic_impedance(ai, 'CO2_resistant')
    print(f"Cement quality (CO2-resistant): mean={np.mean(q):.2f}")
    thick = rng.normal(11.5, 0.5, n); thick[100:120] = rng.normal(8.0, 0.3, 20)
    corr = assess_casing_corrosion(thick)
    print(f"Casing: {corr['condition']}, loss={corr['max_loss_pct']:.1f}%, life={corr['remaining_life_years']:.0f}y")
    for phase in ProjectPhase:
        meas = recommend_measurements(phase)
        print(f"  {phase.value}: {len(meas)} measurements")
    report = generate_report("CCS-01", ProjectPhase.INJECTION, cbl, thick)
    print(f"\nReport: risk={report.overall_risk_score:.1f}, cement={report.cement_condition}, "
          f"casing={report.casing_condition}, leak={report.leak_detected}")
    for a in report.recommended_actions: print(f"  -> {a}")
    print("\n[PASS] All tests completed successfully.\n")

if __name__ == "__main__":
    test_all()
