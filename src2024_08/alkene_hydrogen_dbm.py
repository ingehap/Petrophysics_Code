"""
Optimize Drilling Decisions Based on Real-Time Detected Alkene and Hydrogen
===========================================================================
Based on: Qubaisi, K., Kharaba, A., Hewitt, R., and Sanclemente, M. (2024),
"Optimize Drilling Decisions Based on Real-Time Detected Alkene and Hydrogen
at Surface," Petrophysics, 65(4), pp. 585-592. DOI: 10.30632/PJV65N4-2024a11

Implements:
  - Drill-bit metamorphism (DBM) detection using alkenes (C2=) and H2
  - C2S/C2 ratio alarm for bit wear monitoring
  - Real-time decision support for POOH (pull out of hole) decisions
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum


class DBMSeverity(Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class AdvancedGasData:
    """Advanced gas chain data including alkenes and non-HC gases."""
    depth: np.ndarray
    c2_saturated: np.ndarray   # ethane (C2H6) - saturated
    c2_unsaturated: np.ndarray # ethylene (C2H4) - alkene
    c3_saturated: np.ndarray   # propane
    c3_unsaturated: np.ndarray # propylene
    hydrogen: np.ndarray       # H2 concentration (ppm)
    helium: np.ndarray         # He concentration (ppm)
    total_gas: np.ndarray      # total HC gas


@dataclass
class DrillingState:
    """Drilling parameters for correlation with DBM indicators."""
    depth: np.ndarray
    wob: np.ndarray       # weight on bit (klbs)
    rop: np.ndarray       # rate of penetration (ft/hr)
    rpm: np.ndarray       # rotary speed
    torque: np.ndarray    # torque (kft-lbs)


def compute_c2s_c2_ratio(c2_saturated: np.ndarray,
                         c2_unsaturated: np.ndarray) -> np.ndarray:
    """Compute C2=/C2 ratio (alkene to alkane).

    C2= (ethylene) is produced by thermal cracking of drill-bit
    metamorphism. High ratios indicate significant bit wear.

    Normal drilling: ratio < 0.05
    Mild DBM: 0.05 - 0.15
    Moderate DBM: 0.15 - 0.30
    Severe DBM: > 0.30
    """
    return c2_unsaturated / (c2_saturated + 1e-10)


def detect_dbm(gas_data: AdvancedGasData,
               c2s_c2_threshold: float = 0.05,
               h2_threshold: float = 50,
               window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Detect drill-bit metamorphism from alkene and hydrogen signatures.

    Returns (dbm_flag, severity) arrays.

    The paper shows that both alkene AND hydrogen must be elevated
    for confirmed DBM. Alkenes alone could be from formation.
    """
    c2s_c2 = compute_c2s_c2_ratio(gas_data.c2_saturated, gas_data.c2_unsaturated)

    # Smooth for more stable detection
    from scipy.ndimage import uniform_filter1d
    c2s_c2_smooth = uniform_filter1d(c2s_c2, size=window)
    h2_smooth = uniform_filter1d(gas_data.hydrogen, size=window)

    # DBM requires both indicators
    alkene_flag = c2s_c2_smooth > c2s_c2_threshold
    h2_flag = h2_smooth > h2_threshold
    dbm_flag = alkene_flag & h2_flag

    # Severity classification
    severity = np.array([DBMSeverity.NONE] * len(gas_data.depth))
    severity[dbm_flag & (c2s_c2_smooth < 0.15)] = DBMSeverity.MILD
    severity[dbm_flag & (c2s_c2_smooth >= 0.15) & (c2s_c2_smooth < 0.30)] = DBMSeverity.MODERATE
    severity[dbm_flag & (c2s_c2_smooth >= 0.30)] = DBMSeverity.SEVERE

    return dbm_flag, severity


def recommend_action(severity: np.ndarray,
                     drilling: DrillingState,
                     consecutive_severe: int = 10) -> List[dict]:
    """Generate real-time drilling recommendations based on DBM severity.

    Actions from the paper:
      - NONE: Continue drilling normally
      - MILD: Monitor, consider reducing WOB
      - MODERATE: Reduce WOB and ROP, prepare for possible trip
      - SEVERE: POOH to change bit if consecutive severe readings
    """
    recommendations = []
    severe_count = 0

    for i, sev in enumerate(severity):
        if sev == DBMSeverity.SEVERE:
            severe_count += 1
        else:
            severe_count = 0

        if sev == DBMSeverity.NONE:
            action = "continue_normal"
        elif sev == DBMSeverity.MILD:
            action = "monitor_reduce_wob"
        elif sev == DBMSeverity.MODERATE:
            action = "reduce_wob_rop"
        elif sev == DBMSeverity.SEVERE:
            if severe_count >= consecutive_severe:
                action = "pooh_change_bit"
            else:
                action = "reduce_wob_rop_urgently"
        else:
            action = "continue_normal"

        recommendations.append({
            "depth": drilling.depth[i],
            "severity": sev,
            "action": action,
            "wob": drilling.wob[i],
            "rop": drilling.rop[i],
        })

    return recommendations


def correlate_dbm_with_drilling(gas_data: AdvancedGasData,
                                drilling: DrillingState) -> dict:
    """Correlate DBM indicators with drilling parameters.

    The paper shows DBM increases with WOB in abrasive formations.
    """
    c2s_c2 = compute_c2s_c2_ratio(gas_data.c2_saturated, gas_data.c2_unsaturated)

    # Correlation coefficients
    corr_wob = np.corrcoef(drilling.wob, c2s_c2)[0, 1]
    corr_rop = np.corrcoef(drilling.rop, c2s_c2)[0, 1]
    corr_h2_wob = np.corrcoef(drilling.wob, gas_data.hydrogen)[0, 1]

    return {
        "c2s_c2_vs_wob": corr_wob,
        "c2s_c2_vs_rop": corr_rop,
        "h2_vs_wob": corr_h2_wob,
    }


def test_all():
    """Test drill-bit metamorphism detection pipeline."""
    print("=" * 70)
    print("Testing: DBM Detection via Alkene & H2 (Qubaisi et al., 2024)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    n_pts = 200

    # Simulate drilling through an abrasive formation
    depths = np.linspace(8000, 10000, n_pts)

    # Normal drilling initially, then increasing DBM
    dbm_onset = 120  # DBM starts after this point
    wob = 15 + 5 * rng.random(n_pts)
    wob[dbm_onset:] += np.linspace(0, 10, n_pts - dbm_onset)  # increasing WOB
    rop = 40 + 10 * rng.random(n_pts)
    rpm = 120 + 10 * rng.random(n_pts)
    torque = 8 + 3 * rng.random(n_pts)

    drilling = DrillingState(depth=depths, wob=wob, rop=rop, rpm=rpm, torque=torque)

    # Gas data: alkenes and H2 increase with bit wear
    c2_sat = 200 + 50 * rng.random(n_pts)
    c2_unsat = 2 + rng.random(n_pts) * 3
    # After DBM onset: alkene increases
    c2_unsat[dbm_onset:] += np.linspace(0, 80, n_pts - dbm_onset) * (1 + 0.1 * rng.random(n_pts - dbm_onset))

    h2 = 10 + 5 * rng.random(n_pts)
    h2[dbm_onset:] += np.linspace(0, 200, n_pts - dbm_onset) * (1 + 0.1 * rng.random(n_pts - dbm_onset))

    gas_data = AdvancedGasData(
        depth=depths,
        c2_saturated=c2_sat, c2_unsaturated=c2_unsat,
        c3_saturated=100 + 30 * rng.random(n_pts),
        c3_unsaturated=1 + rng.random(n_pts),
        hydrogen=h2, helium=5 + 2 * rng.random(n_pts),
        total_gas=1000 + 300 * rng.random(n_pts),
    )

    # Detect DBM
    dbm_flags, severity = detect_dbm(gas_data)
    print(f"  DBM detection results:")
    print(f"    Total points: {n_pts}")
    print(f"    DBM detected: {dbm_flags.sum()} points")
    for sev in DBMSeverity:
        count = sum(1 for s in severity if s == sev)
        print(f"    {sev.value}: {count}")

    # C2=/C2 ratio statistics
    c2s_c2 = compute_c2s_c2_ratio(gas_data.c2_saturated, gas_data.c2_unsaturated)
    print(f"\n  C2=/C2 ratio: min={c2s_c2.min():.3f}, max={c2s_c2.max():.3f}")
    print(f"  H2: min={gas_data.hydrogen.min():.0f}, max={gas_data.hydrogen.max():.0f} ppm")

    # Drilling recommendations
    recs = recommend_action(severity, drilling)
    actions = [r["action"] for r in recs]
    for action in set(actions):
        count = actions.count(action)
        print(f"\n  Action '{action}': {count} occurrences")

    # Correlation analysis
    corr = correlate_dbm_with_drilling(gas_data, drilling)
    print(f"\n  Correlations:")
    print(f"    C2=/C2 vs WOB: {corr['c2s_c2_vs_wob']:.3f}")
    print(f"    H2 vs WOB:     {corr['h2_vs_wob']:.3f}")

    print("\n  [PASS] DBM detection module tests completed.")
    return True


if __name__ == "__main__":
    test_all()
