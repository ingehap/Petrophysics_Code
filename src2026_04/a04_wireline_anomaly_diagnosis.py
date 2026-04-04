"""
Intelligent Sensors and Algorithms for Diagnosing Downhole Operating
Conditions of Wireline Logging Instruments

Reference:
    Liu, Z., Zhang, X., Fan, Q., Zhou, L., Zhang, Y., Zhao, Z., and
    Zhang, Z. (2026). Intelligent Sensors and Algorithms for Diagnosing
    Downhole Operating Conditions of Wireline Logging Instruments.
    Petrophysics, 67(2), 295–317. DOI: 10.30632/PJV67N2-2026a4

Implements:
  - Force-balance mechanical model for wireline instruments (Eqs. 1–8)
  - Cable tension dynamic model (Eqs. 9, 7–8)
  - Winch vibration harmonic oscillator (Eqs. 10–12)
  - Moving-window tension feature extraction (Eqs. 13–16)
  - Dual-threshold anomaly detection (Eqs. 17–18, 19)
  - Vibration preprocessing & EWMA baseline (Eqs. 20–22)
  - Moving-window MAD vibration feature (Eq. 23)
  - Z-score standardisation and graded alarm (Eqs. 24–25)
  - Probabilistic dual-signal fusion for obstruction / jamming diagnosis
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import IntEnum


# ---------------------------------------------------------------------------
# 1. Mechanical force model (Eqs. 1–8)
# ---------------------------------------------------------------------------

@dataclass
class InstrumentParams:
    mass_kg:     float = 500.0    # instrument mass, kg
    cable_linear_density: float = 2.0   # λ, kg/m
    fluid_density: float = 1200.0       # ρ, kg/m³
    displaced_vol: float = 0.12         # V, m³
    friction_coeff: float = 0.25        # µ (dimensionless)
    g: float = 9.81                     # m/s²


def gravity(params: InstrumentParams) -> float:
    """Instrument weight (Eq. 1)."""
    return params.mass_kg * params.g


def buoyancy(params: InstrumentParams) -> float:
    """Buoyancy force (Eq. 2)."""
    return params.fluid_density * params.g * params.displaced_vol


def friction_force(params: InstrumentParams, velocity_mps: float) -> float:
    """Friction opposing motion (Eq. 3)."""
    return params.friction_coeff * np.abs(velocity_mps)


def cable_tension_lowering(params: InstrumentParams,
                            depth_m: float,
                            acceleration_mps2: float = 0.0,
                            stuck_force_N: float = 0.0) -> float:
    """
    Cable tension when lowering (Eq. 7).
    T = G - Fb - Ff + m*a + cable_weight(depth) + stuck_force
    """
    G   = gravity(params)
    Fb  = buoyancy(params)
    Ff  = friction_force(params, 0.5)
    Wc  = params.cable_linear_density * params.g * depth_m   # Eq. 6
    T   = G - Fb - Ff + params.mass_kg * acceleration_mps2 + Wc + stuck_force_N
    return max(T, 0.0)


def cable_tension_hoisting(params: InstrumentParams,
                            depth_m: float,
                            acceleration_mps2: float = 0.0,
                            stuck_force_N: float = 0.0) -> float:
    """
    Cable tension when hoisting (Eq. 8).
    T = G + Ff - Fb + m*a + cable_weight + stuck_force
    """
    G  = gravity(params)
    Fb = buoyancy(params)
    Ff = friction_force(params, 0.5)
    Wc = params.cable_linear_density * params.g * depth_m
    T  = G + Ff - Fb + params.mass_kg * acceleration_mps2 + Wc + stuck_force_N
    return max(T, 0.0)


# ---------------------------------------------------------------------------
# 2. Dynamic tension model (Eq. 9) and winch vibration (Eqs. 10–12)
# ---------------------------------------------------------------------------

def tension_dynamic(T0: float, dT: float, omega: float,
                    t: np.ndarray) -> np.ndarray:
    """Periodic tension variation (Eq. 9)."""
    return T0 + dT * np.sin(omega * t)


def winch_acceleration(tension_variation: np.ndarray,
                       winch_mass_kg: float) -> np.ndarray:
    """
    Winch acceleration from tension changes in the rigid-body limit (Eq. 11–12).
    a_winch = ΔT / m_winch
    """
    return tension_variation / winch_mass_kg


# ---------------------------------------------------------------------------
# 3. Tension signal feature extraction (Eqs. 13–16)
# ---------------------------------------------------------------------------

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Moving average MA_t over past N points (Eq. 13)."""
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


def moving_msd(x: np.ndarray, window: int) -> np.ndarray:
    """Moving mean-squared deviation MSD_t (Eq. 14)."""
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = x[lo:i+1]
        out[i] = np.mean((seg - seg.mean())**2)
    return out


def moving_std(x: np.ndarray, window: int) -> np.ndarray:
    """Moving standard deviation σ_t (Eq. 15)."""
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = np.std(x[lo:i+1])
    return out


def moving_abs_deviation(x: np.ndarray, ma: np.ndarray) -> np.ndarray:
    """Absolute deviation from moving average (Eq. 16)."""
    return np.abs(x - ma)


# ---------------------------------------------------------------------------
# 4. Dual-threshold anomaly detection for tension (Eqs. 17–19)
# ---------------------------------------------------------------------------

def detect_tension_anomalies(tension: np.ndarray,
                              window: int = 40,
                              k1: float = 4.0,
                              k2: float = 0.125) -> Dict[str, np.ndarray]:
    """
    Dual-threshold anomaly detection on cable tension signal (Eqs. 17–19).

    Parameters
    ----------
    tension : Raw tension time series (N or kg-force)
    window  : Moving-window size (samples)
    k1      : Extreme-fluctuation multiplier (Eq. 17): flag if |x - MA| > k1*σ
    k2      : Trend-anomaly fraction (Eq. 18): flag if |x - MA|/MA > k2

    Returns
    -------
    dict with:
        ma          : moving average
        sigma       : moving standard deviation
        abs_dev     : absolute deviation
        L1_flags    : boolean array – extreme fluctuation (L1 = k1*σ)
        L2_flags    : boolean array – trend anomaly  (L2 = k2*MA)
        any_anomaly : boolean array – L1 OR L2
    """
    ma    = moving_average(tension, window)
    sigma = moving_std(tension, window)
    dev   = moving_abs_deviation(tension, ma)

    L1_thresh = k1 * sigma                  # Eq. 17
    L2_thresh = k2 * np.abs(ma)             # Eq. 18
    L1_flags  = dev > L1_thresh
    L2_flags  = dev > L2_thresh

    return {
        "ma":          ma,
        "sigma":       sigma,
        "abs_dev":     dev,
        "L1_thresh":   L1_thresh,
        "L2_thresh":   L2_thresh,
        "L1_flags":    L1_flags,
        "L2_flags":    L2_flags,
        "any_anomaly": L1_flags | L2_flags,
    }


# ---------------------------------------------------------------------------
# 5. Vibration preprocessing (Eqs. 20–22)
# ---------------------------------------------------------------------------

def remove_adjacent_spikes(z: np.ndarray,
                            spike_factor_adjacent: float = 2.0,
                            spike_factor_persistence: float = 3.0
                            ) -> np.ndarray:
    """
    Two-pass spike removal based on adjacent-point mutation detection
    (Eqs. 20–21).

    A point z[i] is flagged as a spike if:
      Eq. 20: |z[i] - z[i-1]| > spike_factor_adjacent * |z[i-1] - z[i-2]|
      Eq. 21: The mutation is not sustained (next point returns to baseline).
    """
    z_clean = z.copy().astype(float)
    n = len(z)
    for i in range(2, n - 1):
        abs_diff_curr = abs(z_clean[i]   - z_clean[i-1])
        abs_diff_prev = abs(z_clean[i-1] - z_clean[i-2])
        # Eq. 20: instantaneous jump
        if abs_diff_curr > spike_factor_adjacent * abs_diff_prev:
            # Eq. 21: not persistent → isolated spike
            abs_diff_next = abs(z_clean[i+1] - z_clean[i])
            if abs_diff_next > spike_factor_persistence * abs_diff_prev:
                z_clean[i] = 0.5 * (z_clean[i-1] + z_clean[i+1])
    return z_clean


def ewma_reference(z: np.ndarray, window: int = 20) -> float:
    """
    EWMA-based reference value: arithmetic mean of first `window` points
    as the operational baseline (Eq. 22).
    """
    return float(np.mean(z[:window]))


# ---------------------------------------------------------------------------
# 6. Vibration MAD feature and alarm grading (Eqs. 23–25)
# ---------------------------------------------------------------------------

def moving_mad(z: np.ndarray, ref: float,
               window: int = 10) -> np.ndarray:
    """
    Moving window mean absolute deviation from reference (Eq. 23).
    MAD_t = (1/N) Σ |z_i - ref|  over last N points
    """
    n = len(z)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = np.mean(np.abs(z[lo:i+1] - ref))
    return out


def zscore_standardise(mad_seq: np.ndarray) -> np.ndarray:
    """Z-score standardisation of MAD sequence (Eq. 24)."""
    mu    = np.mean(mad_seq)
    sigma = np.std(mad_seq) + 1e-12
    return (mad_seq - mu) / sigma


class AlarmGrade(IntEnum):
    NORMAL   = 0
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    EXTREME  = 4


def grade_vibration_alarm(z_scores: np.ndarray) -> np.ndarray:
    """
    Map Z-scores to alarm grades (Eq. 25).
    Grade 4 (extreme): |Zj| > 4
    """
    grades = np.zeros(len(z_scores), dtype=int)
    grades[np.abs(z_scores) > 1.0] = AlarmGrade.LOW
    grades[np.abs(z_scores) > 2.0] = AlarmGrade.MEDIUM
    grades[np.abs(z_scores) > 3.0] = AlarmGrade.HIGH
    grades[np.abs(z_scores) > 4.0] = AlarmGrade.EXTREME
    return grades


# ---------------------------------------------------------------------------
# 7. Dual-signal fusion: probabilistic diagnosis
# ---------------------------------------------------------------------------

def fuse_tension_vibration(tension_flags: np.ndarray,
                            vib_grades:    np.ndarray,
                            p_obs: float = 0.6,
                            p_jam: float = 0.8) -> np.ndarray:
    """
    Probabilistic dual-signal fusion for obstruction vs. jamming diagnosis.

    Logic (based on paper's probabilistic scoring approach):
      - Tension L1 flag only          → obstruction candidate
      - Vibration EXTREME only        → mechanical event
      - Both flags simultaneously     → jamming (high confidence)

    Parameters
    ----------
    tension_flags : Boolean array from detect_tension_anomalies()
    vib_grades    : Integer grade array from grade_vibration_alarm()
    p_obs         : Probability weight for obstruction
    p_jam         : Probability weight for jamming

    Returns
    -------
    diagnosis : Integer array  0=Normal, 1=Obstruction, 2=Jamming
    """
    n = len(tension_flags)
    diagnosis = np.zeros(n, dtype=int)
    vib_extreme = vib_grades >= AlarmGrade.EXTREME
    vib_high    = vib_grades >= AlarmGrade.HIGH

    # Jamming: both tension anomaly AND extreme vibration
    jam_mask = tension_flags & vib_extreme
    # Obstruction: tension anomaly with moderate vibration
    obs_mask = tension_flags & vib_high & ~jam_mask
    # Single-signal hits (lower confidence)
    obs_mask |= tension_flags & ~vib_high

    diagnosis[obs_mask] = 1
    diagnosis[jam_mask] = 2
    return diagnosis


DIAGNOSIS_LABELS = {0: "Normal", 1: "Obstruction", 2: "Jamming"}


# ---------------------------------------------------------------------------
# 8. Complete pipeline
# ---------------------------------------------------------------------------

def diagnose_wireline_run(tension: np.ndarray,
                           vib_x:  np.ndarray,
                           vib_y:  np.ndarray,
                           vib_z:  np.ndarray,
                           window_tension: int = 40,
                           window_vib:     int = 10,
                           k1: float = 4.0,
                           k2: float = 0.125) -> Dict:
    """
    Full dual-signal fusion pipeline for a single wireline logging run.

    Parameters
    ----------
    tension : Cable tension time series
    vib_x, vib_y, vib_z : Three-axis winch vibration time series
    window_tension : Moving-window size for tension features
    window_vib     : Moving-window size for vibration MAD
    k1, k2         : Tension anomaly thresholds

    Returns
    -------
    dict with:
        tension_result : output of detect_tension_anomalies()
        vib_grades     : per-sample alarm grade (max over 3 axes)
        diagnosis      : per-sample diagnosis (0/1/2)
        accuracy_info  : summary statistics
    """
    # Tension anomaly detection
    t_result = detect_tension_anomalies(tension, window=window_tension,
                                        k1=k1, k2=k2)

    # Vibration: preprocess each axis
    max_grades = np.zeros(len(vib_x), dtype=int)
    for vib_raw in (vib_x, vib_y, vib_z):
        vib_clean = remove_adjacent_spikes(vib_raw)
        ref       = ewma_reference(vib_clean)
        mad       = moving_mad(vib_clean, ref, window=window_vib)
        z_scores  = zscore_standardise(mad)
        grades    = grade_vibration_alarm(z_scores)
        max_grades = np.maximum(max_grades, grades)

    # Fuse
    diagnosis = fuse_tension_vibration(t_result["any_anomaly"], max_grades)

    n_obs = int((diagnosis == 1).sum())
    n_jam = int((diagnosis == 2).sum())
    n_nor = int((diagnosis == 0).sum())

    return {
        "tension_result": t_result,
        "vib_grades":     max_grades,
        "diagnosis":      diagnosis,
        "summary": {
            "n_normal":      n_nor,
            "n_obstruction": n_obs,
            "n_jamming":     n_jam,
            "anomaly_frac":  (n_obs + n_jam) / len(diagnosis),
        },
    }


# ---------------------------------------------------------------------------
# 9. Example workflow
# ---------------------------------------------------------------------------

def example_workflow():
    print("=" * 60)
    print("Wireline Logging Anomaly Diagnosis – Dual-Signal Fusion")
    print("Ref: Liu et al., Petrophysics 67(2) 2026")
    print("=" * 60)

    rng = np.random.default_rng(0)
    n   = 5000

    # Simulate normal tension (linear increase with depth) + anomalous segment
    depth = np.linspace(500, 3000, n)
    params = InstrumentParams()
    base_tension = np.array([cable_tension_lowering(params, d) for d in depth])
    # Inject two obstruction events
    noise    = rng.normal(0, 50, n)
    stuck    = np.zeros(n)
    stuck[1500:1600] = 800.0   # obstruction
    stuck[3200:3300] = 3000.0  # jamming
    tension  = base_tension + noise + stuck

    # Simulate 3-axis vibration
    vib_x = rng.normal(0, 0.05, n)
    vib_y = rng.normal(0, 0.03, n)
    vib_z = rng.normal(0, 0.02, n)
    # Inject large vibration spike at jamming event
    vib_x[3200:3300] += rng.uniform(1.5, 3.0, 100)
    vib_z[1500:1600] += rng.uniform(0.5, 1.0, 100)

    result = diagnose_wireline_run(tension, vib_x, vib_y, vib_z)
    s = result["summary"]
    print(f"\nRun summary ({n} samples):")
    print(f"  Normal      : {s['n_normal']:5d} ({s['n_normal']/n*100:.1f}%)")
    print(f"  Obstruction : {s['n_obstruction']:5d} ({s['n_obstruction']/n*100:.1f}%)")
    print(f"  Jamming     : {s['n_jamming']:5d} ({s['n_jamming']/n*100:.1f}%)")

    # Force model
    T_low  = cable_tension_lowering(params, depth_m=2000)
    T_hois = cable_tension_hoisting(params, depth_m=2000, stuck_force_N=5000)
    print(f"\nForce model at 2000 m depth:")
    print(f"  Tension (lowering, free): {T_low:.0f} N")
    print(f"  Tension (hoisting, stuck 5 kN): {T_hois:.0f} N")

    return result


if __name__ == "__main__":
    example_workflow()
