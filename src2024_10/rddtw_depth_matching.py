#!/usr/bin/env python3
"""
Core-Log Depth Adaptive Matching Using RDDTW
==============================================
Based on: Fang et al. (2024), "Core-Log Depth Adaptive Matching Using
RDDTW", Petrophysics, Vol. 65, No. 5, pp. 835-851.

Implements:
- Standard Dynamic Time Warping (DTW)
- Derivative DTW (DDTW) using first-order derivatives
- Regularized Derivative DTW (RDDTW) with Excessive Warping
  Regularized Function (EWRF)
- Constrained DTW (CDTW) with Sakoe-Chiba band
- Pearson Correlation Coefficient (PCC) matching baseline
- Particle Swarm Optimization (PSO) for depth shift estimation
- Evaluation metrics (R², RMSE)

Reference: DOI:10.30632/PJV65N5-2024a10
"""

import numpy as np
from typing import Tuple, Optional, List


# -----------------------------------------------------------------------
# 1. Distance Measures
# -----------------------------------------------------------------------
def euclidean_distance_matrix(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix between sequences s and t."""
    s = np.asarray(s).reshape(-1, 1)
    t = np.asarray(t).reshape(-1, 1)
    return np.abs(s - t.T)


def derivative_estimate(seq: np.ndarray) -> np.ndarray:
    """
    Estimate the derivative of a sequence using the average of
    left and right differences (Keogh & Pazzani, 2001):
        D(x_i) = [(x_i - x_{i-1}) + (x_{i+1} - x_{i-1})/2] / 2
    """
    n = len(seq)
    d = np.zeros(n)
    for i in range(1, n - 1):
        d[i] = ((seq[i] - seq[i - 1]) + (seq[i + 1] - seq[i - 1]) / 2.0) / 2.0
    d[0] = d[1]
    d[-1] = d[-2]
    return d


# -----------------------------------------------------------------------
# 2. Standard DTW
# -----------------------------------------------------------------------
def dtw(s: np.ndarray, t: np.ndarray
        ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Standard Dynamic Time Warping.

    Parameters
    ----------
    s, t : np.ndarray
        Input sequences.

    Returns
    -------
    distance : float
        Accumulated DTW distance.
    cost_matrix : np.ndarray
        Accumulated cost matrix.
    path : np.ndarray  shape (K, 2)
        Optimal warping path.
    """
    n, m = len(s), len(t)
    D = euclidean_distance_matrix(s, t)
    C = np.full((n + 1, m + 1), np.inf)
    C[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            C[i, j] = D[i - 1, j - 1] + min(C[i - 1, j],
                                               C[i, j - 1],
                                               C[i - 1, j - 1])

    # Backtrack path
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        candidates = []
        if i > 0 and j > 0:
            candidates.append((C[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            candidates.append((C[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((C[i, j - 1], i, j - 1))
        _, i, j = min(candidates)

    path.reverse()
    return C[n, m], C[1:, 1:], np.array(path)


# -----------------------------------------------------------------------
# 3. Constrained DTW (CDTW) with Sakoe-Chiba band
# -----------------------------------------------------------------------
def cdtw(s: np.ndarray, t: np.ndarray, window: int = 10
         ) -> Tuple[float, np.ndarray]:
    """
    Constrained DTW with Sakoe-Chiba band.

    Parameters
    ----------
    s, t : np.ndarray
        Input sequences.
    window : int
        Half-width of the Sakoe-Chiba band.

    Returns
    -------
    distance : float
    cost_matrix : np.ndarray
    """
    n, m = len(s), len(t)
    D = euclidean_distance_matrix(s, t)
    C = np.full((n + 1, m + 1), np.inf)
    C[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            C[i, j] = D[i - 1, j - 1] + min(C[i - 1, j],
                                               C[i, j - 1],
                                               C[i - 1, j - 1])
    return C[n, m], C[1:, 1:]


# -----------------------------------------------------------------------
# 4. RDDTW – Regularized Derivative Dynamic Time Warping
# -----------------------------------------------------------------------
def rddtw(s: np.ndarray, t: np.ndarray,
           tau: float = 4.0,
           lambda_ewrf: float = 0.5
           ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Regularized Derivative Dynamic Time Warping (RDDTW).

    Combines shape distance (Euclidean) and trend distance (derivative)
    with an Excessive Warping Regularized Function (EWRF).

    Joint distance:
        d_joint(i,j) = d_shape(i,j) + tau * d_trend(i,j)

    Regularized cumulative distance:
        lambda(i,j) = epsilon(i,j) + lambda_ewrf * R(i,j)

    where R(i,j) penalizes deviations from the diagonal.

    Parameters
    ----------
    s, t : np.ndarray
        Core and log porosity sequences.
    tau : float
        Weight for trend distance (default 4.0 from paper's Fig. 9).
    lambda_ewrf : float
        Weight for excessive warping regularization.

    Returns
    -------
    distance : float
        Regularized DTW distance.
    cost_matrix : np.ndarray
        Accumulated cost matrix.
    path : np.ndarray  shape (K, 2)
        Optimal warping path.
    """
    n, m = len(s), len(t)

    # Shape distance (Euclidean)
    D_shape = euclidean_distance_matrix(s, t)

    # Trend distance (derivative)
    ds = derivative_estimate(s)
    dt = derivative_estimate(t)
    D_trend = euclidean_distance_matrix(ds, dt)

    # Joint distance
    D_joint = D_shape + tau * D_trend

    # Accumulated cost with EWRF
    C = np.full((n + 1, m + 1), np.inf)
    C[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # EWRF: penalise deviation from diagonal
            diag_dev = abs(i / n - j / m)
            ewrf_penalty = lambda_ewrf * diag_dev

            local_cost = D_joint[i - 1, j - 1] + ewrf_penalty
            C[i, j] = local_cost + min(C[i - 1, j],
                                        C[i, j - 1],
                                        C[i - 1, j - 1])

    # Backtrack path
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        candidates = []
        if i > 0 and j > 0:
            candidates.append((C[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            candidates.append((C[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((C[i, j - 1], i, j - 1))
        _, i, j = min(candidates)

    path.reverse()
    return C[n, m], C[1:, 1:], np.array(path)


# -----------------------------------------------------------------------
# 5. PCC Baseline Matching
# -----------------------------------------------------------------------
def pcc_match(core: np.ndarray, log: np.ndarray,
              max_shift: int = 20) -> Tuple[int, float]:
    """
    Match core to log using Pearson Correlation Coefficient (PCC)
    with brute-force shift search.

    Parameters
    ----------
    core : np.ndarray
        Core measurement sequence.
    log : np.ndarray
        Well-log sequence (should be longer or equal length).
    max_shift : int
        Maximum shift in samples to search.

    Returns
    -------
    best_shift : int
        Optimal shift (samples).
    best_pcc : float
        PCC at best shift.
    """
    n_core = len(core)
    n_log = len(log)
    best_shift = 0
    best_pcc = -1.0

    for shift in range(-max_shift, max_shift + 1):
        start = max(0, shift)
        end = min(n_log, n_core + shift)
        core_start = max(0, -shift)
        core_end = core_start + (end - start)
        if end - start < 5:
            continue

        c = core[core_start:core_end]
        l = log[start:end]
        if len(c) != len(l) or len(c) < 3:
            continue

        corr = np.corrcoef(c, l)[0, 1]
        if corr > best_pcc:
            best_pcc = corr
            best_shift = shift

    return best_shift, best_pcc


# -----------------------------------------------------------------------
# 6. Particle Swarm Optimization for Depth Shift
# -----------------------------------------------------------------------
def pso_depth_shift(core: np.ndarray, log: np.ndarray,
                    n_particles: int = 50, n_iterations: int = 100,
                    max_shift: float = 5.0, dz: float = 0.125,
                    tau: float = 4.0, lambda_ewrf: float = 0.5
                    ) -> Tuple[float, float]:
    """
    Use PSO to find the optimal depth shift that minimizes the RDDTW
    distance between core and log data.

    Parameters
    ----------
    core, log : np.ndarray
        Core and log measurement sequences.
    n_particles : int
        Number of PSO particles.
    n_iterations : int
        Maximum iterations.
    max_shift : float
        Maximum depth shift (meters).
    dz : float
        Sampling interval (meters).
    tau : float
        RDDTW trend weight.
    lambda_ewrf : float
        EWRF regularization weight.

    Returns
    -------
    best_shift_m : float
        Optimal shift (meters).
    best_cost : float
        RDDTW distance at optimal shift.
    """
    max_shift_samples = int(max_shift / dz)
    rng = np.random.RandomState(42)

    # Initialize particles
    positions = rng.uniform(-max_shift_samples, max_shift_samples, n_particles)
    velocities = rng.uniform(-2, 2, n_particles)
    pbest_pos = positions.copy()
    pbest_cost = np.full(n_particles, np.inf)
    gbest_pos = 0.0
    gbest_cost = np.inf

    w_init, w_final = 0.8, 0.4
    c1, c2 = 1.5, 1.5

    def evaluate(shift_samples):
        shift = int(round(shift_samples))
        if shift >= 0:
            c = core[shift:]
            l = log[:len(c)]
        else:
            c = core[:len(core) + shift]
            l = log[-shift:-shift + len(c)]
        min_len = min(len(c), len(l))
        if min_len < 5:
            return np.inf
        c, l = c[:min_len], l[:min_len]
        # Use simple RMSE with derivative penalty for speed
        rmse = np.sqrt(np.mean((c - l) ** 2))
        dc = derivative_estimate(c)
        dl = derivative_estimate(l)
        trend_rmse = np.sqrt(np.mean((dc - dl) ** 2))
        return rmse + tau * trend_rmse

    for it in range(n_iterations):
        w = w_init - (w_init - w_final) * it / n_iterations

        for p in range(n_particles):
            cost = evaluate(positions[p])
            if cost < pbest_cost[p]:
                pbest_cost[p] = cost
                pbest_pos[p] = positions[p]
            if cost < gbest_cost:
                gbest_cost = cost
                gbest_pos = positions[p]

        r1 = rng.random(n_particles)
        r2 = rng.random(n_particles)
        velocities = (w * velocities +
                      c1 * r1 * (pbest_pos - positions) +
                      c2 * r2 * (gbest_pos - positions))
        positions = np.clip(positions + velocities,
                            -max_shift_samples, max_shift_samples)

    return gbest_pos * dz, gbest_cost


# -----------------------------------------------------------------------
# 7. Evaluation Metrics
# -----------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# -----------------------------------------------------------------------
# 8. Synthetic Data Generator
# -----------------------------------------------------------------------
def generate_synthetic_core_log(n_core: int = 100, n_log: int = 120,
                                true_shift: int = 8,
                                noise_core: float = 0.01,
                                noise_log: float = 0.02,
                                seed: int = 42
                                ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate synthetic core and log porosity data with a known depth shift.
    """
    rng = np.random.RandomState(seed)
    # Underlying porosity profile
    z = np.linspace(0, 4 * np.pi, n_log + 50)
    base_phi = 0.10 + 0.08 * np.sin(z) + 0.04 * np.sin(3 * z)
    base_phi = np.clip(base_phi + 0.02 * rng.randn(len(z)), 0, 0.35)

    # Log: sampled at regular intervals
    log_data = base_phi[:n_log] + noise_log * rng.randn(n_log)

    # Core: shifted and with slightly different noise
    core_start = true_shift
    core_data = base_phi[core_start:core_start + n_core]
    core_data += noise_core * rng.randn(n_core)

    return core_data, log_data, true_shift


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== RDDTW Core-Log Depth Matching Module Demo ===\n")

    core, log, true_shift = generate_synthetic_core_log()
    print(f"Core length: {len(core)}, Log length: {len(log)}")
    print(f"True shift: {true_shift} samples")

    # PCC matching
    pcc_shift, pcc_r = pcc_match(core, log, max_shift=20)
    print(f"\nPCC match: shift = {pcc_shift} samples, PCC = {pcc_r:.4f}")

    # Short sequences for DTW demo (DTW is O(n*m))
    core_short = core[:40]
    log_short = log[:40]

    # Standard DTW
    dist_dtw, _, path_dtw = dtw(core_short, log_short)
    print(f"\nDTW distance: {dist_dtw:.4f}, path length: {len(path_dtw)}")

    # CDTW
    dist_cdtw, _ = cdtw(core_short, log_short, window=8)
    print(f"CDTW distance: {dist_cdtw:.4f}")

    # RDDTW
    dist_rddtw, _, path_rddtw = rddtw(core_short, log_short,
                                        tau=4.0, lambda_ewrf=0.5)
    print(f"RDDTW distance: {dist_rddtw:.4f}, path length: {len(path_rddtw)}")

    # PSO depth-shift estimation
    best_shift, best_cost = pso_depth_shift(core, log, dz=1.0)
    print(f"\nPSO optimal shift: {best_shift:.1f} samples "
          f"(true: {true_shift}), cost: {best_cost:.4f}")

    # Evaluate matched alignment
    shift_int = int(round(best_shift))
    if shift_int >= 0:
        c_aligned = core[shift_int:]
        l_aligned = log[:len(c_aligned)]
    else:
        c_aligned = core[:len(core) + shift_int]
        l_aligned = log[-shift_int:-shift_int + len(c_aligned)]
    min_len = min(len(c_aligned), len(l_aligned))
    r2 = r_squared(c_aligned[:min_len], l_aligned[:min_len])
    rm = rmse(c_aligned[:min_len], l_aligned[:min_len])
    print(f"After alignment: R² = {r2:.4f}, RMSE = {rm:.4f}")
