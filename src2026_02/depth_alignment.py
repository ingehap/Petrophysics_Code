"""
Article 4: Dynamic Depth Alignment of Well Logs: A Continuous Optimization
Framework for Enhanced Petrophysical and Rock Physics Interpretation.

Authors: Westeng, Aursand, Viset, and Van Crombrugge (2026)
DOI: 10.30632/PJV67N1-2026a4

Implements the continuous optimization framework for aligning well logs
from different wireline runs/conveyance methods. The cost function balances
correlation improvement with physical smoothness constraints.

Key ideas implemented:
    - Cost function with correlation, absolute shift, and derivative penalties
    - Pearson correlation for same-type curves (Eq. 3)
    - Adam optimizer for continuous depth shift estimation
    - Bulk shift initialization via cross-correlation
    - Gardner transform for density-sonic preprocessing

References
----------
Westeng et al. (2026), Petrophysics, 67(1), 54-67.
Kingma and Ba (2014) for Adam optimizer.
Gardner et al. (1974) for DEN-Vp transform.
"""

import numpy as np
from typing import Optional


def pearson_correlation(f: np.ndarray, g: np.ndarray) -> float:
    """Compute Pearson correlation between two log arrays.

    Parameters
    ----------
    f, g : np.ndarray
        Well log values at the same depth grid.

    Returns
    -------
    float
        Pearson correlation coefficient in [-1, 1].
    """
    f_c = f - np.mean(f)
    g_c = g - np.mean(g)
    denom = np.sqrt(np.sum(f_c**2) * np.sum(g_c**2))
    if denom < 1e-30:
        return 0.0
    return np.sum(f_c * g_c) / denom


def gardner_transform(vp_km_s: np.ndarray) -> np.ndarray:
    """Gardner et al. (1974) empirical velocity-density transform (Eq. 1).

    DEN = 0.31 * Vp^0.25

    Parameters
    ----------
    vp_km_s : np.ndarray
        Compressional velocity in km/s.

    Returns
    -------
    np.ndarray
        Estimated bulk density in g/cc.
    """
    return 0.31 * vp_km_s**0.25


def apply_depth_shift(g: np.ndarray,
                      z: np.ndarray,
                      shift: np.ndarray) -> np.ndarray:
    """Apply a continuous depth shift function to a well log.

    Parameters
    ----------
    g : np.ndarray
        Original log values.
    z : np.ndarray
        Depth positions.
    shift : np.ndarray
        Depth shift at each position (same units as z).

    Returns
    -------
    np.ndarray
        Shifted log values (via linear interpolation).
    """
    shifted_z = z + shift
    return np.interp(z, shifted_z, g, left=g[0], right=g[-1])


def bulk_shift(f: np.ndarray,
               g: np.ndarray,
               z: np.ndarray,
               max_shift: float = 10.0) -> float:
    """Find the constant (bulk) shift that maximizes correlation.

    Parameters
    ----------
    f : np.ndarray
        Reference log.
    g : np.ndarray
        Log to be shifted.
    z : np.ndarray
        Depth positions.
    max_shift : float
        Maximum allowed shift (in depth units).

    Returns
    -------
    float
        Optimal bulk shift value.
    """
    dz = z[1] - z[0]
    max_lag = int(max_shift / dz)
    best_corr = -2.0
    best_shift = 0.0

    for lag in range(-max_lag, max_lag + 1):
        shift_val = lag * dz
        g_shifted = apply_depth_shift(g, z, np.full_like(z, shift_val))
        corr = pearson_correlation(f, g_shifted)
        if corr > best_corr:
            best_corr = corr
            best_shift = shift_val

    return best_shift


def cost_function(f: np.ndarray,
                  g: np.ndarray,
                  z: np.ndarray,
                  shift: np.ndarray,
                  alpha: float = 0.0,
                  beta: float = 1.0,
                  opposite_trend: bool = False) -> float:
    """Compute the depth alignment cost function (Eqs. 2-3).

    C(l) = -ρ(f, g_shifted) + α·mean(|l|) + β·mean(|l'|)

    where ρ is the Pearson correlation, l is the depth shift, and l' is
    its derivative. α penalizes large absolute shifts, β penalizes rapid
    shift changes (promotes smoothness).

    Parameters
    ----------
    f : np.ndarray
        Reference log values.
    g : np.ndarray
        Log to be aligned.
    z : np.ndarray
        Depth positions.
    shift : np.ndarray
        Current depth shift function l(z).
    alpha : float
        Weight for absolute shift penalty.
    beta : float
        Weight for shift derivative (smoothness) penalty.
    opposite_trend : bool
        If True, maximize negative correlation (e.g., DEN vs. AC).

    Returns
    -------
    float
        Cost function value (lower is better).
    """
    g_shifted = apply_depth_shift(g, z, shift)
    corr = pearson_correlation(f, g_shifted)
    if opposite_trend:
        corr = -corr

    # Correlation term (negative because we minimize)
    cost_corr = -corr

    # Absolute shift penalty
    cost_abs = alpha * np.mean(np.abs(shift))

    # Derivative (smoothness) penalty
    dz = np.diff(z)
    dl = np.diff(shift)
    cost_deriv = beta * np.mean(np.abs(dl / dz))

    return cost_corr + cost_abs + cost_deriv


def align_logs_adam(f: np.ndarray,
                    g: np.ndarray,
                    z: np.ndarray,
                    alpha: float = 0.0,
                    beta: float = 1.0,
                    learning_rate: float = 0.01,
                    max_iter: int = 500,
                    opposite_trend: bool = False,
                    initial_bulk_shift: bool = True,
                    max_shift: float = 10.0) -> dict:
    """Align well logs using Adam optimizer (Kingma and Ba, 2014).

    Minimizes the cost function C(l) that balances correlation improvement
    with smoothness constraints on the shift function.

    Parameters
    ----------
    f : np.ndarray
        Reference well log.
    g : np.ndarray
        Well log to be aligned to f.
    z : np.ndarray
        Depth positions (m).
    alpha : float
        Absolute shift penalty weight.
    beta : float
        Smoothness penalty weight. Larger values produce smoother shifts.
    learning_rate : float
        Adam learning rate.
    max_iter : int
        Maximum optimization iterations.
    opposite_trend : bool
        True for curves with opposite trends (e.g., density vs. sonic).
    initial_bulk_shift : bool
        If True, initialize with the optimal constant shift.
    max_shift : float
        Maximum allowed absolute shift (m).

    Returns
    -------
    dict with keys:
        'shift' : np.ndarray - optimal shift function l(z)
        'correlation_before' : float - initial correlation
        'correlation_after' : float - final correlation
        'cost_history' : list[float] - cost function at each iteration
    """
    n = len(z)
    dz = z[1] - z[0]
    eps = 1e-8

    # Initialize shift
    if initial_bulk_shift:
        bs = bulk_shift(f, g, z, max_shift)
        shift = np.full(n, bs)
    else:
        shift = np.zeros(n)

    corr_before = pearson_correlation(f, g)

    # Adam optimizer state
    m = np.zeros(n)   # First moment
    v = np.zeros(n)   # Second moment
    beta1, beta2 = 0.9, 0.999

    cost_history = []

    for t_step in range(1, max_iter + 1):
        # Compute gradient numerically
        grad = np.zeros(n)
        h = dz * 0.1
        cost_current = cost_function(f, g, z, shift, alpha, beta, opposite_trend)
        cost_history.append(cost_current)

        for i in range(n):
            shift_plus = shift.copy()
            shift_plus[i] += h
            cost_plus = cost_function(f, g, z, shift_plus, alpha, beta,
                                      opposite_trend)
            grad[i] = (cost_plus - cost_current) / h

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t_step)
        v_hat = v / (1 - beta2**t_step)

        shift = shift - learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        # Clip to maximum shift
        shift = np.clip(shift, -max_shift, max_shift)

        # Check convergence
        if t_step > 10 and abs(cost_history[-1] - cost_history[-2]) < 1e-8:
            break

    g_aligned = apply_depth_shift(g, z, shift)
    corr_after = pearson_correlation(f, g_aligned)
    if opposite_trend:
        corr_after = -corr_after

    return {
        "shift": shift,
        "correlation_before": corr_before,
        "correlation_after": corr_after,
        "cost_history": cost_history,
    }


def align_multiple_logs(reference: np.ndarray,
                        logs: list,
                        z: np.ndarray,
                        alpha: float = 0.0,
                        beta: float = 1.0,
                        opposite_trends: Optional[list] = None,
                        **kwargs) -> list:
    """Align multiple logs to a reference curve.

    Parameters
    ----------
    reference : np.ndarray
        Reference well log.
    logs : list[np.ndarray]
        List of logs to align.
    z : np.ndarray
        Depth positions.
    alpha, beta : float
        Cost function weights.
    opposite_trends : list[bool], optional
        Whether each log has opposite trend to reference.

    Returns
    -------
    list[dict]
        Alignment results for each log.
    """
    if opposite_trends is None:
        opposite_trends = [False] * len(logs)

    results = []
    for log, opp in zip(logs, opposite_trends):
        result = align_logs_adam(reference, log, z, alpha=alpha, beta=beta,
                                opposite_trend=opp, **kwargs)
        results.append(result)

    return results
