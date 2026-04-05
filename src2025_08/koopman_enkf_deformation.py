#!/usr/bin/env python3
"""
Module 9: Koopman-Operator Modeling with Ensemble Kalman Filter
for Through-Tubing Casing Deformation Inspection
================================================================
Implements ideas from:
  Manh et al., "Through-Tubing Casing Deformation Inspection Based
  on Data-Driven Koopman Modeling and Ensemble Kalman Filter,"
  Petrophysics, vol. 66, no. 4, pp. 662–676, August 2025.

Key concepts:
  - Casing deformation parameterised by eccentricity (Ecc),
    eccentricity direction (θ), and ovality factor (Def)
  - Dynamic Mode Decomposition (DMD) to approximate the
    Koopman operator for the state transition model
  - Ensemble Kalman Filter (EnKF) for sequential state estimation
  - Forward observation model using simplified EM physics
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 1. State parameterisation
# ---------------------------------------------------------------------------
def pack_state(theta: float, deformation: float, eccentricity: float) -> np.ndarray:
    """Pack DEC state variables into a vector.

    Parameters
    ----------
    theta : float — eccentricity direction (radians, [0, 2π])
    deformation : float — ovality factor Def = R_B / R_A  (0 < Def ≤ 1)
    eccentricity : float — eccentricity ratio Ecc ∈ [0, 1]
    """
    return np.array([theta, deformation, eccentricity])


def unpack_state(x: np.ndarray) -> Tuple[float, float, float]:
    return float(x[0]), float(x[1]), float(x[2])


# ---------------------------------------------------------------------------
# 2. Synthetic state evolution (ground truth for testing)
# ---------------------------------------------------------------------------
def generate_true_states(
    n_depths: int = 200,
    theta0: float = 0.5,
    def0: float = 0.95,
    ecc0: float = 0.1,
    rng=None,
) -> np.ndarray:
    """Generate a smooth ground-truth state trajectory.

    Returns
    -------
    ndarray, shape (n_depths, 3)
    """
    rng = rng or np.random.default_rng(0)
    states = np.zeros((n_depths, 3))
    states[0] = [theta0, def0, ecc0]

    for n in range(1, n_depths):
        # Smooth random walk with mild drift
        states[n, 0] = states[n - 1, 0] + rng.normal(0, 0.02)
        states[n, 1] = np.clip(states[n - 1, 1] + rng.normal(0, 0.002), 0.7, 1.0)
        states[n, 2] = np.clip(states[n - 1, 2] + rng.normal(0, 0.005), 0.0, 1.0)

    return states


# ---------------------------------------------------------------------------
# 3. Forward observation model (simplified EM flux density)
# ---------------------------------------------------------------------------
def observation_model(
    x: np.ndarray,
    n_sensors: int = 16,
    noise_std: float = 0.0,
    rng=None,
) -> np.ndarray:
    """Compute the magnetic flux density observed by DEC Hall sensors.

    The observation depends on eccentricity (distance to casing varies
    azimuthally) and deformation (elliptical casing shape).

    Parameters
    ----------
    x : ndarray, shape (3,) — [theta, Def, Ecc]

    Returns
    -------
    ndarray, shape (n_sensors,) — flux density readings.
    """
    rng = rng or np.random.default_rng(1)
    theta, Def, Ecc = x
    az = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)

    # Elliptical casing radius: R(φ) based on Def
    R_mean = 1.0   # normalised
    a = R_mean / np.sqrt(Def + 1e-12)
    b = R_mean * np.sqrt(Def + 1e-12)
    R_casing = a * b / np.sqrt((a * np.sin(az)) ** 2 + (b * np.cos(az)) ** 2)

    # Tubing offset due to eccentricity
    dx = Ecc * np.cos(theta)
    dy = Ecc * np.sin(theta)
    # Distance from each sensor (on tubing) to casing wall
    sensor_x = 0.3 * np.cos(az) + dx   # tubing radius 0.3 normalised
    sensor_y = 0.3 * np.sin(az) + dy
    dist = R_casing - np.sqrt(sensor_x ** 2 + sensor_y ** 2)
    dist = np.clip(dist, 0.01, None)

    # Flux density ∝ 1 / distance (simplified)
    flux = 1.0 / dist

    if noise_std > 0:
        flux += rng.normal(0, noise_std, flux.shape)
    return flux


# ---------------------------------------------------------------------------
# 4. Dynamic Mode Decomposition (DMD) — Koopman approximation
# ---------------------------------------------------------------------------
def dmd_fit(X: np.ndarray, Y: np.ndarray, rank: Optional[int] = None) -> np.ndarray:
    """Fit a DMD model: Y ≈ A · X.

    Parameters
    ----------
    X : ndarray, shape (state_dim, n_snapshots-1)  — states at n
    Y : ndarray, shape (state_dim, n_snapshots-1)  — states at n+1
    rank : int, optional — truncation rank for SVD

    Returns
    -------
    A : ndarray, shape (state_dim, state_dim) — Koopman matrix approximation.
    """
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    if rank is not None:
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
    S_inv = np.diag(1.0 / (S + 1e-12))
    A = Y @ Vh.T @ S_inv @ U.T
    return A


def koopman_transition(A: np.ndarray, x: np.ndarray, noise_std: float = 0.0,
                        rng=None) -> np.ndarray:
    """One-step state transition using the Koopman matrix."""
    rng = rng or np.random.default_rng(2)
    x_next = A @ x
    if noise_std > 0:
        x_next += rng.normal(0, noise_std, x.shape)
    return x_next


# ---------------------------------------------------------------------------
# 5. Ensemble Kalman Filter (EnKF)
# ---------------------------------------------------------------------------
def enkf_update(
    ensemble: np.ndarray,
    observation: np.ndarray,
    obs_func,
    obs_noise_std: float = 0.05,
    rng=None,
) -> np.ndarray:
    """One EnKF analysis step.

    Parameters
    ----------
    ensemble : ndarray, shape (n_ens, state_dim)
    observation : ndarray, shape (n_obs,)
    obs_func : callable(state) → observation
    obs_noise_std : float

    Returns
    -------
    updated_ensemble : ndarray
    """
    rng = rng or np.random.default_rng(3)
    n_ens, state_dim = ensemble.shape
    n_obs = len(observation)

    # Predicted observations
    H = np.zeros((n_ens, n_obs))
    for i in range(n_ens):
        H[i] = obs_func(ensemble[i])

    # Ensemble means
    x_mean = ensemble.mean(axis=0)
    h_mean = H.mean(axis=0)

    # Anomalies
    X_a = ensemble - x_mean
    H_a = H - h_mean

    # Kalman gain
    P_HH = (H_a.T @ H_a) / (n_ens - 1) + obs_noise_std ** 2 * np.eye(n_obs)
    P_XH = (X_a.T @ H_a) / (n_ens - 1)
    K = P_XH @ np.linalg.solve(P_HH, np.eye(n_obs))

    # Update each ensemble member with perturbed observations
    updated = np.zeros_like(ensemble)
    for i in range(n_ens):
        obs_pert = observation + rng.normal(0, obs_noise_std, n_obs)
        innovation = obs_pert - H[i]
        updated[i] = ensemble[i] + K @ innovation

    return updated


# ---------------------------------------------------------------------------
# 6. Full estimation pipeline
# ---------------------------------------------------------------------------
def run_estimation(
    observations: np.ndarray,
    A_koopman: np.ndarray,
    n_ensemble: int = 50,
    obs_noise_std: float = 0.05,
    process_noise_std: float = 0.01,
) -> np.ndarray:
    """Run the Koopman + EnKF estimation over a depth sequence.

    Parameters
    ----------
    observations : ndarray, shape (n_depths, n_sensors)
    A_koopman : ndarray, shape (3, 3)

    Returns
    -------
    estimated_states : ndarray, shape (n_depths, 3)
    """
    n_depths, n_sensors = observations.shape
    state_dim = 3
    rng = np.random.default_rng(10)

    # Initialise ensemble around a prior guess
    x0 = np.array([0.5, 0.95, 0.1])
    ensemble = x0 + rng.normal(0, 0.1, (n_ensemble, state_dim))
    ensemble[:, 1] = np.clip(ensemble[:, 1], 0.7, 1.0)
    ensemble[:, 2] = np.clip(ensemble[:, 2], 0.0, 1.0)

    estimated = np.zeros((n_depths, state_dim))

    for n in range(n_depths):
        # Forecast step (Koopman transition)
        for i in range(n_ensemble):
            ensemble[i] = koopman_transition(A_koopman, ensemble[i],
                                              noise_std=process_noise_std, rng=rng)
            ensemble[i, 1] = np.clip(ensemble[i, 1], 0.7, 1.0)
            ensemble[i, 2] = np.clip(ensemble[i, 2], 0.0, 1.0)

        # Analysis step (EnKF update)
        obs_func = lambda x: observation_model(x, n_sensors=n_sensors)
        ensemble = enkf_update(ensemble, observations[n], obs_func,
                                obs_noise_std=obs_noise_std, rng=rng)
        estimated[n] = ensemble.mean(axis=0)

    return estimated


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_all():
    n_depths = 100
    n_sensors = 16
    rng = np.random.default_rng(42)

    # Generate true states
    true_states = generate_true_states(n_depths, rng=rng)
    assert true_states.shape == (n_depths, 3)

    # Generate observations
    observations = np.zeros((n_depths, n_sensors))
    for n in range(n_depths):
        observations[n] = observation_model(true_states[n], n_sensors,
                                             noise_std=0.05, rng=rng)

    # Fit DMD/Koopman from true states (training phase)
    X_train = true_states[:-1].T
    Y_train = true_states[1:].T
    A = dmd_fit(X_train, Y_train, rank=3)
    assert A.shape == (3, 3)

    # One-step prediction test
    x_pred = koopman_transition(A, true_states[0])
    assert x_pred.shape == (3,)

    # Run EnKF estimation
    estimated = run_estimation(observations, A, n_ensemble=30,
                                obs_noise_std=0.05)
    assert estimated.shape == (n_depths, 3)

    # Check estimation accuracy (eccentricity)
    ecc_error = np.abs(estimated[:, 2] - true_states[:, 2])
    mean_ecc_error = np.mean(ecc_error)
    assert mean_ecc_error < 0.3, f"Mean Ecc error {mean_ecc_error:.3f} too large"

    # Deformation accuracy
    def_error = np.abs(estimated[:, 1] - true_states[:, 1])
    mean_def_error = np.mean(def_error)
    assert mean_def_error < 0.15, f"Mean Def error {mean_def_error:.3f} too large"

    print("[PASS] koopman_enkf_deformation — all tests passed")


if __name__ == "__main__":
    test_all()
