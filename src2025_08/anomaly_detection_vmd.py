#!/usr/bin/env python3
"""
Module 8: Automated Anomaly Detection via Signal Mode Decomposition
===================================================================
Implements ideas from:
  Wang et al., "Automated Anomaly Detection of Multimetallic Tubulars
  in Well Integrity Logs Using Signal Mode Decomposition and
  Physics-Informed Decision Making,"
  Petrophysics, vol. 66, no. 4, pp. 647–661, August 2025.

Key concepts:
  - Variational Mode Decomposition (VMD) for single-channel signals
  - Multivariate VMD (MVMD) for multi-channel VDL data
  - Hierarchical Multiresolution VMD (HMVMD) feature extraction
  - Bayesian decision tree with Markov simplification for
    collar/anomaly classification
  - SNR enhancement through mode separation
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Variational Mode Decomposition (VMD) — single-channel
# ---------------------------------------------------------------------------
def vmd(
    signal: np.ndarray,
    n_modes: int = 4,
    alpha: float = 2000.0,
    tau: float = 0.0,
    n_iter: int = 200,
    tol: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Variational Mode Decomposition (Dragomiretskiy & Zosso, 2013).

    Parameters
    ----------
    signal : 1-D array
    n_modes : int — number of modes K
    alpha : float — bandwidth constraint (penalty parameter)
    tau : float — noise-tolerance (Lagrangian update step)
    n_iter : int — maximum ADMM iterations
    tol : float — convergence tolerance

    Returns
    -------
    modes : ndarray, shape (n_modes, N)
    centre_freqs : ndarray, shape (n_modes,)
    """
    N = len(signal)
    f_hat = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N)

    # Initialise modes in frequency domain
    u_hat = np.zeros((n_modes, N), dtype=complex)
    omega = np.linspace(0, 0.5, n_modes + 2)[1:-1]  # initial centre freqs
    lam_hat = np.zeros(N, dtype=complex)

    for _ in range(n_iter):
        u_hat_old = u_hat.copy()

        for k in range(n_modes):
            # Sum of all other modes
            sum_other = np.sum(u_hat, axis=0) - u_hat[k]
            # Wiener filter update
            numerator = f_hat - sum_other + lam_hat / 2
            denominator = 1 + alpha * (freqs - omega[k]) ** 2
            u_hat[k] = numerator / (denominator + 1e-12)

            # Centre frequency update
            power = np.abs(u_hat[k]) ** 2
            denom = np.sum(power) + 1e-12
            omega[k] = np.sum(np.abs(freqs) * power) / denom

        # Dual ascent
        residual = f_hat - np.sum(u_hat, axis=0)
        lam_hat += tau * residual

        # Convergence check
        change = np.sum(np.abs(u_hat - u_hat_old) ** 2)
        if change < tol:
            break

    modes = np.real(np.fft.ifft(u_hat, axis=1))
    return modes, omega


# ---------------------------------------------------------------------------
# 2. Multivariate VMD (MVMD) — multi-channel extension
# ---------------------------------------------------------------------------
def mvmd(
    data: np.ndarray,
    n_modes: int = 4,
    alpha: float = 2000.0,
    n_iter: int = 150,
    tol: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simplified Multivariate VMD for multi-channel data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, N)

    Returns
    -------
    modes : ndarray, shape (n_modes, n_channels, N)
    centre_freqs : ndarray, shape (n_modes,)
    """
    n_ch, N = data.shape
    # Apply VMD to each channel, enforce shared centre frequencies
    all_modes = np.zeros((n_modes, n_ch, N))
    freq_accum = np.zeros(n_modes)

    for c in range(n_ch):
        modes_c, omega_c = vmd(data[c], n_modes=n_modes, alpha=alpha,
                                n_iter=n_iter, tol=tol)
        all_modes[:, c, :] = modes_c
        freq_accum += omega_c

    centre_freqs = freq_accum / n_ch
    return all_modes, centre_freqs


# ---------------------------------------------------------------------------
# 3. Hierarchical Multiresolution VMD (HMVMD)
# ---------------------------------------------------------------------------
def hmvmd(
    data: np.ndarray,
    levels: int = 2,
    modes_per_level: int = 3,
    alpha: float = 2000.0,
) -> dict:
    """Hierarchical multiresolution VMD.

    Level 0: decompose the raw data into `modes_per_level` modes.
    Level 1+: further decompose selected modes from the previous level.

    Returns
    -------
    dict with keys 'level_0', 'level_1', etc., each containing
    (modes, centre_freqs).
    """
    results = {}
    if data.ndim == 1:
        data = data.reshape(1, -1)

    modes_0, freqs_0 = mvmd(data, n_modes=modes_per_level, alpha=alpha)
    results['level_0'] = (modes_0, freqs_0)

    if levels > 1:
        # Decompose the lowest-frequency mode further
        low_mode = modes_0[0]  # shape (n_ch, N)
        modes_1, freqs_1 = mvmd(low_mode, n_modes=modes_per_level,
                                 alpha=alpha * 2)
        results['level_1'] = (modes_1, freqs_1)

    return results


# ---------------------------------------------------------------------------
# 4. Feature extraction from decomposed modes
# ---------------------------------------------------------------------------
def extract_features(modes: np.ndarray) -> np.ndarray:
    """Extract amplitude and energy features from VMD modes.

    Parameters
    ----------
    modes : ndarray, shape (n_modes, n_channels, N) or (n_modes, N)

    Returns
    -------
    features : ndarray, shape (N, n_features)
    """
    if modes.ndim == 2:
        modes = modes[:, np.newaxis, :]  # add channel dim

    n_modes, n_ch, N = modes.shape
    # Features: per-mode energy across channels
    features = np.zeros((N, n_modes * 2))
    for k in range(n_modes):
        # Mean amplitude across channels
        features[:, 2 * k] = np.mean(np.abs(modes[k]), axis=0)
        # Energy (variance in a sliding window)
        win = 11
        for i in range(N):
            lo, hi = max(0, i - win // 2), min(N, i + win // 2 + 1)
            features[:, 2 * k + 1] = np.var(modes[k, :, lo:hi])

    return features


# ---------------------------------------------------------------------------
# 5. Bayesian decision tree for collar/anomaly classification
# ---------------------------------------------------------------------------
@dataclass
class DecisionResult:
    depth_idx: int
    label: str        # 'nominal', 'collar', 'anomaly', 'artifact'
    confidence: float


def bayesian_decision_tree(
    features: np.ndarray,
    collar_freq_band: int = 2,   # which mode index corresponds to collars
    anomaly_freq_band: int = 0,  # which mode corresponds to anomalies
    collar_threshold: float = 0.5,
    anomaly_threshold: float = 0.3,
    prior_collar_spacing: Optional[float] = None,
) -> List[DecisionResult]:
    """Physics-informed decision tree for classifying well-integrity features.

    Uses Bayesian reasoning with Markov simplification: the probability
    of a collar at depth i depends on whether one was recently detected.

    Parameters
    ----------
    prior_collar_spacing : float, optional
        Expected collar spacing in depth samples.  If set, the Markov
        prior increases collar probability near expected intervals.
    """
    N = features.shape[0]
    collar_energy = features[:, 2 * collar_freq_band]
    anomaly_energy = features[:, 2 * anomaly_freq_band]

    # Normalise
    collar_norm = collar_energy / (collar_energy.max() + 1e-12)
    anomaly_norm = anomaly_energy / (anomaly_energy.max() + 1e-12)

    results: List[DecisionResult] = []
    last_collar_idx = -1000

    for i in range(N):
        # Prior based on expected collar spacing (Markov)
        collar_prior = 0.5
        if prior_collar_spacing and last_collar_idx >= 0:
            dist = i - last_collar_idx
            collar_prior = np.exp(-0.5 * ((dist - prior_collar_spacing) /
                                           (prior_collar_spacing * 0.2)) ** 2)
            collar_prior = np.clip(collar_prior, 0.05, 0.95)

        # Likelihood
        p_collar = collar_norm[i] * collar_prior
        p_anomaly = anomaly_norm[i]

        if p_collar > collar_threshold and p_collar > p_anomaly:
            results.append(DecisionResult(i, 'collar', float(p_collar)))
            last_collar_idx = i
        elif p_anomaly > anomaly_threshold:
            results.append(DecisionResult(i, 'anomaly', float(p_anomaly)))
        else:
            results.append(DecisionResult(i, 'nominal', 1.0 - max(p_collar, p_anomaly)))

    return results


# ---------------------------------------------------------------------------
# 6. SNR computation
# ---------------------------------------------------------------------------
def compute_snr_improvement(
    original: np.ndarray, mode_signal: np.ndarray,
) -> float:
    """Estimate SNR improvement (dB) of extracted mode vs. original."""
    noise_est = original - mode_signal
    sig_power = np.mean(mode_signal ** 2)
    noise_power = np.mean(noise_est ** 2) + 1e-30
    return 10.0 * np.log10(sig_power / noise_power)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_all():
    rng = np.random.default_rng(42)
    N = 500

    # Synthetic VDL-like signal: low-freq anomaly + mid-freq collars + noise
    depth = np.arange(N, dtype=float)

    # Anomaly: slow Gaussian bump
    anomaly = 2.0 * np.exp(-0.5 * ((depth - 200) / 30) ** 2)
    # Collars: periodic sharp features every ~40 samples
    collars = np.zeros(N)
    for c in range(20, N, 40):
        collars += 1.5 * np.exp(-0.5 * ((depth - c) / 3) ** 2)
    # Noise
    noise = rng.normal(0, 0.2, N)
    signal = anomaly + collars + noise

    # Single-channel VMD
    modes, freqs = vmd(signal, n_modes=4, alpha=1000, n_iter=100)
    assert modes.shape == (4, N)
    assert len(freqs) == 4

    # MVMD on 3-channel data
    data_3ch = np.vstack([signal, signal * 0.9 + rng.normal(0, 0.1, N),
                           signal * 0.8 + rng.normal(0, 0.15, N)])
    modes_mv, freqs_mv = mvmd(data_3ch, n_modes=3)
    assert modes_mv.shape == (3, 3, N)

    # HMVMD
    result_h = hmvmd(data_3ch, levels=2, modes_per_level=3)
    assert 'level_0' in result_h and 'level_1' in result_h

    # Feature extraction
    feats = extract_features(modes_mv)
    assert feats.shape[0] == N

    # Bayesian decision tree
    decisions = bayesian_decision_tree(feats, collar_threshold=0.3,
                                        anomaly_threshold=0.2,
                                        prior_collar_spacing=40)
    assert len(decisions) == N
    labels = [d.label for d in decisions]
    assert 'collar' in labels or 'anomaly' in labels, \
        "Should detect at least one collar or anomaly"

    # SNR improvement
    snr_db = compute_snr_improvement(signal, modes[0])
    assert snr_db > 0, "Mode extraction should improve SNR"

    print("[PASS] anomaly_detection_vmd — all tests passed")


if __name__ == "__main__":
    test_all()
