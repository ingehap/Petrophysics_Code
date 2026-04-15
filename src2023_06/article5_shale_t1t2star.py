"""
article5_shale_t1t2star.py
===========================
Implementation of ideas from:

    Zamiri, M.S., Guo, J., Marica, F., Romero-Zeron, L., Balcom, B.J.
    "Shale Characterization Using T1-T2* Magnetic Resonance Relaxation
    Correlation Measurement at Low and High Magnetic Fields"
    Petrophysics, Vol. 64, No. 3 (June 2023), pp. 384-401
    DOI: 10.30632/PJV64N3-2023a5

Key ideas implemented:

  * Effective transverse relaxation (Eq. 1)
        1/T2* = 1/T2 + gamma*dB0 + gamma*dchi*B0
  * Saturation-recovery / FID forward model used in T1-T2*
        S(tau_r, t) = sum_i  A_i * (1 - exp(-tau_r/T1_i)) * exp(-t/T2*_i)
  * Look-Locker T1*-T2* model (Eq. 2a)
        1/T1* = 1/T1 - ln(cos(alpha)) / tau
  * A simple non-negative least squares fit to recover the amplitudes
    of three known relaxation populations (kerogen / oil / water)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# T2* from T2 and field inhomogeneities (Eq. 1)
# ---------------------------------------------------------------------------
def t2_star(t2: float, gamma: float, dB0: float = 0.0,
            dchi: float = 0.0, B0: float = 0.05) -> float:
    """
    Effective transverse relaxation time T2*.

        1/T2* = 1/T2 + gamma*dB0 + gamma*dchi*B0

    All quantities in SI units.  Returns T2* in seconds.
    """
    inv = 1.0 / t2 + gamma * dB0 + gamma * dchi * B0
    return 1.0 / inv


# ---------------------------------------------------------------------------
# Look-Locker effective T1* (Eq. 2a)
# ---------------------------------------------------------------------------
def t1_star_look_locker(t1: float, alpha_rad: float, tau: float) -> float:
    """1/T1* = 1/T1 - ln(cos(alpha)) / tau   (note sign: cos<1 -> ln<0)."""
    return 1.0 / (1.0 / t1 - np.log(np.cos(alpha_rad)) / tau)


# ---------------------------------------------------------------------------
# Forward 2-D T1-T2* signal
# ---------------------------------------------------------------------------
def synthetic_t1t2star_signal(amplitudes: np.ndarray,
                              t1_list: np.ndarray,
                              t2s_list: np.ndarray,
                              tau_r: np.ndarray,
                              t_acq: np.ndarray) -> np.ndarray:
    """
    Build a synthetic 2D saturation-recovery + FID signal matrix.

    S[i, j] = sum_k A_k * (1 - exp(-tau_r[i]/T1_k)) * exp(-t_acq[j]/T2s_k)
    """
    amp = np.asarray(amplitudes, dtype=float)
    t1 = np.asarray(t1_list, dtype=float)
    t2s = np.asarray(t2s_list, dtype=float)
    tau_r = np.asarray(tau_r, dtype=float)
    t_acq = np.asarray(t_acq, dtype=float)

    sat = 1.0 - np.exp(-tau_r[:, None] / t1[None, :])           # (n_tau, K)
    decay = np.exp(-t_acq[:, None] / t2s[None, :])              # (n_t,   K)
    # S[i, j] = sum_k A_k * sat[i, k] * decay[j, k]
    return (sat * amp[None, :]) @ decay.T                       # (n_tau, n_t)


# ---------------------------------------------------------------------------
# NNLS-style amplitude inversion when relaxation times are known
# ---------------------------------------------------------------------------
def fit_amplitudes(signal: np.ndarray,
                   t1_list: np.ndarray,
                   t2s_list: np.ndarray,
                   tau_r: np.ndarray,
                   t_acq: np.ndarray) -> np.ndarray:
    """
    Recover the population amplitudes A_k given the (known) relaxation
    times.  Solves a non-negative least-squares problem in closed form
    using a simple projected linear inversion (sufficient for the
    well-conditioned synthetic test).
    """
    K = len(t1_list)
    # Build a flattened design matrix M of shape (n_tau*n_t, K)
    rows = []
    for k in range(K):
        kernel = np.outer(1.0 - np.exp(-tau_r / t1_list[k]),
                          np.exp(-t_acq / t2s_list[k]))
        rows.append(kernel.ravel())
    M = np.stack(rows, axis=1)
    y = signal.ravel()
    amp, *_ = np.linalg.lstsq(M, y, rcond=None)
    # Project to non-negative
    amp = np.clip(amp, 0.0, None)
    return amp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    """Synthetic-data test for module 5 (Shale T1-T2*)."""
    print("[article5] testing T2* from T2 + dB0 + dchi*B0 ...")
    gamma_h = 2.675e8                     # rad/s/T  (1H)
    t2 = 1e-3                              # 1 ms
    t2s = t2_star(t2, gamma_h, dB0=0, dchi=0)
    assert abs(t2s - t2) < 1e-12, "no inhomogeneity -> T2* = T2"
    t2s = t2_star(t2, gamma_h, dB0=1e-6, dchi=0)
    assert t2s < t2, "with inhomogeneity T2* must be shorter than T2"

    print("[article5] testing Look-Locker T1* ...")
    t1 = 0.5
    t1s = t1_star_look_locker(t1, alpha_rad=np.deg2rad(10), tau=0.005)
    assert t1s < t1
    print(f"           T1={t1:.3f}s  alpha=10deg  tau=5ms  -> "
          f"T1*={t1s:.3f}s")

    print("[article5] inverting a synthetic 2-D T1-T2* dataset ...")
    # 3 populations: kerogen (very short T1, very short T2*)
    #                oil     (medium)
    #                water   (long)
    amp_true = np.array([1.0, 0.6, 0.4])
    t1_list = np.array([5e-4, 0.05, 0.5])      # s
    t2s_list = np.array([5e-5, 1e-3, 5e-3])    # s

    tau_r = np.logspace(-4, 0, 25)
    t_acq = np.logspace(-5, -2, 60)

    signal_clean = synthetic_t1t2star_signal(amp_true, t1_list, t2s_list,
                                             tau_r, t_acq)
    rng = np.random.default_rng(7)
    signal = signal_clean + 1e-4 * rng.standard_normal(signal_clean.shape)

    amp_hat = fit_amplitudes(signal, t1_list, t2s_list, tau_r, t_acq)
    print(f"           true amplitudes      = {amp_true}")
    print(f"           recovered amplitudes = {np.round(amp_hat, 3)}")
    err = np.max(np.abs(amp_hat - amp_true) / amp_true)
    assert err < 0.05, f"amplitude recovery error too large: {err:.3f}"
    print("[article5] OK")


if __name__ == "__main__":
    test_all()
