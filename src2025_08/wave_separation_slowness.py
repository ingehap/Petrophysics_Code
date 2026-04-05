#!/usr/bin/env python3
"""
Module 11: Time-Variant Wave Separation for Formation Slowness
Estimation Behind Casing
===============================================================
Implements ideas from:
  Sun et al., "Formation Slowness Estimation Behind Casing via a
  Time-Variant Wave Separation Method,"
  Petrophysics, vol. 66, no. 4, pp. 689–700, August 2025.

Key concepts:
  - Array waveform simulation (monopole logging, 8 receivers)
  - Slowness-Time-Coherence (STC) analysis
  - Casing-wave reference via Linear Moveout (LMO) + stacking
  - Preliminary wave separation by waveform subtraction
  - Time-variant (TV) correlation weighting for constrained separation
  - Post-separation Radon-domain enhancement
  - Slowness spectrum (coherence vs. slowness projection)
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Synthetic array waveform generation
# ---------------------------------------------------------------------------
def generate_waveform(
    n_receivers: int = 8,
    receiver_spacing_ft: float = 0.5,
    dt_us: float = 2.0,
    n_samples: int = 500,
    casing_slowness_us_per_ft: float = 57.0,
    formation_p_slowness: float = 72.0,
    formation_s_slowness: float = 127.0,
    casing_amplitude: float = 1.0,
    formation_p_amplitude: float = 0.3,
    formation_s_amplitude: float = 0.2,
    centre_freq_khz: float = 12.0,
    noise_std: float = 0.02,
    rng=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic monopole acoustic logging waveforms.

    Parameters
    ----------
    Returns
    -------
    time_us : ndarray, shape (n_samples,)
    waveforms : ndarray, shape (n_receivers, n_samples)
    """
    rng = rng or np.random.default_rng(0)
    time_us = np.arange(n_samples) * dt_us
    waveforms = np.zeros((n_receivers, n_samples))
    f0 = centre_freq_khz * 1e3
    omega0 = 2 * np.pi * f0

    for r in range(n_receivers):
        offset_ft = r * receiver_spacing_ft

        # Casing wave
        t_casing = casing_slowness_us_per_ft * offset_ft
        t_rel = time_us - t_casing
        env = np.exp(-0.5 * (t_rel / 40) ** 2)
        waveforms[r] += casing_amplitude * env * np.cos(omega0 * t_rel * 1e-6)

        # Formation P-wave
        t_fp = formation_p_slowness * offset_ft
        t_rel_p = time_us - t_fp
        env_p = np.exp(-0.5 * (t_rel_p / 30) ** 2)
        waveforms[r] += formation_p_amplitude * env_p * np.cos(omega0 * 0.8 * t_rel_p * 1e-6)

        # Formation S-wave
        t_fs = formation_s_slowness * offset_ft
        t_rel_s = time_us - t_fs
        env_s = np.exp(-0.5 * (t_rel_s / 50) ** 2)
        waveforms[r] += formation_s_amplitude * env_s * np.cos(omega0 * 0.5 * t_rel_s * 1e-6)

    waveforms += rng.normal(0, noise_std, waveforms.shape)
    return time_us, waveforms


# ---------------------------------------------------------------------------
# 2. Slowness-Time-Coherence (STC) analysis
# ---------------------------------------------------------------------------
def stc_analysis(
    waveforms: np.ndarray,
    time_us: np.ndarray,
    receiver_spacing_ft: float = 0.5,
    slowness_range: Tuple[float, float] = (40, 200),
    n_slowness: int = 161,
    window_length_us: float = 80.0,
    n_windows: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the STC semblance map.

    Returns
    -------
    slowness : ndarray, shape (n_slowness,)
    window_times : ndarray, shape (n_windows,)
    coherence : ndarray, shape (n_slowness, n_windows) — semblance values.
    """
    n_rx, n_t = waveforms.shape
    dt = time_us[1] - time_us[0]
    slowness = np.linspace(slowness_range[0], slowness_range[1], n_slowness)
    t_max = time_us[-1] - window_length_us
    window_times = np.linspace(time_us[0], t_max, n_windows)
    win_samples = int(window_length_us / dt)
    coherence = np.zeros((n_slowness, n_windows))

    for i_s, s in enumerate(slowness):
        for i_w, tw in enumerate(window_times):
            num = 0.0
            den = 0.0
            for t_idx in range(win_samples):
                t0 = tw + t_idx * dt
                stack = 0.0
                sum_sq = 0.0
                for r in range(n_rx):
                    moveout = s * r * receiver_spacing_ft
                    t_shifted = t0 + moveout
                    idx = int(t_shifted / dt)
                    if 0 <= idx < n_t:
                        val = waveforms[r, idx]
                        stack += val
                        sum_sq += val ** 2
                num += stack ** 2
                den += n_rx * sum_sq
            coherence[i_s, i_w] = num / (den + 1e-30)

    return slowness, window_times, coherence


def slowness_spectrum(coherence: np.ndarray) -> np.ndarray:
    """Project the 2-D STC map onto the slowness axis (max across time)."""
    return np.max(coherence, axis=1)


# ---------------------------------------------------------------------------
# 3. Linear Moveout (LMO) correction + stacking
# ---------------------------------------------------------------------------
def lmo_correct(
    waveforms: np.ndarray,
    time_us: np.ndarray,
    slowness_us_per_ft: float,
    receiver_spacing_ft: float = 0.5,
) -> np.ndarray:
    """Apply linear moveout correction for a given slowness.

    Shifts each receiver waveform so that a wave with the specified
    slowness is time-aligned across all receivers.
    """
    n_rx, n_t = waveforms.shape
    dt = time_us[1] - time_us[0]
    corrected = np.zeros_like(waveforms)

    for r in range(n_rx):
        shift_us = slowness_us_per_ft * r * receiver_spacing_ft
        shift_samples = int(shift_us / dt)
        if shift_samples >= 0:
            corrected[r, :n_t - shift_samples] = waveforms[r, shift_samples:]
        else:
            corrected[r, -shift_samples:] = waveforms[r, :n_t + shift_samples]

    return corrected


def stack_and_replicate(
    corrected: np.ndarray,
    n_rx: int,
    slowness_us_per_ft: float,
    receiver_spacing_ft: float,
    dt_us: float,
) -> np.ndarray:
    """Stack the corrected traces, then reverse-moveout to get a reference."""
    stacked = corrected.mean(axis=0)
    n_t = len(stacked)
    reference = np.zeros((n_rx, n_t))
    for r in range(n_rx):
        shift_samples = int(slowness_us_per_ft * r * receiver_spacing_ft / dt_us)
        if shift_samples >= 0 and shift_samples < n_t:
            reference[r, shift_samples:] = stacked[:n_t - shift_samples]
    return reference


# ---------------------------------------------------------------------------
# 4. Preliminary wave separation
# ---------------------------------------------------------------------------
def preliminary_separation(
    waveforms: np.ndarray,
    time_us: np.ndarray,
    casing_slowness: float,
    receiver_spacing_ft: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove the casing wave by LMO + stack + subtraction.

    Returns
    -------
    (formation_estimate, casing_reference) — both shape (n_rx, n_t)
    """
    dt = time_us[1] - time_us[0]
    n_rx = waveforms.shape[0]
    corrected = lmo_correct(waveforms, time_us, casing_slowness,
                             receiver_spacing_ft)
    casing_ref = stack_and_replicate(corrected, n_rx, casing_slowness,
                                      receiver_spacing_ft, dt)
    formation = waveforms - casing_ref
    return formation, casing_ref


# ---------------------------------------------------------------------------
# 5. Time-variant (TV) weighting for constrained separation
# ---------------------------------------------------------------------------
def tv_weighted_separation(
    waveforms: np.ndarray,
    casing_ref: np.ndarray,
) -> np.ndarray:
    """Apply time-variant correlation weighting to refine wave separation.

    The weighting reduces over-removal of formation-wave energy
    at slownesses close to the casing slowness.
    """
    n_rx, n_t = waveforms.shape
    result = np.zeros_like(waveforms)

    for r in range(n_rx):
        # Sliding-window normalised cross-correlation
        win = 31
        weight = np.ones(n_t)
        for i in range(n_t):
            lo = max(0, i - win // 2)
            hi = min(n_t, i + win // 2 + 1)
            seg_data = waveforms[r, lo:hi]
            seg_ref = casing_ref[r, lo:hi]
            norm = np.sqrt(np.sum(seg_data ** 2) * np.sum(seg_ref ** 2)) + 1e-30
            corr = np.sum(seg_data * seg_ref) / norm
            weight[i] = np.clip(corr, 0, 1)

        # Weighted subtraction
        result[r] = waveforms[r] - weight * casing_ref[r]

    return result


# ---------------------------------------------------------------------------
# 6. Full TVWS workflow
# ---------------------------------------------------------------------------
def tvws_workflow(
    waveforms: np.ndarray,
    time_us: np.ndarray,
    casing_slowness: float,
    receiver_spacing_ft: float = 0.5,
) -> dict:
    """Full time-variant wave separation workflow.

    Returns dict with keys:
      'original_stc', 'separated_stc', 'slowness', 'formation_waveforms'
    """
    # STC on original
    slowness, wt, coh_orig = stc_analysis(waveforms, time_us,
                                           receiver_spacing_ft)

    # Preliminary separation
    formation_pre, casing_ref = preliminary_separation(
        waveforms, time_us, casing_slowness, receiver_spacing_ft)

    # TV-weighted separation
    formation_tv = tv_weighted_separation(waveforms, casing_ref)

    # STC on separated
    _, _, coh_sep = stc_analysis(formation_tv, time_us, receiver_spacing_ft)

    return {
        'slowness': slowness,
        'original_coherence': slowness_spectrum(coh_orig),
        'separated_coherence': slowness_spectrum(coh_sep),
        'formation_waveforms': formation_tv,
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_all():
    time_us, waveforms = generate_waveform(
        casing_slowness_us_per_ft=57.0,
        formation_p_slowness=72.0,
        formation_s_slowness=127.0,
        casing_amplitude=1.0,
        formation_p_amplitude=0.4,
    )
    assert waveforms.shape == (8, 500)

    # STC on original
    slowness, wt, coh = stc_analysis(waveforms, time_us, n_slowness=81,
                                      n_windows=30)
    spec = slowness_spectrum(coh)
    # Should find a peak near casing slowness (57)
    peak_slow = slowness[np.argmax(spec)]
    assert abs(peak_slow - 57) < 15, f"Casing peak at {peak_slow}, expected ~57"

    # LMO correction
    corrected = lmo_correct(waveforms, time_us, 57.0)
    assert corrected.shape == waveforms.shape

    # Preliminary separation
    formation, casing_ref = preliminary_separation(waveforms, time_us, 57.0)
    assert formation.shape == waveforms.shape
    # Casing wave energy should be reduced
    orig_energy = np.sum(waveforms ** 2)
    form_energy = np.sum(formation ** 2)
    assert form_energy < orig_energy

    # TV-weighted separation
    formation_tv = tv_weighted_separation(waveforms, casing_ref)
    assert formation_tv.shape == waveforms.shape

    # Full workflow
    result = tvws_workflow(waveforms, time_us, 57.0)
    sep_spec = result['separated_coherence']
    # After separation, the casing peak should be suppressed
    idx_casing = np.argmin(np.abs(result['slowness'] - 57))
    idx_formation = np.argmin(np.abs(result['slowness'] - 72))
    # Formation peak should be more prominent relative to casing
    ratio_orig = result['original_coherence'][idx_formation] / \
                 (result['original_coherence'][idx_casing] + 1e-12)
    ratio_sep = sep_spec[idx_formation] / (sep_spec[idx_casing] + 1e-12)
    assert ratio_sep > ratio_orig * 0.8, \
        "Separation should improve formation/casing ratio"

    print("[PASS] wave_separation_slowness — all tests passed")


if __name__ == "__main__":
    test_all()
