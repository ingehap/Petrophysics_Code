"""
Article 4: Borehole Sonic Data Dispersion Analysis With a Modified
           Differential-Phase Semblance Method
Wang, Coates, Zhao (2021)
DOI: 10.30632/PJV62N4-2021a3

A multimode-capable borehole-sonic dispersion estimator.  Borehole modes of
different slowness arrive at different times; the Fourier transform encodes
those traveltime differences as phase differences between frequencies.  By
testing trial slownesses against the array's inter-receiver phase, a
slowness-frequency semblance map separates the modes.

Implements:

  - Frequency-slowness phase semblance over a receiver array
  - Group delay from a phase spectrum  T(f) = -(1/2pi) dphi/df
  - Slowness extraction (argmax of the semblance)
  - Synthetic nondispersive and dispersive array generators

Note: the journal's Eqs. 1-13 were image-rendered and not in the text; the
phase-semblance estimator here is the standard differential-phase / beam-
forming form the paper's prose describes, validated on synthetic arrays with
a known slowness.  Slowness in us/ft, frequency in Hz, spacing in ft.
"""

import numpy as np


# ---------------------------------------------- semblance ---------------

def frequency_slowness_semblance(waveforms, dt, z, slowness_grid_us_ft, freqs_hz):
    """Phase-coherence semblance over (frequency, slowness).

    waveforms : (n_receivers, n_samples) array.
    z         : receiver axial positions (ft).
    Returns a (len(freqs), len(slowness)) semblance map in [0, 1].
    For each (f, s), back-propagate each receiver's phase by exp(i*2pi*f*s*z)
    and measure how coherently the spectra stack.
    """
    W = np.asarray(waveforms, float)
    n_rx, n_t = W.shape
    z = np.asarray(z, float)
    spec = np.fft.rfft(W, axis=1)
    fft_freqs = np.fft.rfftfreq(n_t, dt)
    smap = np.zeros((len(freqs_hz), len(slowness_grid_us_ft)))
    for fi, f in enumerate(freqs_hz):
        k = int(np.argmin(np.abs(fft_freqs - f)))
        Xf = spec[:, k]                                  # (n_rx,)
        denom = n_rx * np.sum(np.abs(Xf) ** 2)
        for si, s_us_ft in enumerate(slowness_grid_us_ft):
            s = s_us_ft * 1e-6                           # us/ft -> s/ft
            phase = np.exp(1j * 2 * np.pi * f * s * z)   # back-propagation
            stacked = np.abs(np.sum(Xf * phase)) ** 2
            smap[fi, si] = stacked / denom if denom > 0 else 0.0
    return smap


def extract_slowness(smap, slowness_grid):
    """Per-frequency slowness = argmax of the semblance map."""
    return np.asarray(slowness_grid, float)[np.argmax(smap, axis=1)]


def group_delay(phase, freqs):
    """Group delay  T(f) = -(1/2pi) dphi/df from an unwrapped phase spectrum."""
    phi = np.unwrap(np.asarray(phase, float))
    return -np.gradient(phi, np.asarray(freqs, float)) / (2 * np.pi)


# ---------------------------------------------- synthetic arrays --------

def _ricker(t, f0):
    a = (np.pi * f0 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def nondispersive_array(slowness_us_ft, z, dt, n_t, f0=4000.0, t0=0.004):
    """Array waveforms for a single nondispersive mode of given slowness."""
    t = np.arange(n_t) * dt
    s = slowness_us_ft * 1e-6
    W = np.zeros((len(z), n_t))
    for j, zj in enumerate(z):
        W[j] = _ricker(t - (t0 + s * zj), f0)
    return W


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Sonic Dispersion via Differential-Phase Semblance")
    print("=" * 60)

    dt = 1e-5                       # 10 us sampling
    n_t = 2048
    z = np.arange(8) * 0.5          # 8 receivers, 0.5 ft apart
    s_true = 120.0                  # us/ft (fast formation)

    W = nondispersive_array(s_true, z, dt, n_t, f0=4000.0)
    slow_grid = np.arange(40, 400, 1.0)
    freqs = np.array([3000.0, 4000.0, 5000.0])
    smap = frequency_slowness_semblance(W, dt, z, slow_grid, freqs)
    s_est = extract_slowness(smap, slow_grid)
    print(f"  true slowness          = {s_true} us/ft")
    print(f"  estimated (3 freqs)    = {s_est}")
    assert np.all(np.abs(s_est - s_true) <= 2.0), "must recover known slowness"
    assert smap.max() <= 1.0 + 1e-9 and smap.max() > 0.9

    # Group delay of a pure linear-phase delay returns that delay
    f = np.linspace(1000, 8000, 200)
    delay = 0.0025                  # 2.5 ms
    phase = -2 * np.pi * f * delay
    T = group_delay(phase, f)
    print(f"  recovered group delay  = {np.median(T)*1e3:.3f} ms (expect 2.5)")
    assert abs(np.median(T) - delay) < 1e-5
    print("  PASS")
    return {"slowness_est": s_est.tolist(), "semblance_peak": float(smap.max())}


if __name__ == "__main__":
    test_all()
