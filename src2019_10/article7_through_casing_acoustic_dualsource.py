"""
Article 7: A Through-Casing Acoustic Logging Tool Using Dual-Source Transmitters
Tang, Su, Zhuang (2019)
DOI: 10.30632/PJV60N5-2019a7

In cased holes the strong casing wave masks the weak formation arrival.  This
tool fires two transmitters (near and far) with a programmed delay so that, by
superposition at the receiver array, the casing wave is cancelled while the
formation arrival survives - then slowness-time-coherence (STC) processing
recovers the formation compressional slowness through casing.

Implements:

  - Dual-source superposition with a casing-cancellation delay  tau = L/v_casing
  - Casing-wave suppression by destructive interference (notch in omega-k)
  - Slowness-time coherence (semblance) over the receiver array
  - Formation slowness picking

Note: this issue's PDF text layer kept the equation numbers and variable
definitions but dropped the typeset glyphs, so these are the standard
superposition / semblance forms anchored to those definitions.  Paper anchor:
recovered formation P-wave velocity ~ 3700 m/s.
"""

import numpy as np


# ---------------------------------------------- dual source -------------

def cancellation_delay(transmitter_spacing, v_casing):
    """Delay that aligns the two casing arrivals  tau = L/v_casing."""
    return transmitter_spacing / v_casing


def dual_source_combine(near_traces, far_traces, dt, tau):
    """Subtract the delay-shifted far record from the near record.

    With tau = L/v_casing the casing arrivals coincide and cancel in
    (near - far_shifted), leaving the formation arrival.
    """
    # the far record's casing arrival is later by tau, so shift it earlier
    shift = int(round(tau / dt))
    far_shifted = np.roll(np.asarray(far_traces, float), -shift, axis=1)
    return np.asarray(near_traces, float) - far_shifted


# ---------------------------------------------- semblance ---------------

def semblance(traces, dt, dz_ft, slowness_us_ft, t0_us, window_us=60.0):
    """Slowness-time coherence over a receiver array at one (slowness, t0)."""
    traces = np.asarray(traces, float)
    nrec, nsamp = traces.shape
    nwin = int(window_us / dt)
    t_idx = (t0_us / dt) + np.arange(nwin)
    aligned = np.zeros((nrec, nwin))
    for r in range(nrec):
        idx = t_idx + slowness_us_ft * dz_ft[r] / dt
        aligned[r] = np.interp(idx, np.arange(nsamp), traces[r], left=0.0, right=0.0)
    stacked = aligned.sum(0)
    den = nrec * np.sum(aligned ** 2)
    return float(np.sum(stacked ** 2) / den) if den > 0 else 0.0


def pick_slowness(traces, dt, dz_ft, slowness_grid, t0_grid, window_us=60.0):
    """Pick (slowness, t0, coherence) maximizing semblance over a grid."""
    best = (slowness_grid[0], t0_grid[0], -1.0)
    for s in slowness_grid:
        for t0 in t0_grid:
            c = semblance(traces, dt, dz_ft, s, t0, window_us)
            if c > best[2]:
                best = (s, t0, c)
    return best


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Through-Casing Acoustic Dual-Source")
    print("=" * 60)

    rng = np.random.default_rng(7)
    dt = 3.0                                   # us (tau/dt is an integer here)
    nsamp = 700
    dz = np.arange(8) * 0.5                     # 0.5-ft receivers
    t = np.arange(nsamp) * dt

    def wavelet(tc, amp):
        x = t - tc
        return amp * (1 - 0.5 * (x / 20.0) ** 2) * np.exp(-(x / 40.0) ** 2)

    # Casing wave: fast (57 us/ft ~ steel) and strong; formation: slow (~82
    # us/ft ~ 3700 m/s) and weak.  Far transmitter is L ft beyond the near one.
    s_cas, s_fm = 57.0, 82.0
    L_ft = 2.0
    v_casing_ft_us = 1.0 / s_cas               # ft/us
    t0n_cas, t0n_fm = 150.0, 300.0
    near = np.array([wavelet(t0n_cas + s_cas * dz[r], 1.0)
                     + wavelet(t0n_fm + s_fm * dz[r], 0.25) for r in range(8)])
    # far transmitter: extra L ft of path for BOTH waves
    far = np.array([wavelet(t0n_cas + s_cas * (dz[r] + L_ft), 1.0)
                    + wavelet(t0n_fm + s_fm * (dz[r] + L_ft), 0.25) for r in range(8)])

    # Without cancellation, the casing wave dominates the semblance pick
    s_grid = np.arange(45.0, 110.0, 1.0)
    t0_grid = np.arange(120.0, 460.0, 4.0)
    s_raw, _, _ = pick_slowness(near, dt, dz, s_grid, t0_grid)
    print(f"  raw pick (casing)      = {s_raw:.0f} us/ft")
    assert abs(s_raw - s_cas) < 4.0            # casing wins on the raw record

    # Dual-source cancellation removes the casing wave -> formation slowness wins
    tau = cancellation_delay(L_ft, v_casing_ft_us)     # = L*s_cas (us)
    combined = dual_source_combine(near, far, dt, tau)
    s_fm_pick, t0_pick, coh = pick_slowness(combined, dt, dz, s_grid, t0_grid)
    print(f"  dual-source pick (fm)  = {s_fm_pick:.0f} us/ft  (true {s_fm:.0f})")
    assert abs(s_fm_pick - s_fm) <= 2.0
    print("  PASS")
    return {"s_casing": s_raw, "s_formation": s_fm_pick, "coherence": coh}


if __name__ == "__main__":
    test_all()
