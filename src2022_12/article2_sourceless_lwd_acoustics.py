"""
Article 2: Sourceless LWD Borehole Acoustics: Field Testing the Concept
Bolshakov, Walker, Marksamer, Samano, Reynolds (2022)
DOI: 10.30632/PJV63N6-2022a2

Implements the multipole-mode recombination and semblance-based
velocity extraction used to convert drill-bit-generated noise on a
four-azimuth LWD receiver array into formation Vp, Vs, and Stoneley
velocities.  Eqs. 1-3 give the monopole / quadrupole / dipole
combinations:

    monopole  m(t) = (a0 + a1 + a2 + a3) / 4                   (Eq. 1)
    quadrupole q(t) = (a0 - a1 + a2 - a3) / 4                  (Eq. 2)
    dipole     d_x(t) = ((a0 + a2) - (a1 + a3)) / 4            (Eq. 3)

where a_i(t) is the time series at the four azimuthal positions
(0, 90, 180, 270 deg).  Receiver-station semblance picks the slowness
at which the modes align across the six-ring array.
"""

import numpy as np


# ---------------------------------------------- mode recombination -------

def monopole(a0, a1, a2, a3):
    """Eq. 1 - sum-of-azimuths."""
    return (a0 + a1 + a2 + a3) / 4.0


def quadrupole(a0, a1, a2, a3):
    """Eq. 2 - alternating-sign sum."""
    return (a0 - a1 + a2 - a3) / 4.0


def dipole(a0, a1, a2, a3, axis="x"):
    """Eq. 3 - dipole along x (axis='x') or y (axis='y')."""
    if axis == "x":
        return ((a0 + a2) - (a1 + a3)) / 4.0
    if axis == "y":
        return ((a1 + a3) - (a0 + a2)) / 4.0
    raise ValueError(axis)


# ---------------------------------------------- synthetic LWD record -----

def synth_lwd_record(rings_z_m=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                     azim_deg=(0, 90, 180, 270),
                     vp_m_s=3500.0, vs_m_s=2000.0, vst_m_s=1450.0,
                     fs_Hz=41_667, n_samples=4096, noise=0.10, seed=0):
    """Build a six-ring four-azimuth array of "listening-mode" time series.

    Drill-bit excitation is modelled as a Ricker pulse at the bit (the
    deepest receiver is 36.5 m above the bit; here we simulate the
    relative arrival times across the six rings).  Pulse arrives first at
    the bottom ring (closest to bit) and propagates upward with the
    requested mode velocity.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs_Hz
    t = np.arange(n_samples) * dt
    pulse = lambda t0: np.exp(-((t - t0) / 0.0003) ** 2) \
                     * np.cos(2 * np.pi * 4_000 * (t - t0))

    # Stack of (n_rings, n_az) time series
    out = np.zeros((len(rings_z_m), len(azim_deg), n_samples))
    bit_offset = 36.5  # m below bottom ring
    for r, z_r in enumerate(rings_z_m):
        d_p = (bit_offset + z_r) / vp_m_s
        d_s = (bit_offset + z_r) / vs_m_s
        d_st = (bit_offset + z_r) / vst_m_s
        for az_idx, _ in enumerate(azim_deg):
            # Monopole-like P, isotropic across azimuths
            sig = 1.0 * pulse(d_p)
            # Quadrupole shear (azimuth-dependent sign)
            sign = 1.0 if az_idx in (0, 2) else -1.0
            sig += 0.6 * sign * pulse(d_s)
            # Stoneley - low-frequency monopole tail (4 kHz pulse with a
            # second envelope at lower freq)
            sig += 0.4 * np.exp(-((t - d_st) / 0.0006) ** 2) \
                       * np.cos(2 * np.pi * 1_000 * (t - d_st))
            sig += noise * rng.standard_normal(n_samples)
            out[r, az_idx] = sig
    return t, np.array(rings_z_m), out


# ---------------------------------------------- semblance ---------------

def semblance(traces, rings_z_m, dt, slowness_us_ft,
              win_us=600, t_start_s=0.0, t_stop_s=0.020):
    """Classical multi-receiver coherence over a slowness grid.

    `traces` shape (n_rings, n_samples).  `slowness_us_ft` array.
    Returns (slowness, t_centres, semblance_matrix).
    """
    n_r = traces.shape[0]
    win_n = max(8, int(win_us * 1e-6 / dt))
    t_centres = np.arange(int(t_start_s / dt) + win_n,
                          int(t_stop_s / dt) - win_n,
                          win_n // 2)
    out = np.zeros((len(slowness_us_ft), len(t_centres)))
    ring_offsets_ft = (rings_z_m - rings_z_m[0]) * 3.2808
    for si, s in enumerate(slowness_us_ft):
        shifts = (s * ring_offsets_ft * 1e-6 / dt).astype(int)
        for ci, tc in enumerate(t_centres):
            stack = np.zeros(win_n)
            sumsq = 0.0
            for r in range(n_r):
                idx = tc - win_n // 2 + shifts[r]
                if idx < 0 or idx + win_n > traces.shape[1]:
                    stack += np.zeros(win_n)
                    continue
                seg = traces[r, idx:idx + win_n]
                stack += seg
                sumsq += float((seg ** 2).sum())
            num = float((stack ** 2).sum())
            den = max(n_r * sumsq, 1e-12)
            out[si, ci] = num / den
    return slowness_us_ft, t_centres * dt, out


def peak_slowness(slow, sem_map):
    """Slowness of the global semblance peak."""
    i, _ = np.unravel_index(np.argmax(sem_map), sem_map.shape)
    return float(slow[i])


# ---------------------------------------------- tests ------------------

def test_all():
    print("=" * 60)
    print("Article 2: Sourceless LWD Acoustics from Drill-Bit Noise")
    print("=" * 60)

    t, rings, recs = synth_lwd_record(seed=0)
    n_rings, n_az, _ = recs.shape
    dt = float(t[1] - t[0])

    # Recombine into multipole modes per ring
    mono = np.zeros((n_rings, len(t)))
    quad = np.zeros((n_rings, len(t)))
    dip_x = np.zeros((n_rings, len(t)))
    for r in range(n_rings):
        a = recs[r]
        mono[r] = monopole(a[0], a[1], a[2], a[3])
        quad[r] = quadrupole(a[0], a[1], a[2], a[3])
        dip_x[r] = dipole(a[0], a[1], a[2], a[3], axis="x")

    # Quick sanity: monopole and quadrupole are orthogonal in expectation
    corr = float(np.corrcoef(mono.mean(0), quad.mean(0))[0, 1])
    print(f"  corr(monopole, quadrupole)   = {corr:+.3f}  (should be small)")
    assert abs(corr) < 0.30

    slow_grid = np.linspace(50.0, 250.0, 41)  # us/ft
    _, _, sem_mono = semblance(mono, rings, dt, slow_grid)
    _, _, sem_quad = semblance(quad, rings, dt, slow_grid)

    s_p = peak_slowness(slow_grid, sem_mono)
    s_s = peak_slowness(slow_grid, sem_quad)
    vp_hat = 1e6 / s_p * 0.3048
    vs_hat = 1e6 / s_s * 0.3048
    print(f"  Picked Vp = {vp_hat:6.0f} m/s  (true 3500)")
    print(f"  Picked Vs = {vs_hat:6.0f} m/s  (true 2000)")

    # +/- 15 % tolerance on the slowness pick
    assert 2950.0 < vp_hat < 4050.0, "Vp recovery out of band"
    assert 1700.0 < vs_hat < 2350.0, "Vs recovery out of band"
    print("  PASS")
    return {"vp_hat": float(vp_hat), "vs_hat": float(vs_hat),
            "corr_mono_quad": corr}


if __name__ == "__main__":
    test_all()
