"""
Article 7: Revealing Hidden Information - High-Resolution Logging-While-Drilling
           Slowness Measurements and Imaging Using Advanced Dual Ultrasonic
           Technology
Blyth, Sakiyama, Hori, Yamamoto, Nakajima, Fahim Ud Din, Haecker, Kittridge (2021)
DOI: 10.30632/PJV62N1-2021a6

A dual ultrasonic LWD tool measures both refracted-headwave slowness across a
short receiver array (high-resolution compressional slowness) and pulse-echo
amplitude for a borehole image.  The slowness is extracted with slowness-time
coherence (STC / semblance) processing; the image is built from the
normal-incidence reflection coefficient between borehole fluid and formation.

Implements:

  - Slowness-time coherence (semblance) over a receiver array
  - Slowness picking by maximizing semblance
  - Acoustic impedance  Z = rho * v  and reflection coefficient R
  - Pulse-echo image amplitude from impedance contrast

Note: this issue's source PDF has no usable text layer, so the semblance and
reflection-coefficient relations are faithful standard-form reconstructions of
the STC slowness processing and pulse-echo imaging the paper applies.
Slowness in us/ft; spacings in ft; time in us.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- STC semblance -----------

def semblance(traces, dt_us, dz_ft, slowness_us_ft, t0_us, window_us=40.0):
    """Slowness-time coherence over a receiver array at one (slowness, t0).

        rho = sum_t ( sum_r x_r(t + s*dz_r) )^2 / ( N sum_t sum_r x_r^2 )
    traces: (n_receivers, n_samples); dz_ft: receiver offsets (ft) from the
    first receiver; slowness in us/ft; t0_us window start.  Returns coherence
    in [0, 1].
    """
    traces = np.asarray(traces, float)
    nrec, nsamp = traces.shape
    nwin = int(window_us / dt_us)
    t_idx = (t0_us / dt_us) + np.arange(nwin)
    num = den = 0.0
    aligned = np.zeros((nrec, nwin))
    for r in range(nrec):
        shift = slowness_us_ft * dz_ft[r] / dt_us          # samples
        idx = t_idx + shift
        aligned[r] = np.interp(idx, np.arange(nsamp), traces[r],
                               left=0.0, right=0.0)
    stacked = aligned.sum(axis=0)
    num = np.sum(stacked ** 2)
    den = nrec * np.sum(aligned ** 2)
    return num / den if den > 0 else 0.0


def pick_slowness(traces, dt_us, dz_ft, slowness_grid, t0_grid, window_us=40.0):
    """Pick (slowness, t0) maximizing semblance over a search grid.

    Returns (best_slowness, best_t0, best_coherence).
    """
    best = (slowness_grid[0], t0_grid[0], -1.0)
    for s in slowness_grid:
        for t0 in t0_grid:
            c = semblance(traces, dt_us, dz_ft, s, t0, window_us)
            if c > best[2]:
                best = (s, t0, c)
    return best


# ---------------------------------------------- pulse-echo imaging ------

def impedance(rho, v):
    """Acoustic impedance  Z = rho * v  (rho kg/m^3, v m/s -> Rayl)."""
    return petrolib.acoustic_geomech.acoustic_impedance(rho, v)


def reflection_coefficient(Z1, Z2):
    """Normal-incidence reflection coefficient  R = (Z2 - Z1)/(Z2 + Z1)."""
    return petrolib.acoustic_geomech.reflection_coefficient(Z1, Z2)


def pulse_echo_image(rho_fluid, v_fluid, rho_fm, v_fm):
    """Pulse-echo image amplitude |R| from fluid->formation impedance contrast."""
    Zf = impedance(rho_fluid, v_fluid)
    Zr = impedance(rho_fm, v_fm)
    return np.abs(reflection_coefficient(Zf, Zr))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: LWD Dual Ultrasonic Slowness & Imaging")
    print("=" * 60)

    # Build a synthetic 6-receiver array with a planted slowness of 80 us/ft
    rng = np.random.default_rng(7)
    dt = 2.0                                  # us sampling
    nsamp = 400
    dz = np.arange(6) * 0.5                    # 0.5-ft receiver spacing
    s_true = 80.0                             # us/ft
    t0_true = 120.0                           # us first-arrival at receiver 0
    t = np.arange(nsamp) * dt
    # a Ricker-like wavelet arriving at t0 + s_true*dz on each receiver
    def wavelet(tc):
        x = (t - tc)
        return (1 - 0.5 * (x / 8.0) ** 2) * np.exp(-(x / 16.0) ** 2)
    traces = np.array([wavelet(t0_true + s_true * dz[r]) for r in range(6)])
    traces += 0.02 * rng.standard_normal(traces.shape)

    s_grid = np.arange(50.0, 121.0, 2.0)
    t0_grid = np.arange(90.0, 151.0, 2.0)
    s_pick, t0_pick, coh = pick_slowness(traces, dt, dz, s_grid, t0_grid)
    print(f"  picked slowness        = {s_pick:.0f} us/ft  (true {s_true:.0f})")
    print(f"  picked t0 / coherence  = {t0_pick:.0f} us / {coh:.3f}")
    assert abs(s_pick - s_true) <= 2.0       # within one grid step
    assert coh > 0.9                          # coherent headwave

    # Reflection-coefficient imaging: hard formation reflects more than soft
    Zfluid = impedance(1000.0, 1500.0)         # borehole fluid
    R_hard = pulse_echo_image(1000.0, 1500.0, 2600.0, 5500.0)   # tight carbonate
    R_soft = pulse_echo_image(1000.0, 1500.0, 2100.0, 2400.0)   # soft shale
    print(f"  |R| hard / soft        = {R_hard:.3f} / {R_soft:.3f}")
    assert R_hard > R_soft > 0
    assert abs(reflection_coefficient(Zfluid, Zfluid)) < 1e-12   # no contrast
    print("  PASS")
    return {"slowness": s_pick, "coherence": coh,
            "R_hard": R_hard, "R_soft": R_soft}


if __name__ == "__main__":
    test_all()
