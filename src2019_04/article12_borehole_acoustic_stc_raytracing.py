"""
Article 12: Borehole Acoustic Imaging Using 3D STC and Ray Tracing to Determine
            Far-Field Reflector Dip and Azimuth
Bennett, Donald, Ghadiry, Nassar, Kumar, Biswas (2019)
DOI: 10.30632/PJV60N2-2019a10

Borehole acoustic-reflection imaging detects geological reflectors away from the
wellbore.  A 3D slowness-time-coherence (STC) analysis over the azimuthal
receiver array measures the moveout slowness and azimuth of the reflected
arrival, and ray tracing back-projects the reflection to recover the far-field
reflector's dip and azimuth.

Implements:

  - Slowness-time coherence (semblance) over a receiver array
  - Azimuthal reflection amplitude fit (reflector azimuth)
  - Ray-traced reflector distance from two-way reflection time
  - Reflector dip from the depth-vs-offset moveout

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard STC / ray-tracing reflection-imaging relations
the paper applies.
"""

import numpy as np


# ---------------------------------------------- STC --------------------

def semblance(traces, dt, dz, slowness, t0, window):
    """Slowness-time coherence over a receiver array at one (slowness, t0)."""
    traces = np.asarray(traces, float)
    nrec, nsamp = traces.shape
    nwin = int(window / dt)
    t_idx = t0 / dt + np.arange(nwin)
    aligned = np.array([np.interp(t_idx + slowness * dz[r] / dt,
                                  np.arange(nsamp), traces[r], left=0.0, right=0.0)
                        for r in range(nrec)])
    stacked = aligned.sum(0)
    den = nrec * np.sum(aligned ** 2)
    return float(np.sum(stacked ** 2) / den) if den > 0 else 0.0


def pick_slowness(traces, dt, dz, s_grid, t0_grid, window):
    """Pick (slowness, t0, coherence) maximizing semblance over a grid."""
    best = (s_grid[0], t0_grid[0], -1.0)
    for s in s_grid:
        for t0 in t0_grid:
            c = semblance(traces, dt, dz, s, t0, window)
            if c > best[2]:
                best = (s, t0, c)
    return best


# ---------------------------------------------- azimuth / geometry ------

def fit_reflector_azimuth(azimuths_deg, amplitudes):
    """Reflector azimuth from the azimuthal amplitude lobe  A(az)=A0*cos(az-phi)+c."""
    az = np.radians(np.asarray(azimuths_deg, float))
    A = np.asarray(amplitudes, float)
    M = np.vstack([np.cos(az), np.sin(az), np.ones_like(az)]).T
    c1, c2, _ = np.linalg.lstsq(M, A, rcond=None)[0]
    return float(np.degrees(np.arctan2(c2, c1)) % 360.0)


def reflector_distance(two_way_time, velocity):
    """Reflector distance from the borehole  d = v*t/2."""
    return velocity * np.asarray(two_way_time, float) / 2.0


def reflector_dip(depth_near, depth_far, offset):
    """Reflector dip from the change in reflection depth over a lateral offset."""
    return float(np.degrees(np.arctan((depth_far - depth_near) / offset)))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 12: Borehole Acoustic 3D STC & Ray Tracing")
    print("=" * 60)

    # Synthetic receiver array with a planted reflected-arrival slowness
    rng = np.random.default_rng(10)
    dt, nsamp = 4.0, 400
    dz = np.arange(6) * 0.5
    s_true, t0_true = 95.0, 200.0
    t = np.arange(nsamp) * dt
    wav = lambda tc: (1 - 0.5 * ((t - tc) / 16.0) ** 2) * np.exp(-((t - tc) / 30.0) ** 2)
    traces = np.array([wav(t0_true + s_true * dz[r]) for r in range(6)])
    traces += 0.02 * rng.standard_normal(traces.shape)
    s_pick, t0_pick, coh = pick_slowness(traces, dt, dz,
                                         np.arange(70.0, 121.0, 1.0),
                                         np.arange(170.0, 231.0, 4.0), window=80.0)
    print(f"  STC slowness / coherence = {s_pick:.0f} us/ft / {coh:.3f}")
    assert abs(s_pick - s_true) <= 1.0 and coh > 0.9

    # Reflector azimuth from the azimuthal amplitude lobe
    az = np.arange(0, 360, 30.0)
    amp = 1.0 + 0.6 * np.cos(np.radians(az - 110.0))
    phi = fit_reflector_azimuth(az, amp)
    print(f"  reflector azimuth      = {phi:.0f} deg  (true 110)")
    assert abs(((phi - 110.0 + 180) % 360) - 180) < 1.0

    # Ray tracing: distance from two-way time, dip from depth moveout
    d = reflector_distance(0.02, 4000.0)           # 20 ms two-way, 4000 m/s
    print(f"  reflector distance     = {d:.0f} m")
    assert abs(d - 40.0) < 1e-6
    dip = reflector_dip(1000.0, 1030.0, 60.0)
    print(f"  reflector dip          = {dip:.1f} deg")
    assert dip > 0 and abs(dip - np.degrees(np.arctan(30.0 / 60.0))) < 1e-6
    print("  PASS")
    return {"slowness": s_pick, "azimuth": phi, "distance": float(d), "dip": dip}


if __name__ == "__main__":
    test_all()
