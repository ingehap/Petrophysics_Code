"""
Article 2: A New Apparatus for Coupled Low-Field NMR and Ultrasonic Measurements
           in Rocks at Reservoir Conditions
Connolly, Sarout, Dautriat, May, Johns (2020)
DOI: 10.30632/PJV61N2-2020a2

A novel NMR-compatible core holder measures low-field (2 MHz) NMR T2 relaxometry
and ultrasonic P-/S-wave velocities simultaneously on a core at reservoir
pressure and temperature.  NMR T2 is governed by surface relaxation
(Brownstein-Tarr fast-diffusion limit) and inverted to a pore-size-weighted
multiexponential distribution; the ultrasonic transit times give the elastic
moduli.

Implements:

  - Longitudinal T1 recovery and transverse T2 decay              (Eqs. 1-2)
  - Surface relaxation  1/T2 = 1/T2bulk + rho_s*(S/V)
    + D0*gamma^2*G^2*te^2/12                                      (Eq. 3)
  - Multiexponential T2 magnetization sum                         (Eq. 4)
  - Elastic velocities  Vp = sqrt((K+4mu/3)/rho), Vs = sqrt(mu/rho)(Eqs. 5-6)

Note: this issue's PDF text layer kept the equation numbers and definitions but
dropped the typeset glyphs, so these are the standard NMR-relaxation and
elastic-velocity forms anchored to those definitions.  Paper anchors: 0.049 T
(2 MHz 1H), 100 us min echo time, Tikhonov-regularized T2 inversion.
"""

import numpy as np

GAMMA_H = 2.675e8        # rad/s/T


# ---------------------------------------------- relaxation --------------

def t1_recovery(t, M0, T1):
    """Longitudinal recovery  M(t) = M0*(1 - exp(-t/T1))  (Eq. 1)."""
    return M0 * (1.0 - np.exp(-np.asarray(t, float) / T1))


def t2_decay(t, M0, T2):
    """Transverse decay  M(t) = M0*exp(-t/T2)  (Eq. 2)."""
    return M0 * np.exp(-np.asarray(t, float) / T2)


def surface_relaxation(rho_s, s_over_v, T2_bulk=3.0, D0=0.0, G=0.0,
                       te=100e-6, gamma=GAMMA_H):
    """Total T2 from bulk + surface + diffusion relaxation  (Eq. 3).

        1/T2 = 1/T2_bulk + rho_s*(S/V) + D0*gamma^2*G^2*te^2/12
    rho_s in m/s, S/V in 1/m, T2_bulk in s -> returns T2 in s.
    """
    inv = 1.0 / T2_bulk + rho_s * s_over_v + D0 * gamma ** 2 * G ** 2 * te ** 2 / 12.0
    return 1.0 / inv


def multiexponential(t, amplitudes, T2s):
    """Multiexponential magnetization  M(t) = sum_i A_i*exp(-t/T2_i)  (Eq. 4)."""
    t = np.asarray(t, float)
    A = np.asarray(amplitudes, float)
    T2 = np.asarray(T2s, float)
    return (A[:, None] * np.exp(-t[None, :] / T2[:, None])).sum(axis=0)


# ---------------------------------------------- elastic -----------------

def vp(K, mu, rho):
    """Compressional velocity  Vp = sqrt((K + 4*mu/3)/rho)  (Eq. 5)."""
    return np.sqrt((K + 4.0 * mu / 3.0) / rho)


def vs(mu, rho):
    """Shear velocity  Vs = sqrt(mu/rho)  (Eq. 6)."""
    return np.sqrt(mu / rho)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Coupled Low-Field NMR + Ultrasonic")
    print("=" * 60)

    # T1 recovery approaches M0; one time constant -> ~63% recovered
    assert abs(t1_recovery(1e9, 1.0, 0.5) - 1.0) < 1e-9
    assert abs(t1_recovery(0.5, 1.0, 0.5) - (1 - np.exp(-1))) < 1e-9

    # T2 decay: one time constant -> ~37% remaining
    assert abs(t2_decay(0.1, 1.0, 0.1) - np.exp(-1)) < 1e-9

    # Surface relaxation: larger S/V (smaller pores) -> shorter T2
    T2_big = surface_relaxation(5e-6, 1.0 / 50e-6)     # 50 um pore (S/V=3/r approx)
    T2_small = surface_relaxation(5e-6, 1.0 / 5e-6)    # 5 um pore
    print(f"  T2 big / small pore    = {T2_big*1e3:.1f} / {T2_small*1e3:.1f} ms")
    assert T2_big > T2_small
    # diffusion term adds relaxation in a gradient -> shorter T2
    assert surface_relaxation(5e-6, 1e4, D0=2.5e-9, G=0.1) < \
        surface_relaxation(5e-6, 1e4)

    # Multiexponential decay starts at the total amplitude and decays
    t = np.linspace(0, 1.0, 50)
    M = multiexponential(t, [0.6, 0.4], [0.05, 0.5])
    assert abs(M[0] - 1.0) < 1e-9 and M[-1] < M[0]

    # Elastic velocities: Vp > Vs, both physical for a sandstone
    K, mu, rho = 16e9, 6e9, 2300.0
    Vp, Vs = vp(K, mu, rho), vs(mu, rho)
    print(f"  Vp / Vs                = {Vp:.0f} / {Vs:.0f} m/s")
    assert Vp > Vs and 2000 < Vp < 5000
    print("  PASS")
    return {"T2_small_ms": float(T2_small * 1e3), "Vp": float(Vp), "Vs": float(Vs)}


if __name__ == "__main__":
    test_all()
