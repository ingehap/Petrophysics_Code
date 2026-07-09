"""
Article 5: NMR Evaluation of Light-Hydrocarbon Composition, Pore Size, and
           Tortuosity in Organic-Rich Chalks
Chen, Singer, Wang, Vinegar, Nguyen, Hirasaki (2019)
DOI: 10.30632/PJV60N6-2019a5

Core-log integration on organic-rich chalk: lab T1-T2 / D-T2 NMR on cores
saturated with water and light hydrocarbons (C1-C10) is converted to a downhole
apparent T2 by adding the diffusion-in-gradient term, then matched to the log.
Restricted diffusion fit with a Pade approximation yields the pore radius and
tortuosity.

Implements:

  - Hydrogen-index porosity rescaling  phi_l = HI*phi_l(HI=1)       (Eq. 1)
  - Apparent T2  1/T2app = 1/T2 + 1/T2D                             (Eq. 2)
  - Diffusion relaxation  1/T2D = (gamma*G*TE)^2 * D / 12           (Eq. 3)
  - Diffusion length  L_D = sqrt(D0*Delta)                          (Eq. 5)
  - Pade restricted-diffusion D/D0 (short-time) and tortuosity       (Eqs. 6-7)
  - Surface relaxation  1/T2 = 1/T2B + rho*(S/V) + 1/T2D

Note: this issue's PDF text layer kept the equation numbers and variable
definitions but dropped the typeset glyphs, so these are the standard NMR
relaxation / restricted-diffusion forms anchored to those definitions.
gamma/2pi = 42.58 MHz/T; example Pade fit gives r_p ~ 4.6 um, tortuosity ~ 85.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

GAMMA_H = 2.0 * np.pi * 42.58e6      # rad/s/T (gamma/2pi = 42.58 MHz/T)


# ---------------------------------------------- relaxation --------------

def hi_porosity(phi_hi1, hydrogen_index):
    """Hydrogen-index porosity rescaling  phi_l = HI*phi_l(HI=1)  (Eq. 1)."""
    return hydrogen_index * phi_hi1


def diffusion_relaxation(D, G, TE, gamma=GAMMA_H):
    """Diffusion relaxation rate  1/T2D = (gamma*G*TE)^2 * D / 12  (Eq. 3).

    D in m^2/s, G (gradient) in T/m, TE (echo spacing) in s -> 1/s.
    """
    return petrolib.nmr.diffusion_relaxation_rate(D, G=G, TE=TE, gamma=gamma)


def t2_apparent(T2, D, G, TE, gamma=GAMMA_H):
    """Apparent (downhole) T2  1/T2app = 1/T2 + 1/T2D  (Eq. 2).  T2 in s."""
    return petrolib.nmr.t2_apparent(t2_bulk=T2, D=D, G=G, TE=TE, gamma=gamma)


def surface_relaxation(T2_bulk, rho, s_over_v, D=0.0, G=0.0, TE=0.0, gamma=GAMMA_H):
    """Total T2 from bulk + surface + diffusion  1/T2 = 1/T2B + rho*(S/V) + 1/T2D."""
    return petrolib.nmr.t2_apparent(
        t2_bulk=T2_bulk, rho=rho, s_over_v=s_over_v, D=D, G=G, TE=TE, gamma=gamma)


# ---------------------------------------------- restricted diffusion ----

def diffusion_length(D0, delta):
    """Diffusion length  L_D = sqrt(D0*Delta)  (Eq. 5)."""
    return petrolib.flow_transport.diffusion_length(D0, delta)


def pade_short_time(D0, s_over_v, delta):
    """Short-time Pade restricted-diffusion ratio  D/D0 = 1 - (4/(9*sqrt(pi)))*(S/V)*sqrt(D0*Delta)  (Eq. 6)."""
    return petrolib.nmr.mitra_short_time(D0, delta, s_over_v, normalized=True)


def pore_radius_from_sv(s_over_v):
    """Spherical-pore radius from S/V (S/V = 3/r_p)."""
    return 3.0 / s_over_v


def tortuosity(D_inf, D0):
    """Tortuosity  tau = D0/D(inf)  (Eq. 7)."""
    # library tortuosity(d0, d_inf) = d0/d_inf; this article's arg order is reversed.
    return petrolib.nmr.tortuosity(D0, D_inf)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: NMR Light-HC Composition / Pore Size / Tortuosity")
    print("=" * 60)

    # HI rescaling: methane HI < 1 suppresses apparent porosity
    assert abs(hi_porosity(10.0, 0.6) - 6.0) < 1e-9

    # Diffusion shortens the apparent T2 (T2app < T2), more so at higher gradient
    T2 = 0.5                                   # 500 ms
    D = 2.5e-9                                 # m^2/s
    TE = 0.4e-3                                # 0.4 ms
    t2a_low = t2_apparent(T2, D, 17.0e-4 * 100, TE)   # 17 G/cm -> T/m
    t2a_high = t2_apparent(T2, D, 38.7e-4 * 100, TE)  # 38.7 G/cm
    print(f"  T2app low/high gradient = {t2a_low*1e3:.1f} / {t2a_high*1e3:.1f} ms")
    assert t2a_low < T2 and t2a_high < t2a_low

    # Diffusion length grows with observation time
    assert diffusion_length(D, 30e-3) > diffusion_length(D, 10e-3)

    # Pade: restricted diffusion reduces D below D0; pore radius from S/V
    r_p = 4.6e-6
    s_v = 3.0 / r_p
    ratio = pade_short_time(D, s_v, 1e-4)
    print(f"  D/D0 (short-time)      = {ratio:.3f}")
    assert 0.0 < ratio < 1.0
    assert abs(pore_radius_from_sv(s_v) - r_p) < 1e-12

    # Tortuosity exceeds 1 (restricted long-time diffusion)
    tau = tortuosity(D / 85.0, D)
    print(f"  tortuosity             = {tau:.1f}")
    assert abs(tau - 85.0) < 1e-9 and tau > 1.0

    # Surface relaxation: smaller pores (higher S/V) relax faster
    assert surface_relaxation(3.0, 5e-6, 3.0 / 1e-6) < \
        surface_relaxation(3.0, 5e-6, 3.0 / 50e-6)
    print("  PASS")
    return {"t2app_ms": float(t2a_low * 1e3), "DD0": float(ratio),
            "tortuosity": float(tau)}


if __name__ == "__main__":
    test_all()
