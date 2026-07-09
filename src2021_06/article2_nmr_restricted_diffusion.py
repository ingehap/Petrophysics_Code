"""
Article 2: Pore Size, Tortuosity, and Permeability From NMR Restricted
           Diffusion in Organic-Rich Chalks
Wang, Singer, Liu, Chen, Hirasaki, Vinegar (2021)
DOI: 10.30632/PJV62N3-2021a1

NMR restricted-diffusion (PFG stimulated-echo D-T2) measurements give a time-
dependent apparent diffusion coefficient D(L_D)/D0 that is fit with a Pade
interpolation bridging the short-time (Mitra surface-to-volume) limit and the
long-time (tortuosity) limit, yielding pore-body diameter d and diffusive
tortuosity tau, which feed a modified Carman-Kozeny permeability.

Implements:

  - Timur-Coates permeability  k = A*phi^m*(FFV/BFV)^n            (Eq. 1)
  - SDR permeability  k = A*phi^m*T2LM^n                          (Eq. 2)
  - Modified Carman-Kozeny  k = (phi/32)*d^2/(BTR^2*tau)          (Eq. 3)
  - Electrical tortuosity tau_e = F*phi; Archie F = phi^-m        (Eqs. 4-5)
  - Diffusive tortuosity  tau = D0/D_inf                          (Eq. 7)
  - Pade restricted-diffusion model (cylindrical, S/V = 4/d)      (Eq. 11)
  - Diffusion length  L_D = sqrt(D0*t)                            (Eq. 9)

Equations transcribed from the rendered article.  D0 in um^2/ms, lengths in
um, tau dimensionless, permeability in consistent units (report mD).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# Bulk self-diffusion coefficients (um^2/ms) at the paper's conditions
D0_WATER = 2.3
D0_METHANE = 250.0
D0_DECANE = 1.6


# ---------------------------------------------- Eqs. 1-2: k models ------

def timur_coates(phi, ffv, bfv, A=0.1, m=4.0, n=2.0):
    """Timur-Coates permeability  k = A*phi^m*(FFV/BFV)^n  (Eq. 1)."""
    return petrolib.nmr.timur_coates(phi, ffv, bfv, C=A, m=m, n=n, form="prefactor")


def sdr_permeability(phi, t2lm, A=0.1, m=4.0, n=2.0):
    """SDR permeability  k = A*phi^m*T2LM^n  (Eq. 2)."""
    return petrolib.nmr.sdr(phi, t2lm, a=A, m=m, n=n)


# ---------------------------------------------- Eq. 3: Carman-Kozeny ----

def carman_kozeny(phi, d, BTR, tau):
    """Modified Carman-Kozeny  k = (phi/32) * d^2 / (BTR^2 * tau)  (Eq. 3).

    d = pore-body diameter, BTR = body-to-throat ratio, tau = diffusive
    tortuosity.  Returns permeability in the same length-unit^2 as d^2.
    """
    return (phi / 32.0) * d ** 2 / (BTR ** 2 * tau)


# ---------------------------------------------- Eqs. 4-7: tortuosity ----

def electrical_tortuosity(formation_factor, phi):
    """Electrical tortuosity  tau_e = F_R * phi  (Eq. 4)."""
    return formation_factor * phi


def archie_formation_factor(phi, m):
    """Archie's law  F_R = phi^(-m)  (Eq. 5)."""
    return petrolib.saturation_resistivity.formation_factor(phi, m=m)


def diffusive_tortuosity(D0, D_inf):
    """Diffusive tortuosity  tau = D0 / D_inf  (Eq. 7)."""
    return petrolib.nmr.tortuosity(D0, D_inf)


# ---------------------------------------------- Eqs. 9-11: diffusion ----

def diffusion_length(D0, t_delta):
    """Diffusion length  L_D = sqrt(D0 * t_delta)  (Eq. 9).  um."""
    return petrolib.flow_transport.diffusion_length(D0, t_delta)


def sv_cylinder(d):
    """Surface-to-volume ratio of a cylindrical pore  S/V = 4/d  (Eq. 10)."""
    return 4.0 / d


def pade_diffusion(L_D, d, tau, L_M):
    """Pade restricted-diffusion model, cylindrical geometry (Eq. 11).

    D(L_D)/D0 = 1 - (1-1/tau) * X / (X + (1-1/tau)),
    X = (16/(9*sqrt(pi)))*(L_D/d) + (1-1/tau)*(L_D^2/L_M^2).
    -> 1 at L_D=0 (free diffusion); -> 1/tau at large L_D (tortuosity limit).
    """
    L_D = np.asarray(L_D, float)
    g = 1.0 - 1.0 / tau
    X = (16.0 / (9.0 * np.sqrt(np.pi))) * (L_D / d) + g * (L_D ** 2 / L_M ** 2)
    return 1.0 - g * X / (X + g)


def fit_pore_size_tortuosity(L_D, ratio, L_M, d_grid=None, tau_grid=None):
    """Recover (d, tau) by grid search against measured D/D0 vs L_D (Eq. 11)."""
    if d_grid is None:
        d_grid = np.linspace(0.5, 20.0, 80)
    if tau_grid is None:
        tau_grid = np.linspace(1.05, 6.0, 80)
    best, best_err = (None, None), np.inf
    for d in d_grid:
        for tau in tau_grid:
            pred = pade_diffusion(L_D, d, tau, L_M)
            err = np.sum((pred - ratio) ** 2)
            if err < best_err:
                best_err, best = err, (float(d), float(tau))
    return best


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Pore Size / Tortuosity / Permeability From NMR")
    print("=" * 60)

    # Pade limits: free diffusion at L_D->0, tortuosity plateau at large L_D
    d, tau, L_M = 5.0, 2.0, 5.0
    assert abs(pade_diffusion(1e-6, d, tau, L_M) - 1.0) < 1e-6
    plateau = pade_diffusion(1e6, d, tau, L_M)
    print(f"  D/D0 at large L_D      = {plateau:.4f}  (expect 1/tau = 0.5)")
    assert abs(plateau - 1.0 / tau) < 1e-3

    # Monotonic decrease with diffusion length
    L_D = diffusion_length(D0_DECANE, np.array([1, 5, 14.7, 25, 50, 100, 200, 400.0]))
    curve = pade_diffusion(L_D, d, tau, L_M)
    assert np.all(np.diff(curve) < 0)
    print(f"  D/D0 curve             = {np.round(curve, 3)}")

    # Round-trip: recover planted (d, tau) from a clean synthetic curve
    d_hat, tau_hat = fit_pore_size_tortuosity(L_D, curve, L_M)
    print(f"  recovered d / tau      = {d_hat:.2f} / {tau_hat:.2f} (true 5.0 / 2.0)")
    assert abs(d_hat - 5.0) < 0.7 and abs(tau_hat - 2.0) < 0.2

    # Diffusive tortuosity definition
    assert abs(diffusive_tortuosity(D0_DECANE, D0_DECANE / 2.0) - 2.0) < 1e-9

    # Permeability models are positive; Carman-Kozeny uses pore size & tortuosity
    k_ck = carman_kozeny(phi=0.30, d=5e-4, BTR=3.0, tau=2.0)   # d in cm
    k_tc = timur_coates(0.30, ffv=0.6, bfv=0.4)
    k_sdr = sdr_permeability(0.30, t2lm=30.0)
    print(f"  k Carman-Kozeny        = {k_ck:.3e} (cm^2 units)")
    print(f"  k Timur-Coates / SDR   = {k_tc:.4f} / {k_sdr:.4f}")
    assert k_ck > 0 and k_tc > 0 and k_sdr > 0

    # Electrical tortoisity / Archie
    F = archie_formation_factor(0.30, m=2.0)
    assert abs(electrical_tortuosity(F, 0.30) - F * 0.30) < 1e-12
    print("  PASS")
    return {"d_hat": d_hat, "tau_hat": tau_hat, "plateau": plateau}


if __name__ == "__main__":
    test_all()
