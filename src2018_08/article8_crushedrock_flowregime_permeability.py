"""
Article 8: Incorporating Flow Regimes Into Crushed-Rock Analysis to Better
           Understand Matrix Permeability and Pore Structure in Shales
Royer, Hobbs, Bonar (2018)
DOI: 10.30632/PJV59V4-2018a7

Crushed-rock (GRI) pressure-decay permeability on shales is inflated by gas
slippage and Knudsen diffusion in nanopores.  This module reconstructs the
flow regime via the Knudsen number, applies the Klinkenberg slip correction,
and implements the paper's "lambda plot": because the apparent permeability is
log-linear in the gas mean free path, fitting ln(ka) vs lambda across several
gases and pressures and extrapolating to a 1-nm mean free path gives a
slip-corrected permeability (k1lambda) and an effective pore diameter.

Implements:

  - Darcy flow rate  Q = k*A*dP/(mu*L)
  - Gas mean free path  lambda = kB*T/(sqrt(2)*pi*delta^2*P)
  - Knudsen number and flow-regime label  Kn = lambda/d
  - Klinkenberg apparent permeability  ka = k_inf*(1 + b/P)
  - Lambda-plot fit -> 1-nm-equivalent permeability k1lambda and pore diameter

Note: this issue's PDF has a text layer; the variables and constants are defined
in prose but the printed display fractions were dropped in extraction, so the
numbered relations (Eqs. 1-12) are faithful standard-form reconstructions.  The
He routine-GRI check (delta~0.25 nm, 56 psi, ambient -> lambda~38 nm) is
reproduced.  SI units (lambda in m, P in Pa, k in m^2).
"""

import numpy as np

KB = 1.380649e-23            # Boltzmann constant (J/K)


# ---------------------------------------------- flow regime --------------

def darcy_rate(k, area, dp, mu, length):
    """Darcy flow rate  Q = k*A*dP/(mu*L)  (Eq. 1)."""
    return k * area * dp / (mu * length)


def mean_free_path(temperature, pressure, collision_diameter):
    """Gas mean free path  lambda = kB*T/(sqrt(2)*pi*delta^2*P)  (Eq. 3).

    delta = gas collision diameter (He ~ 0.25 nm, N2 ~ 0.36 nm).
    """
    return KB * temperature / (np.sqrt(2.0) * np.pi * collision_diameter ** 2 * pressure)


def knudsen_number(lmbda, pore_diameter):
    """Knudsen number  Kn = lambda/d  (Eq. 2)."""
    return np.asarray(lmbda, float) / pore_diameter


def flow_regime(kn):
    """Flow regime from the Knudsen number (Roy/Civan thresholds)."""
    if kn < 1e-3:
        return "Darcy (continuum)"
    if kn < 0.1:
        return "slip"
    if kn < 10.0:
        return "transition"
    return "free-molecular"


def klinkenberg(k_inf, b, pressure):
    """Klinkenberg apparent permeability  ka = k_inf*(1 + b/P)  (Eq. 4)."""
    return k_inf * (1.0 + b / np.asarray(pressure, float))


# ---------------------------------------------- lambda plot --------------

def lambda_plot_fit(lambdas, ka):
    """Fit the lambda plot  ln(ka) = B + M*lambda  (Eqs. 10-11).

    Returns (intercept B, slope M) of the line through several gases/pressures;
    apparent permeability is log-linear in the gas mean free path.
    """
    m, b = np.polyfit(np.asarray(lambdas, float), np.log(np.asarray(ka, float)), 1)
    return b, m


def k_one_lambda(intercept_b, slope_m, lambda_ref=1e-9):
    """1-nm-equivalent permeability  k1lambda = exp(B + M*lambda_ref)  (Eq. 11).

    Extrapolating the lambda-plot line to a 1-nm mean free path gives a slip-
    corrected, intrinsic-proxy matrix permeability.
    """
    return np.exp(intercept_b + slope_m * lambda_ref)


def effective_pore_diameter(slope_m, const=1.0):
    """Effective pore diameter from the lambda-plot slope (Eq. 12, proxy)

        d ~ const/|M|  (tube assumption): a steeper slope -> tighter pores.
    """
    return const / abs(slope_m)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Crushed-Rock Flow-Regime Permeability")
    print("=" * 60)

    psi = 6894.76                                      # Pa per psi

    # Routine-GRI He check: ~38 nm mean free path at 56 psi, ambient
    lam_he = mean_free_path(298.0, 56.0 * psi, 0.25e-9)
    print(f"  He mean free path 56psi = {lam_he * 1e9:.1f} nm")
    assert np.isclose(lam_he * 1e9, 38.0, atol=2.0)

    # Knudsen regime: a 10-nm pore at this lambda is in the transition regime
    kn = knudsen_number(lam_he, 10e-9)
    print(f"  Knudsen number / regime = {kn:.2f} / {flow_regime(kn)}")
    assert flow_regime(kn) in ("transition", "free-molecular")

    # Klinkenberg raises apparent k at low pressure
    assert klinkenberg(5e-21, 5e5, 56 * psi) > klinkenberg(5e-21, 5e5, 170 * psi)

    # Lambda plot: plant a line ln(ka)=B+M*lambda over He+N2 at three pressures,
    # recover B, M and the 1-nm-equivalent permeability.  Apparent permeability
    # rises with mean free path (more slip), so the slope M is positive.
    nD = 9.869e-22                                    # m^2 per nanodarcy
    B_true, M_true = np.log(5.0 * nD), 1.0e8          # B at lambda=0, slope (1/m)
    pres = np.array([56.0, 130.0, 170.0]) * psi
    lam = np.concatenate([mean_free_path(298.0, pres, 0.25e-9),     # He
                          mean_free_path(298.0, pres, 0.36e-9)])    # N2
    ka = np.exp(B_true + M_true * lam)
    B, M = lambda_plot_fit(lam, ka)
    k1 = k_one_lambda(B, M)
    print(f"  k(1-lambda)            = {k1 / nD:.1f} nD  (uncorrected ~{ka.max() / nD:.0f} nD)")
    assert np.isclose(B, B_true, rtol=1e-6) and np.isclose(M, M_true, rtol=1e-6)
    # Slip-corrected k is far below the largest-lambda (most slip-inflated) ka
    assert k1 < ka.max()
    assert effective_pore_diameter(M_true) > 0
    print("  PASS")
    return {"lambda_He_nm": float(lam_he * 1e9), "k1lambda_m2": float(k1)}


if __name__ == "__main__":
    test_all()
