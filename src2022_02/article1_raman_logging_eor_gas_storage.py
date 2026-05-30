"""
Article 1: New Logging Tool for Enhanced Oil Recovery and Gas Storage
           Monitoring Applications
Andrews, Speck (2022)
DOI: 10.30632/PJV63N1-2022a1

A downhole production-logging tool performs in-situ Raman spectroscopy to
measure gas composition zone-by-zone for EOR, CO2 sequestration and
hydrogen-storage monitoring.  Raman intensities of the characteristic
bands of H2, N2, CO2, CH4, C2-C3, H2O and H2S are inverted (via a linear
calibration of Raman intensity vs molar density, plus pressure/temperature
constraints) into mole fractions, which are then allocated to per-zone,
per-component production flow rates.

Implements:

  - Linear Raman forward model  X = G * M @ rho                  (Eq. 3)
  - Ideal-gas number density     rho_m = f_m * P / (k_B * T)      (Eq. 4)
  - Lorentz-Lorenz molar-refractivity excitation-volume term     (inline)
  - Sum(f) = 1 constrained composition + gain inversion          (Eqs. 5-6)
  - Beer-Lambert cross-absorption response                       (Eqs. 7-8)
  - Ideal-gas per-zone / per-component flow allocation           (Eqs. 1-2)

Note: the journal's typeset equations are image-rendered and were not part
of the machine-readable text; the forms below are faithful standard
reconstructions of the methodology described in the paper body.
"""

import numpy as np

K_B = 1.380649e-23          # Boltzmann constant, J/K


# ---------------------------------------------- Eq. 4: ideal-gas density -

def ideal_gas_number_density(f, P_pa, T_k):
    """rho_m = f_m * P / (k_B * T)  (Eq. 4).  Number density in 1/m^3."""
    f = np.asarray(f, dtype=float)
    return f * P_pa / (K_B * T_k)


# ---------------------------------------------- molar refractivity ------

def molar_refractivity(n_index, rho_number):
    """Lorentz-Lorenz r_m = (1/rho) * (n^2-1)/(n^2+2).  rho is number density."""
    return (1.0 / rho_number) * (n_index ** 2 - 1.0) / (n_index ** 2 + 2.0)


def refractivity_correction(r_m, rho_number):
    """Excitation-volume correction factor  1 + 3 * r_m * rho  (inline term)."""
    return 1.0 + 3.0 * r_m * rho_number


# ---------------------------------------------- Eq. 3: forward model ----

def raman_forward(M, rho, gain=1.0):
    """Linear Raman channel response  X = G * M @ rho  (Eq. 3).

    M : (o x l) response matrix, o channels by l gases.
    rho : (l,) absolute number densities.
    """
    M = np.asarray(M, dtype=float)
    return gain * (M @ np.asarray(rho, dtype=float))


# ---------------------------------------------- Eqs. 7-8: Beer-Lambert --

def beer_lambert_response(X0, alpha, rho, path_len):
    """Transmission response  X = X0 * exp(-(alpha @ rho) * L)  (Eq. 8).

    alpha : (o x l) absorption-coefficient matrix; the exponential
    cross-absorption term also multiplies the linear Raman model (Eq. 7).
    """
    alpha = np.asarray(alpha, dtype=float)
    absorbance = (alpha @ np.asarray(rho, dtype=float)) * path_len
    return np.asarray(X0, dtype=float) * np.exp(-absorbance)


# ---------------------------------------------- Eqs. 5-6: inversion -----

def invert_composition(X, M, P_pa, T_k):
    """Recover mole fractions f and the system gain G from channel signals.

    Uses the linear model X = G * c * (M @ f) with c = P/(k_B T) and the
    closure Sum(f) = 1 (Eq. 5).  The over-determined system fixes one extra
    parameter - here the overall calibration gain G (Eq. 6).

    Returns (f, G).
    """
    X = np.asarray(X, dtype=float)
    M = np.asarray(M, dtype=float)
    c = P_pa / (K_B * T_k)
    # Solve M @ w = X  (least squares) where w = G * c * f
    w, *_ = np.linalg.lstsq(M, X, rcond=None)
    gain_c = float(np.sum(w))            # since Sum(f) = 1  =>  Sum(w) = G * c
    f = w / gain_c
    G = gain_c / c
    return f, G


# ---------------------------------------------- Eqs. 1-2: flow alloc. ---

def component_flow_rate(f_zone, Q_zone, P_zone, T_zone, z_zone,
                        P_ref=101325.0, T_ref=288.15, z_ref=1.0):
    """Per-component flow rate above a zone  (Eq. 1).

    q_x = f_x * Q * (P/P_ref) * (T_ref/T) * (z_ref/z)

    f_zone : (l,) mole fractions; Q_zone : cumulative volumetric flow.
    Returns (l,) standard-condition component flow rates.
    """
    corr = (P_zone / P_ref) * (T_ref / T_zone) * (z_ref / z_zone)
    return np.asarray(f_zone, dtype=float) * Q_zone * corr


def zonal_contribution(cumulative_above, cumulative_below):
    """Per-zone contribution = difference of cumulative component flows (Eq. 2)."""
    return np.asarray(cumulative_above, dtype=float) - np.asarray(cumulative_below, dtype=float)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: In-Situ Raman Composition Logging Tool")
    print("=" * 60)

    gases = ["CH4", "C2H6", "C3H8", "CO2", "N2"]
    f_true = np.array([0.80, 0.05, 0.03, 0.07, 0.05])
    assert abs(f_true.sum() - 1.0) < 1e-9

    P = 5000 * 6894.76          # 5,000 psi -> Pa
    T = 273.15 + 90.0           # 90 C downhole

    # Per-gas Raman response matrix (diagonal-dominant with small cross terms)
    M = np.array([
        [1.00, 0.02, 0.01, 0.00, 0.00],
        [0.03, 0.90, 0.05, 0.00, 0.00],
        [0.01, 0.04, 0.85, 0.00, 0.00],
        [0.00, 0.00, 0.00, 1.10, 0.02],
        [0.00, 0.00, 0.00, 0.01, 0.95],
    ])
    G_true = 0.73               # optical-path calibration gain

    rho = ideal_gas_number_density(f_true, P, T)
    X = raman_forward(M, rho, gain=G_true)
    print(f"  total number density   = {rho.sum():.3e} 1/m^3")
    print(f"  channel signals X      = {np.array2string(X, precision=2)}")

    # Inversion recovers mole fractions and the gain (Eqs. 5-6)
    f_hat, G_hat = invert_composition(X, M, P, T)
    print(f"  recovered gain  G      = {G_hat:.4f}  (true {G_true})")
    for g, ft, fh in zip(gases, f_true, f_hat):
        print(f"    f[{g:4s}] true={ft:.3f}  est={fh:.3f}")
    assert np.allclose(f_hat, f_true, atol=1e-6), "composition not recovered"
    assert abs(G_hat - G_true) < 1e-6, "gain not recovered"
    assert abs(f_hat.sum() - 1.0) < 1e-9

    # Refractivity excitation-volume correction is a small positive factor
    r_m = molar_refractivity(1.0005, rho[0])
    corr = refractivity_correction(r_m, rho[0])
    print(f"  refractivity factor    = {corr:.5f}")
    assert corr > 1.0

    # Beer-Lambert cross-absorption attenuates the signal (Eqs. 7-8)
    alpha = 1e-26 * np.ones((5, 5))
    Xatt = beer_lambert_response(X, alpha, rho, path_len=0.01)
    print(f"  Beer-Lambert X (att.)  = {np.array2string(Xatt, precision=2)}")
    assert np.all(Xatt <= X + 1e-12)

    # Per-zone flow allocation (Eqs. 1-2): two zones, cumulative flows
    Q_above_z1 = 1000.0         # rm3/d above zone 1 (deepest)
    Q_above_z2 = 1700.0         # rm3/d above zone 2 (cumulative)
    q1 = component_flow_rate(f_true, Q_above_z1, P, T, z_zone=0.95)
    q2 = component_flow_rate(f_true, Q_above_z2, P, T, z_zone=0.95)
    q_zone2 = zonal_contribution(q2, q1)
    print(f"  zone-2 CH4 contribution = {q_zone2[0]:.1f} (std)")
    assert np.all(q_zone2 >= -1e-9), "zonal contribution must be non-negative"
    print("  PASS")
    return {"f_hat": f_hat, "G_hat": G_hat, "q_zone2": q_zone2}


if __name__ == "__main__":
    test_all()
