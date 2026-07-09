"""
Article 4: Nonlinear Acoustics Applications for Near-Wellbore Formation
           Evaluation
Skelt, TenCate, Guyer, Johnson, Larmat, Le Bas, Nihei, Vu (2021)
DOI: 10.30632/PJV62N2-2021a4

A near-wellbore tool based on noncollinear wave mixing: two acoustic beams of
frequencies w1, w2 intersect and, only when selection rules on convergence
angle and frequency ratio are met, generate a scattered wave at the difference
frequency w3 = w1 - w2 whose amplitude is proportional to the rock's
nonlinearity (Landau third-order moduli A, B, C).

Implements:

  - Nonlinear stress-strain  sigma = M*(e + beta*e^2 + delta*e^3)  (Eq. 1)
  - Nonlinearity parameter beta = 3/4 + (A+B+C)/(2*rho*Vp^2)       (Eq. 2)
  - Convergence angle phi                                          (Eq. 3)
  - Scattering angle gamma                                         (Eq. 4)
  - Scattering coefficient W (exact, Eqs. 5-8; approx, Eq. 9)
  - Frequency-ratio validity range                                (Eq. 10)

Equations transcribed from the rendered article.  Velocities m/s, density
kg/m^3, moduli Pa (Landau A,B,C given in GPa), angles in degrees.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- moduli helpers ----------

def lame_from_velocities(rho, Vp, Vs):
    """Lame parameters from velocities:  mu = rho*Vs^2, lambda = rho*Vp^2 - 2*mu."""
    return petrolib.acoustic_geomech.lame_from_velocity(Vp, Vs, rho)


# ---------------------------------------------- Eqs. 1-2 ----------------

def nonlinear_stress(strain, M, beta, delta=0.0):
    """Nonlinear stress-strain  sigma = M*(e + beta*e^2 + delta*e^3)  (Eq. 1)."""
    e = np.asarray(strain, float)
    return M * (e + beta * e ** 2 + delta * e ** 3)


def beta_parameter(A, B, C, rho, Vp):
    """Quadratic nonlinearity  beta = 3/4 + (A+B+C)/(2*rho*Vp^2)  (Eq. 2)."""
    return 0.75 + (A + B + C) / (2.0 * rho * Vp ** 2)


# ---------------------------------------------- Eqs. 3-4: geometry ------

def convergence_angle(Vs_over_Vp, freq_ratio):
    """Convergence angle phi between two converging P-waves (Eq. 3).  degrees.

    cos(phi) = 1/k^2 - [(1-k^2)(1+r^2)] / [2*r*k^2],  k = Vs/Vp, r = w2/w1.
    """
    k2 = Vs_over_Vp ** 2
    r = freq_ratio
    cos_phi = 1.0 / k2 - ((1.0 - k2) * (1.0 + r ** 2)) / (2.0 * r * k2)
    return np.degrees(np.arccos(np.clip(cos_phi, -1.0, 1.0)))


def scattering_angle(phi_deg, freq_ratio):
    """Scattering angle gamma  tan(gamma) = -r*sin(phi)/(1 - r*cos(phi))  (Eq. 4)."""
    phi = np.radians(phi_deg)
    r = freq_ratio
    return np.degrees(np.arctan2(-r * np.sin(phi), 1.0 - r * np.cos(phi)))


# ---------------------------------------------- Eqs. 5-9: scattering ----

def _c_coeffs(lam, mu, A, B):
    """Landau C-coefficients (Eqs. 6-8)."""
    C1 = mu + A / 4.0
    C2 = lam + mu + A / 4.0 + B
    C3 = A / 4.0 + B
    return C1, C2, C3


def scattering_coefficient_exact(lam, mu, A, B, Vs_over_Vp, phi_deg, freq_ratio):
    """Exact scattering coefficient W (P+P -> S)  (Eq. 5)."""
    C1, C2, C3 = _c_coeffs(lam, mu, A, B)
    phi = np.radians(phi_deg); r = freq_ratio
    pref = -(r) / (4 * np.pi * (lam + 2 * mu)) * (1 + r) / (4 * Vs_over_Vp)
    return pref * np.sin(2 * phi) * (2 * C1 + C2 + C3)


def scattering_coefficient_approx(lam, mu, A, B, Vs_over_Vp, phi_deg, freq_ratio):
    """Approximate scattering coefficient (Murnaghan m = A+2B)  (Eq. 9)."""
    phi = np.radians(phi_deg); r = freq_ratio
    pref = -(r) / (4 * np.pi * (lam + 2 * mu)) * (1 + r) / (4 * Vs_over_Vp)
    return pref * np.sin(2 * phi) * (A + 2 * B)


# ---------------------------------------------- Eq. 10: validity --------

def valid_frequency_ratio(Vs_over_Vp, freq_ratio):
    """Selection rule  (1-Vs/Vp)/(1+Vs/Vp) <= w2/w1 < 1  (Eq. 10)."""
    lo = (1.0 - Vs_over_Vp) / (1.0 + Vs_over_Vp)
    return lo <= freq_ratio < 1.0


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Nonlinear Acoustics (noncollinear wave mixing)")
    print("=" * 60)

    # Table 1: two converging P-waves -> scattered P
    Vp, Vs, rho = 3420.0, 1200.0, 2100.0
    A, B, C = -451e9, -3216e9, 1580e9      # Landau moduli (Pa)
    r = 0.74
    k = Vs / Vp

    # Eq. 10 validity
    assert valid_frequency_ratio(k, r)
    print(f"  freq-ratio valid       = True")

    # Eq. 3 / Eq. 4: convergence and scattering angles ~ 47.5 deg
    phi = convergence_angle(k, r)
    gam = scattering_angle(phi, r)
    print(f"  convergence angle phi  = {phi:.1f} deg  (expect ~47.5)")
    print(f"  scattering angle gamma = {gam:.1f} deg  (expect ~-47.5)")
    assert 47.0 < phi < 48.5
    assert 47.0 < abs(gam) < 48.5

    # Eq. 2: rock beta is large and negative for these moduli
    beta = beta_parameter(A, B, C, rho, Vp)
    print(f"  nonlinearity beta      = {beta:.1f}")
    assert beta < -30.0

    # Eq. 1: nonlinear stress deviates from linear at finite strain
    M = rho * Vp ** 2
    s_lin = M * 1e-6
    s_nl = nonlinear_stress(1e-6, M, beta)
    assert s_nl != s_lin and abs(s_nl - s_lin) > 0

    # Eqs. 5 vs 9: exact and approximate scattering coefficients nearly equal
    lam, mu = lame_from_velocities(rho, Vp, Vs)
    W_ex = scattering_coefficient_exact(lam, mu, A, B, k, phi, r)
    W_ap = scattering_coefficient_approx(lam, mu, A, B, k, phi, r)
    print(f"  W exact / approx       = {W_ex:.3e} / {W_ap:.3e}")
    assert abs(W_ex - W_ap) / abs(W_ap) < 0.02     # match within 2%

    # Sweep: a different Vp/Vs (1.601) peaks the scattering near phi=30 deg
    k2 = 1.0 / 1.601
    phi2 = convergence_angle(k2, 0.66)
    print(f"  phi at Vp/Vs=1.601, r=0.66 = {phi2:.1f} deg (expect ~30)")
    assert 26.0 < phi2 < 34.0
    print("  PASS")
    return {"phi": phi, "gamma": gam, "beta": beta, "W_exact": W_ex}


if __name__ == "__main__":
    test_all()
