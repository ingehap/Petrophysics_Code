#!/usr/bin/env python3
"""
Article 2: Damage Constitutive Model of Wellbore Rock Based on M-integral
Authors: Weihang Liu, Zhan Qu, Han Jiang, and Jianjun Wang
Ref: Petrophysics, Vol. 66, No. 5 (October 2025), pp. 728-740.
     DOI: 10.30632/PJV66N5-2025a2

Implements:
  - M-integral computation for 2-D strain/stress fields (Eq. 1)
  - Local mechanical failure driving factor Psi (Eq. 7)
  - Initial damage D0i from elastic modulus ratio (Eq. 8)
  - Local microscopic damage D0t from driving factor (Eq. 9)
  - Total damage variable with M-integral (Eqs. 10-12)
  - Damage constitutive model sigma-epsilon with Weibull-based
    subsequent damage (Eq. 14)
"""

import numpy as np


# ---------------------------------------------------------------------------
# M-integral on a 2-D discretised closed path (Eq. 1)
# ---------------------------------------------------------------------------

def strain_energy_density_2d(stress_xx, stress_yy, stress_xy,
                              strain_xx, strain_yy, strain_xy):
    """Strain energy density W = 0.5 * sigma_ij * epsilon_ij (2-D)."""
    return 0.5 * (stress_xx * strain_xx +
                  stress_yy * strain_yy +
                  2.0 * stress_xy * strain_xy)


def m_integral_2d(coords, normals, W, stress, displacement_grad):
    """Discretised M-integral along a closed contour (Eq. 1).

    Parameters
    ----------
    coords   : (N,2) array – (x1, x2) coordinates on contour
    normals  : (N,2) array – outward normal at each point (n1, n2)
    W        : (N,)  array – strain energy density at each point
    stress   : (N,2,2)     – Cauchy stress tensor [sigma_kj]
    displacement_grad : (N,2,2) – du_k / dx_i

    Returns
    -------
    M : scalar M-integral value (J/m)
    """
    N = len(coords)
    ds = np.zeros(N)
    for i in range(N):
        j = (i + 1) % N
        ds[i] = np.linalg.norm(coords[j] - coords[i])

    M = 0.0
    for i in range(N):
        x = coords[i]
        n = normals[i]
        sig = stress[i]
        ug = displacement_grad[i]
        # W * x_j * n_j
        term1 = W[i] * np.dot(x, n)
        # sigma_kj * u_{k,i} * x_i * n_j
        term2 = 0.0
        for k in range(2):
            for jj in range(2):
                for ii in range(2):
                    term2 += sig[k, jj] * ug[k, ii] * x[ii] * n[jj]
        M += (term1 - term2) * ds[i]
    return M


# ---------------------------------------------------------------------------
# Local mechanical failure driving factor (Eq. 7)
# ---------------------------------------------------------------------------

def failure_driving_factor(M_integral, Et, hole_area, n_cracks,
                            crack_length, crack_width, theta_rad, sigma):
    """Local mechanical failure driving factor Psi (Eq. 7).

    Psi = M / (Et * A * n * l * d * sin(2*theta + pi/4) * sigma)

    Parameters
    ----------
    M_integral   : M-integral value (J/m)
    Et           : instantaneous elastic modulus (Pa)
    hole_area    : total area of holes (m^2)
    n_cracks     : number of cracks
    crack_length : crack length (m)
    crack_width  : crack width (m)
    theta_rad    : angle between crack and bedding (radians)
    sigma        : applied stress (Pa)
    """
    A = max(hole_area, 1e-12)
    n = max(n_cracks, 1)
    l = max(crack_length, 1e-12)
    d = max(crack_width, 1e-12)
    angle_term = np.abs(np.sin(2.0 * theta_rad + np.pi / 4.0))
    angle_term = max(angle_term, 1e-12)
    denom = Et * A * n * l * d * angle_term * sigma
    return M_integral / denom


# ---------------------------------------------------------------------------
# Damage variables (Eqs. 8-12)
# ---------------------------------------------------------------------------

def initial_damage(Ed, Ei):
    """Initial damage D0i (Eq. 8).
    D0i = 1 - Ed / Ei
    """
    return 1.0 - np.asarray(Ed, dtype=float) / np.asarray(Ei, dtype=float)


def local_microscopic_damage(Ed, Ei, Et, psi_t, psi_p):
    """Local microscopic damage D0t before failure (Eq. 9).
    D0t = (1 - Ed/Ei) * (1 - Et/Ed) * (Psi_t / Psi_p)
    """
    d0i = 1.0 - Ed / Ei
    return d0i * (1.0 - Et / Ed) * (psi_t / psi_p)


def total_initial_damage(D0i, D0t):
    """Total initial damage D0 (Eq. 10-11)."""
    return D0i + D0t - D0i * D0t


# ---------------------------------------------------------------------------
# Damage constitutive model (Eq. 14) with Weibull subsequent damage
# ---------------------------------------------------------------------------

def weibull_damage(strain, m_weibull, eps_0):
    """Subsequent damage Df using Weibull distribution (Eq. 13).
    Df = 1 - exp(-(eps/eps_0)^m)
    """
    e = np.asarray(strain, dtype=float)
    return 1.0 - np.exp(-((e / eps_0) ** m_weibull))


def stress_strain_m_integral(strain, Ei, Ed, Et, psi_t, psi_p,
                              m_weibull, eps_0):
    """Full damage constitutive model (Eq. 14).

    sigma = Ei * (1 - D) * epsilon
    where D = D0 + Df - D0*Df  (combined damage).
    """
    eps = np.asarray(strain, dtype=float)
    D0i = initial_damage(Ed, Ei)
    D0t = local_microscopic_damage(Ed, Ei, Et, psi_t, psi_p)
    D0 = total_initial_damage(D0i, D0t)
    Df = weibull_damage(eps, m_weibull, eps_0)
    D = D0 + Df - D0 * Df
    D = np.clip(D, 0.0, 0.999)
    sigma = Ei * (1.0 - D) * eps
    return sigma, D


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== Article 2: M-integral Damage Model Demo ===\n")

    Ei = 25e9     # intact elastic modulus (Pa)
    Ed = 20e9     # defective rock modulus
    Et = 18e9     # modulus at local failure time
    psi_t = 4.5   # driving factor at time t
    psi_p = 5.84  # critical driving factor (material constant)

    D0i = initial_damage(Ed, Ei)
    D0t = local_microscopic_damage(Ed, Ei, Et, psi_t, psi_p)
    D0 = total_initial_damage(D0i, D0t)
    print(f"Initial damage  D0i = {D0i:.4f}")
    print(f"Micro damage    D0t = {D0t:.4f}")
    print(f"Total init dam  D0  = {D0:.4f}")

    strain = np.linspace(0, 0.015, 200)
    m_w, eps_0 = 3.0, 0.008
    sigma, D = stress_strain_m_integral(strain, Ei, Ed, Et, psi_t, psi_p,
                                        m_w, eps_0)
    idx_peak = np.argmax(sigma)
    print(f"\nPeak stress  = {sigma[idx_peak]/1e6:.1f} MPa at strain = {strain[idx_peak]:.4f}")
    print(f"Damage at peak = {D[idx_peak]:.4f}")
    print()


if __name__ == "__main__":
    demo()
