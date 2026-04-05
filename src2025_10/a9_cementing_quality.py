#!/usr/bin/env python3
"""
Article 9: Application of Slip Interface Theory in the Evaluation of
           Casing Well Cementing Quality
Authors: Jinlin Pan, Xuelian Chen, Yuanda Su, Xiaoming Tang
Ref: Petrophysics, Vol. 66, No. 5 (October 2025), pp. 872-885.
     DOI: 10.30632/PJV66N5-2025a9

Implements:
  - Cased-well cylindrical model parameters
  - Slip interface boundary condition (Eqs. 3-6)
  - Acoustic pressure field in borehole (simplified, Eq. 1)
  - Relative amplitude vs. shear coupling stiffness
  - Relative amplitude vs. uncemented sector angle (USA)
  - Inversion of shear coupling stiffness from first-arrival amplitude
"""

import numpy as np
from scipy.special import jv as besselj  # Bessel J


# ---------------------------------------------------------------------------
# Default cased-well parameters (Table 1 in article)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    'r_inner':    0.06067,   # inner radius of casing (m)  = 139.7/2 - 9.19 mm
    'r_outer':    0.06985,   # outer radius of casing (m)  = 139.7/2 mm
    'r_wellbore': 0.08985,   # wellbore radius (m)         = 179.7/2 mm
    'vp_casing':  5860.0,    # P-wave velocity in casing (m/s)
    'vs_casing':  3130.0,    # S-wave velocity in casing (m/s)
    'rho_casing': 7850.0,    # casing density (kg/m^3)
    'vp_cement':  2823.0,    # P-wave in cement (m/s)
    'vs_cement':  1620.0,    # S-wave in cement (m/s)
    'rho_cement': 1900.0,    # cement density (kg/m^3)
    'vp_mud':     1500.0,    # P-wave in borehole fluid (m/s)
    'rho_mud':    1200.0,    # mud density (kg/m^3)
    'vp_form':    3500.0,    # P-wave in formation (m/s)
    'vs_form':    2000.0,    # S-wave in formation (m/s)
    'rho_form':   2300.0,    # formation density (kg/m^3)
    'f0':         20000.0,   # source frequency (Hz)
}


# ---------------------------------------------------------------------------
# Gaussian source spectrum (Eq. 7)
# ---------------------------------------------------------------------------

def gaussian_source_spectrum(f, f0, bandwidth=0.5):
    """Gaussian source spectrum G(f) = exp(-((f-f0)/(bw*f0))^2)."""
    return np.exp(-((f - f0) / (bandwidth * f0)) ** 2)


# ---------------------------------------------------------------------------
# Simplified acoustic pressure (axisymmetric, Eq. 1 form)
# ---------------------------------------------------------------------------

def borehole_pressure(r, z, omega, kf, A0=1.0, source_amp=1.0):
    """Simplified borehole pressure field (Eq. 1 concept).
    phi_f ~ A0 * J0(kr * r) * exp(ikz * z - i*omega*t)

    Returns amplitude at r, z for a given angular frequency omega.
    """
    kz = np.sqrt(np.maximum(kf ** 2 - (1.0 / r) ** 2, 0.0))
    kr = np.sqrt(np.maximum(kf ** 2 - kz ** 2, 0.0))
    p = np.abs(source_amp * A0 * besselj(0, kr * r) * np.exp(1j * kz * z))
    return p


# ---------------------------------------------------------------------------
# Coupling stiffness matrix (Eq. 6)
# ---------------------------------------------------------------------------

def coupling_stiffness_matrix(eta_N, eta_T):
    """6×6 slip-interface coupling matrix M (Eq. 6).
    Diagonal: 1/(1 + eta_N^-1 * ...) etc.
    Simplified to capture shear vs normal stiffness effects.
    """
    M = np.eye(6)
    # Normal coupling modifies radial displacement (index 0)
    M[0, 0] = eta_N / (eta_N + 1e-10)
    # Shear coupling modifies tangential and axial (indices 1,2)
    M[1, 1] = eta_T / (eta_T + 1e-10)
    M[2, 2] = eta_T / (eta_T + 1e-10)
    return M


# ---------------------------------------------------------------------------
# Relative casing-wave amplitude models
# ---------------------------------------------------------------------------

def relative_amplitude_slip(eta_T, eta_T_free=0.1, eta_T_bonded=1e4,
                             amp_free=0.50, amp_bonded=0.02):
    """Relative first-arrival amplitude as function of shear coupling stiffness.

    Empirical logistic fit to theoretical curves in Fig. 3b.
    eta_T in GPa/m.
    """
    eta = np.asarray(eta_T, dtype=float)
    log_eta = np.log10(np.maximum(eta, 1e-3))
    log_free = np.log10(eta_T_free)
    log_bond = np.log10(eta_T_bonded)
    frac = (log_eta - log_free) / (log_bond - log_free)
    frac = np.clip(frac, 0, 1)
    amp = amp_free + (amp_bonded - amp_free) * frac ** 1.5
    return amp


def relative_amplitude_usa(usa_deg, amp_0=0.02, amp_360=0.50):
    """Relative amplitude vs. uncemented sector angle (USA).

    Linear interpolation based on Fig. 3c.
    """
    usa = np.clip(np.asarray(usa_deg, dtype=float), 0, 360)
    return amp_0 + (amp_360 - amp_0) * (usa / 360.0)


# ---------------------------------------------------------------------------
# Inversion: shear coupling stiffness from measured amplitude
# ---------------------------------------------------------------------------

def invert_coupling_stiffness(measured_amplitude,
                               eta_T_free=0.1, eta_T_bonded=1e4,
                               amp_free=0.50, amp_bonded=0.02):
    """Invert shear coupling stiffness from measured relative amplitude."""
    amp = np.clip(measured_amplitude, amp_bonded, amp_free)
    frac = ((amp - amp_free) / (amp_bonded - amp_free)) ** (1.0 / 1.5)
    log_eta = np.log10(eta_T_free) + frac * (np.log10(eta_T_bonded) - np.log10(eta_T_free))
    return 10.0 ** log_eta


# ---------------------------------------------------------------------------
# Cement quality classification
# ---------------------------------------------------------------------------

def classify_cement_quality(relative_amp):
    """Classify based on conventional 15%/30% thresholds."""
    amp = np.asarray(relative_amp, dtype=float)
    quality = np.where(amp <= 0.15, 'Good',
              np.where(amp <= 0.30, 'Medium', 'Poor'))
    return quality


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== Article 9: Slip Interface Cementing Quality Demo ===\n")

    # Relative amplitude vs coupling stiffness
    eta_range = np.logspace(-1, 4, 50)  # GPa/m
    amp = relative_amplitude_slip(eta_range)
    print("Shear stiffness (GPa/m) -> Relative amplitude:")
    for eta_val in [0.1, 1, 10, 100, 1000, 10000]:
        a = relative_amplitude_slip(eta_val)
        print(f"  eta_T = {eta_val:8.1f}  ->  amp = {a:.3f}  ({classify_cement_quality(a)})")

    # USA model
    print("\nUSA (degrees) -> Relative amplitude:")
    for usa in [0, 30, 90, 171, 241, 360]:
        a = relative_amplitude_usa(usa)
        print(f"  USA = {usa:3d}°  ->  amp = {a:.3f}  ({classify_cement_quality(a)})")

    # Inversion example
    amp_measured = 0.12
    eta_inv = invert_coupling_stiffness(amp_measured)
    print(f"\nInversion: amp = {amp_measured} -> eta_T = {eta_inv:.1f} GPa/m")
    print()


if __name__ == "__main__":
    demo()
