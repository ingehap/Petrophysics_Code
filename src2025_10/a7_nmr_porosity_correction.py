#!/usr/bin/env python3
"""
Article 7: Nuclear Magnetic Resonance T2 Spectrum and Porosity Correction
           Model of Shale Reservoir Based on Mineral Magnetic Susceptibility
Authors: Mingyue Zhu, Baozhi Pan, Bing Xie, et al.
Ref: Petrophysics, Vol. 66, No. 5 (October 2025), pp. 840-857.
     DOI: 10.30632/PJV66N5-2025a7

Implements:
  - NMR T2 relaxation model (bulk + surface + diffusion, Eqs. 1-3)
  - Rock magnetic susceptibility from mineral volumes (Eqs. 4-5)
  - Internal gradient magnetic field G from delta-chi
  - T2 spectrum correction to remove iron-mineral effects (Eqs. 8-9)
  - Porosity correction model (Eq. 10)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Mineral volume magnetic susceptibilities (×10^-6 SI)
# Table 2 in the article
# ---------------------------------------------------------------------------

MAGNETIC_SUSCEPTIBILITY = {
    'quartz':     -14.5,
    'plagioclase': -13.0,
    'k_feldspar':  -13.0,
    'calcite':     -13.0,
    'pyrite':      35.0,
    'ankerite':    900.0,
    'chlorite':    600.0,
    'illite':      100.0,
    'siderite':   1300.0,
    'kaolinite':   -12.0,
    'dolomite':    -13.0,
    'water':       -9.0,
}


# ---------------------------------------------------------------------------
# Rock magnetic susceptibility (Eqs. 4-5)
# ---------------------------------------------------------------------------

def rock_magnetic_susceptibility(mineral_volumes, minerals=None):
    """Compute rock matrix susceptibility chi_g (Eq. 4).

    Parameters
    ----------
    mineral_volumes : dict  mineral_name -> volume fraction (0-1)

    Returns
    -------
    chi_g : rock matrix susceptibility (×10^-6 SI)
    """
    chi_g = 0.0
    for mineral, vol in mineral_volumes.items():
        chi = MAGNETIC_SUSCEPTIBILITY.get(mineral, 0.0)
        chi_g += chi * vol
    return chi_g


def delta_chi(chi_matrix, chi_fluid=-9.0):
    """Magnetic susceptibility difference (Eq. 5).
    delta_chi = chi_matrix - chi_fluid  (×10^-6 SI)
    """
    return chi_matrix - chi_fluid


# ---------------------------------------------------------------------------
# NMR relaxation equations (Eqs. 1-3)
# ---------------------------------------------------------------------------

def t2_bulk(fluid='water'):
    """Bulk transverse relaxation time T2B (seconds)."""
    return 2.5 if fluid == 'water' else 1.0


def t2_surface(rho2, s_over_v):
    """Surface relaxation 1/T2S = rho2 * S/V (Eq. 1 contribution)."""
    return 1.0 / (rho2 * s_over_v)


def internal_gradient(B0, dchi_val, r_pore):
    """Internal magnetic field gradient G (Eq. 2 from article).
    G = delta_chi * B0 / (3 * r)   in T/m

    B0: static field (T), dchi: susceptibility diff (SI, not ×10^-6),
    r_pore: pore radius (m)
    """
    return np.abs(dchi_val) * B0 / (3.0 * np.maximum(r_pore, 1e-9))


def t2_diffusion(D, gamma, G, TE):
    """Diffusion relaxation time T2D (Eq. 3).
    1/T2D = D * (gamma * G * TE)^2 / 12
    D: diffusion coefficient (m^2/s), gamma: gyromagnetic ratio (rad/(s·T)),
    G: gradient (T/m), TE: echo interval (s)
    """
    val = D * (gamma * G * TE) ** 2 / 12.0
    if val < 1e-30:
        return 1e30
    return 1.0 / val


def t2_total(rho2, s_v, D, gamma, G, TE, T2B=2.5):
    """Total T2 (Eq. 1).
    1/T2 = 1/T2B + rho2*(S/V) + D*(gamma*G*TE)^2/12
    """
    inv = 1.0 / T2B + rho2 * s_v + D * (gamma * G * TE) ** 2 / 12.0
    return 1.0 / np.maximum(inv, 1e-30)


# ---------------------------------------------------------------------------
# T2 spectrum correction (Eqs. 6-9)
# ---------------------------------------------------------------------------

def generate_t2_spectrum(t2_values, amps, t2_bins):
    """Generate a T2 spectrum from a set of T2 components."""
    spectrum = np.zeros(len(t2_bins))
    for t2v, amp in zip(t2_values, amps):
        spectrum += amp * np.exp(-1.0 / (np.maximum(t2v, 1e-10)) *
                                 (1.0 / t2_bins))
    return spectrum


def correct_t2_decay(M_measured, echo_times, T2D):
    """Correct NMR decay by removing diffusion relaxation effect (Eq. 8).
    M'(t_i) = M(t_i) * exp(+t_i / T2D)
    """
    correction = np.exp(echo_times / T2D)
    return M_measured * correction


def nmr_porosity_from_decay(M_corrected, M0, scale_factor):
    """Corrected NMR porosity (Eq. 9).
    phi_corr = K * sum(M') / M0
    """
    return scale_factor * np.sum(M_corrected) / M0


# ---------------------------------------------------------------------------
# Porosity correction model (Eq. 10)
# ---------------------------------------------------------------------------

def porosity_correction(V_illite, V_ankerite, V_chlorite,
                        a=0.8, b=1.2, c=1.0, d=0.43):
    """NMR porosity correction amount delta_phi1 (Eq. 10).
    delta_phi1 = a * V_illite + b * V_ankerite + c * V_chlorite + d
    All volumes in %, returns correction in %.
    """
    return a * V_illite + b * V_ankerite + c * V_chlorite + d


def corrected_porosity(phi_nmr, dphi):
    """phi_corr = phi_NMR + delta_phi  (all in %)."""
    return phi_nmr + dphi


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== Article 7: NMR Porosity Correction Demo ===\n")

    # Example mineral composition
    minerals = {
        'quartz': 0.30, 'calcite': 0.15, 'dolomite': 0.05,
        'ankerite': 0.10, 'pyrite': 0.03, 'illite': 0.15,
        'chlorite': 0.08, 'kaolinite': 0.04,
    }
    chi_g = rock_magnetic_susceptibility(minerals)
    dchi = delta_chi(chi_g)
    print(f"Matrix susceptibility : {chi_g:.1f} ×10^-6 SI")
    print(f"Delta chi             : {dchi:.1f} ×10^-6 SI")

    # NMR parameters
    B0 = 0.047  # T (2 MHz)
    r_pore = 0.5e-6  # 0.5 microns
    G = internal_gradient(B0, dchi * 1e-6, r_pore)
    D = 2.3e-9  # m^2/s (water at 25C)
    gamma = 2.675e8  # rad/(s·T) proton
    TE = 100e-6  # 100 us
    T2D_val = t2_diffusion(D, gamma, G, TE)
    rho2 = 5e-6  # m/s
    s_v = 2.0 / r_pore
    T2 = t2_total(rho2, s_v, D, gamma, G, TE)
    print(f"Internal gradient G   : {G:.2f} T/m")
    print(f"T2D                   : {T2D_val:.4f} s")
    print(f"Total T2              : {T2*1000:.2f} ms")

    # Porosity correction
    V_il, V_an, V_ch = 15.0, 10.0, 8.0  # volume %
    dphi = porosity_correction(V_il, V_an, V_ch)
    phi_nmr = 2.5  # measured NMR porosity %
    phi_corr = corrected_porosity(phi_nmr, dphi)
    print(f"\nPorosity correction   : {dphi:.2f} %")
    print(f"Corrected porosity    : {phi_corr:.2f} %")
    print()


if __name__ == "__main__":
    demo()
