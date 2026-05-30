"""
Article 6: Quantitative Demonstration of a High-Fidelity Oil-Based Mud
           Resistivity Imager Using a Controlled Experiment
Guner, Fouda, Ewe, Torres, Barrett (2021)
DOI: 10.30632/PJV62N4-2021a5

A high-resolution oil-based-mud imager (HROBMI) measures complex button-
electrode impedance at several MHz-range frequencies, enabling quantitative
formation resistivity / permittivity and standoff rather than just a blended
image.  A formation pixel is modeled as a parallel resistor-capacitor seen
through a (capacitive) oil-mud impedance.

Implements:

  - RC element values from medium properties  R = kb*rho, C = eps*eps0/kb (Eq.1)
  - Complex impedance  Z = R / (1 + j*w*R*C)                       (Eq. 2)
  - Apparent impedivity  xi = Z * kb  (Re(xi) ~ rho at low rho)     (Eqs. 3-4)
  - Magnitude / phase
  - DC-conductivity decoupling  sigma = sigma_DC + w*eps''*eps0    (Eq. 6)

Note: the journal's Eqs. 1-6 were image-rendered and not in the text; the
forms here are standard capacitively-coupled-imager physics consistent with
the paper's prose and nomenclature.  Resistivity in ohm-m, impedance in ohms,
frequency in Hz, permittivity relative (eps0 = 8.854e-12 F/m).
"""

import numpy as np

EPS0 = 8.854e-12       # F/m


# ---------------------------------------------- Eq. 1: RC elements ------

def rc_elements(rho, eps_r, kb):
    """Parallel-RC element values from medium properties (Eq. 1).

    R = kb * rho ;  C = eps_r * eps0 / kb.
    """
    R = kb * rho
    C = eps_r * EPS0 / kb
    return R, C


# ---------------------------------------------- Eq. 2: impedance --------

def complex_impedance(rho, eps_r, kb, freq_hz):
    """Complex impedance of the parallel RC  Z = R/(1 + j w R C)  (Eq. 2)."""
    R, C = rc_elements(rho, eps_r, kb)
    w = 2 * np.pi * freq_hz
    return R / (1.0 + 1j * w * R * C)


# ---------------------------------------------- Eqs. 3-4: impedivity ----

def apparent_impedivity(Z, kb):
    """Apparent impedivity  xi = Z / kb  (units ohm-m).

    Since R = kb*rho (Eq. 1), at low resistivity / low frequency Z ~ R and
    xi = Z/kb ~ rho (Eq. 3).
    """
    return Z / kb


def magnitude_phase(Z):
    """Return (|Z|, phase in degrees)."""
    return np.abs(Z), np.degrees(np.angle(Z))


# ---------------------------------------------- series mud + formation --

def series_impedance(Z_rock, Z_mud):
    """Total apparent impedance with a series (capacitive) mud term (Eq. 4)."""
    return Z_rock + Z_mud


# ---------------------------------------------- Eq. 6: decoupling -------

def formation_conductivity(sigma_dc, eps_imag, freq_hz):
    """sigma = sigma_DC + w * eps'' * eps0  (Eq. 6)."""
    w = 2 * np.pi * freq_hz
    return sigma_dc + w * eps_imag * EPS0


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Oil-Based-Mud Resistivity Imager")
    print("=" * 60)

    kb = 0.01            # tool constant (geometry; proprietary, assumed)
    f_lo, f_hi = 1e6, 20e6

    # Low resistivity: Re(apparent impedivity) ~ true rock resistivity (Eq. 3)
    rho, eps_r = 5.0, 10.0
    Z = complex_impedance(rho, eps_r, kb, f_lo)
    xi = apparent_impedivity(Z, kb)
    print(f"  rho=5 ohm-m: Re(xi) @1MHz = {xi.real:.3f} ohm-m")
    assert abs(xi.real - rho) / rho < 0.05, "low-rho limit: Re(xi) ~ rho"

    # Round trip: recover rho from the low-frequency impedance magnitude
    rho_rec = np.abs(Z) / kb               # |xi| ~ rho at low rho/low f
    assert abs(rho_rec - rho) / rho < 0.1

    # Dielectric rollover: at high resistivity the high frequency reads lower
    rho_hi = 1000.0
    xi_lo = apparent_impedivity(complex_impedance(rho_hi, eps_r, kb, f_lo), kb)
    xi_hi = apparent_impedivity(complex_impedance(rho_hi, eps_r, kb, f_hi), kb)
    print(f"  rho=1000: Re(xi) lo/hi f = {xi_lo.real:.1f} / {xi_hi.real:.1f}")
    assert xi_hi.real < xi_lo.real, "dielectric rollover at high resistivity"

    # Oil mud is strongly capacitive -> impedance phase near -90 deg
    Z_mud = complex_impedance(8000.0, 6.0, kb, f_lo)
    mag, phase = magnitude_phase(Z_mud)
    print(f"  mud impedance phase    = {phase:.1f} deg")
    assert phase < -45.0

    # Eq. 6: dielectric losses add to DC conductivity with frequency
    s_lo = formation_conductivity(0.01, eps_imag=5.0, freq_hz=f_lo)
    s_hi = formation_conductivity(0.01, eps_imag=5.0, freq_hz=f_hi)
    assert s_hi > s_lo > 0.01
    print("  PASS")
    return {"Re_xi_lowrho": xi.real, "mud_phase": phase}


if __name__ == "__main__":
    test_all()
