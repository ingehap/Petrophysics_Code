"""
Article 4: Utilization of Electromagnetic Acoustic Transducers in Downhole
           Cement Evaluation
Patterson, Bolshakov, Matuszyk (2015)
Reference: Petrophysics Vol. 56, No. 5 (October 2015), pp. 479-492
DOI: none assigned (this issue predates SPWLA DOI assignment)

Electromagnetic acoustic transducers (EMATs) generate guided shear-horizontal
(SH) and Lamb (A0) waves in the casing and measure their circumferential
propagation.  SH-mode attenuation depends only on the cement shear modulus
(density times shear-velocity squared), so it works even for lightweight cements
where the conventional cement-bond log fails.  This module implements the SH
plate-mode group velocity and cutoff, the shear modulus, an SH attenuation model
vs. cement shear modulus, and the Rayleigh-Lamb dispersion residual for the A0
mode.

Implements:

  - SH plate-mode cutoff frequency  f_n = n*Vs/(2h)
  - SH plate-mode group velocity (Eq. 1; SH0 nondispersive, higher orders dispersive)
  - Shear modulus  G = rho*Vs^2
  - SH attenuation vs. cement shear modulus (zero against a liquid)
  - Rayleigh-Lamb dispersion residual for symmetric/antisymmetric modes (Eq. 3)

Note: this issue's PDF has a text layer (Article 4's body is partly present);
the SH/Lamb relations (Eqs. 1-3) are transcribed from the body, while the
typeset glyphs were dropped and reconstructed in standard plate-wave form.
Velocities in m/s, thickness in m, frequency in Hz, density in kg/m^3.
"""

import numpy as np


# ---------------------------------------------- SH plate modes --------------

def sh_cutoff_frequency(vs, thickness, n):
    """SH plate-mode cutoff frequency  f_n = n*Vs/(2*h)  (Eq. 2).

    SH0 (n = 0) has no cutoff; mode n propagates only above f_n.
    """
    return n * vs / (2.0 * thickness)


def sh_group_velocity(vs, thickness, freq, n):
    """SH plate-mode group velocity (Eq. 1)

        Vg,n = Vs*sqrt(1 - (f_n/f)^2)   for f > f_n,  else 0 (evanescent),

    so SH0 is constant (= Vs) and higher-order modes are dispersive and slower.
    """
    fn = sh_cutoff_frequency(vs, thickness, n)
    if freq <= fn:
        return 0.0
    return vs * np.sqrt(1.0 - (fn / freq) ** 2)


# ---------------------------------------------- cement properties --------------

def shear_modulus(rho, vs):
    """Shear modulus  G = rho*Vs^2  (the cement property the SH attenuation reads)."""
    return rho * vs ** 2


def sh_attenuation(rho_cement, vs_cement, geometric=3.0, scale=1.0e-7):
    """SH-mode attenuation vs. cement shear properties (dB/ft)

        att = geometric + scale*rho_cement*Vs_cement,

    increasing with the cement shear impedance / shear modulus; for a liquid on
    the casing backside (Vs_cement = 0) only the small geometric-spreading term
    remains (no shear coupling).
    """
    return geometric + scale * rho_cement * vs_cement


# ---------------------------------------------- Lamb dispersion --------------

def rayleigh_lamb_residual(phase_velocity, freq, thickness, vp, vs, symmetric=False):
    """Rayleigh-Lamb dispersion residual (Eq. 3); roots in phase velocity give
    the Lamb-mode dispersion curves.

        antisymmetric (A0):  tan(q h)/q + 4 k^2 p tan(p h)/(q^2 - k^2)^2 = 0
        symmetric    (S0):   tan(q h)/q + (q^2 - k^2)^2 tan(p h)/(4 k^2 p) = 0,

    with k = omega/c, p = sqrt((omega/vp)^2 - k^2), q = sqrt((omega/vs)^2 - k^2),
    omega = 2*pi*f and h the half-thickness.  Returns the residual (0 at a mode).
    """
    omega = 2.0 * np.pi * freq
    h = thickness / 2.0
    k = omega / phase_velocity
    p = np.sqrt(np.complex128((omega / vp) ** 2 - k ** 2))
    q = np.sqrt(np.complex128((omega / vs) ** 2 - k ** 2))
    if symmetric:
        res = np.tan(q * h) / q + (q ** 2 - k ** 2) ** 2 * np.tan(p * h) / (4.0 * k ** 2 * p)
    else:
        res = np.tan(q * h) / q + 4.0 * k ** 2 * p * np.tan(p * h) / (q ** 2 - k ** 2) ** 2
    return res.real


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: EMAT Cement Evaluation")
    print("=" * 60)

    vs_steel = 3250.0
    h = 0.008                                     # 8 mm casing

    # SH0 is nondispersive and equals the shear velocity; SH1 is dispersive/slower
    assert sh_group_velocity(vs_steel, h, 200e3, 0) == vs_steel
    fc1 = sh_cutoff_frequency(vs_steel, h, 1)
    vg1 = sh_group_velocity(vs_steel, h, 400e3, 1)
    print(f"  SH1 cutoff / Vg        = {fc1/1e3:.1f} kHz / {vg1:.0f} m/s")
    assert 0 < vg1 < vs_steel
    # Below cutoff the SH1 mode is evanescent
    assert sh_group_velocity(vs_steel, h, fc1 * 0.5, 1) == 0.0

    # Shear modulus and SH attenuation: zero shear coupling against a liquid
    g = shear_modulus(1800.0, 1500.0)
    att_cement = sh_attenuation(1800.0, 1500.0)
    att_liquid = sh_attenuation(1000.0, 0.0)
    print(f"  cement G / att         = {g:.2e} Pa / {att_cement:.1f} dB/ft")
    assert att_cement > att_liquid and np.isclose(att_liquid, 3.0)
    # Stiffer (faster) cement attenuates more
    assert sh_attenuation(1900.0, 2500.0) > att_cement

    # Rayleigh-Lamb residual is finite and changes sign across an A0 mode
    r1 = rayleigh_lamb_residual(2000.0, 250e3, h, 5930.0, vs_steel, symmetric=False)
    r2 = rayleigh_lamb_residual(3200.0, 250e3, h, 5930.0, vs_steel, symmetric=False)
    print(f"  Lamb residual @2000/3200 = {r1:.3e} / {r2:.3e}")
    assert np.isfinite(r1) and np.isfinite(r2)
    print("  PASS")
    return {"SH1_cutoff_kHz": float(fc1 / 1e3), "G": float(g), "att_cement": float(att_cement)}


if __name__ == "__main__":
    test_all()
