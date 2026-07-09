"""
Article 3: Deducing Electrical Permittivity of Formations From LWD Resistivity
           Measurements
Stalheim (2019)
DOI: 10.30632/PJV60N6-2019a3

Classical EM plane-wave theory extracts the dispersive permittivity and a
dielectric-corrected resistivity from LWD propagation resistivities (phase and
amplitude), without a proprietary inversion.  The complex wavenumber is
decomposed into real and imaginary parts that determine the conductivity and
permittivity; a CRIM mixing law and the imaginary wavenumber give water
saturation, and the validity is classified by the fraction of a wavelength.

Implements:

  - Lossy-medium complex wavenumber  k_r, k_i  from sigma, eps, f   (Eqs. 1-3, 7-8)
  - Conductivity  sigma = 2*k_r*k_i/(w*mu*mu0)                      (Eq. 4)
  - Permittivity  eps_r = (k_r^2 - k_i^2)/(w^2*mu0*eps0)            (Eq. 5)
  - Wavelength  lambda = 2*pi/k_r                                   (Eq. 13)
  - CRIM mixing and water saturation  Sw = k_i/(phi*k_wi)          (Eqs. 14, 20)

Note: this issue's PDF text layer kept the equation numbers and variable
definitions but dropped the typeset glyphs, so these are the standard
lossy-medium / CRIM forms anchored to those definitions.  Paper anchors:
mu = 1, dielectric effect significant for Ra > 10 ohm-m, k_wi = 12.5.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

MU0 = 4e-7 * np.pi
EPS0 = 8.854e-12
C_LIGHT = 1.0 / np.sqrt(MU0 * EPS0)


# ---------------------------------------------- wavenumber --------------

def complex_wavenumber(sigma, eps_r, freq_hz, mu_r=1.0):
    """Real / imaginary wavenumber of a lossy medium  (Eqs. 1-3, 7-8).

    Returns (k_r, k_i) for E ~ exp(i*k_r*z - k_i*z).
    """
    w = 2.0 * np.pi * freq_hz
    base = w * np.sqrt(MU0 * mu_r * EPS0 * eps_r)
    p = sigma / (w * EPS0 * eps_r)                    # loss tangent
    root = np.sqrt(1.0 + p ** 2)
    k_r = base * np.sqrt((root + 1.0) / 2.0)
    k_i = base * np.sqrt((root - 1.0) / 2.0)
    return k_r, k_i


def conductivity_from_wavenumber(k_r, k_i, freq_hz, mu_r=1.0):
    """Conductivity  sigma = 2*k_r*k_i/(w*mu_r*mu0)  (Eq. 4)."""
    w = 2.0 * np.pi * freq_hz
    return 2.0 * k_r * k_i / (w * mu_r * MU0)


def permittivity_from_wavenumber(k_r, k_i, freq_hz, mu_r=1.0):
    """Relative permittivity  eps_r = (k_r^2 - k_i^2)/(w^2*mu_r*mu0*eps0)  (Eq. 5)."""
    w = 2.0 * np.pi * freq_hz
    return (k_r ** 2 - k_i ** 2) / (w ** 2 * mu_r * MU0 * EPS0)


def wavelength(k_r):
    """Wavelength  lambda = 2*pi/k_r  (Eq. 13)."""
    return 2.0 * np.pi / k_r


# ---------------------------------------------- CRIM --------------------

def crim_permittivity(phi, sw, eps_w=78.0, eps_hc=2.0, eps_m=5.0):
    """CRIM effective permittivity  sqrt(eps) = phi*Sw*sqrt(ew) + phi*(1-Sw)*sqrt(ehc)
    + (1-phi)*sqrt(em)  (Eq. 14, high-frequency form)."""
    return petrolib.em_dielectric.crim(
        phi, sw, eps_w=eps_w, eps_hc=eps_hc, eps_matrix=eps_m
    )


def water_saturation_crim(eps_eff, phi, eps_w=78.0, eps_hc=2.0, eps_m=5.0):
    """Invert CRIM for water saturation."""
    return petrolib.em_dielectric.sw_from_permittivity(
        eps_eff, phi, eps_w=eps_w, eps_hc=eps_hc, eps_matrix=eps_m, clip=False
    )


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Permittivity From LWD Resistivity")
    print("=" * 60)

    # Round-trip: plant sigma & eps_r, forward to (k_r,k_i), recover them
    sigma, eps_r, f = 0.05, 15.0, 2e6          # 2 MHz LWD
    k_r, k_i = complex_wavenumber(sigma, eps_r, f)
    sig_rec = conductivity_from_wavenumber(k_r, k_i, f)
    eps_rec = permittivity_from_wavenumber(k_r, k_i, f)
    print(f"  sigma rec / eps rec    = {sig_rec:.4f} / {eps_rec:.2f}")
    assert abs(sig_rec - sigma) < 1e-9 and abs(eps_rec - eps_r) < 1e-6

    # Speed of light from the constants (sanity)
    assert abs(C_LIGHT - 2.998e8) / 2.998e8 < 1e-3

    # Wavelength shortens at higher frequency and lower resistivity
    kr_lo, _ = complex_wavenumber(0.01, 15.0, 4e5)
    kr_hi, _ = complex_wavenumber(0.01, 15.0, 2e6)
    print(f"  wavelength 400kHz/2MHz = {wavelength(kr_lo):.2f} / {wavelength(kr_hi):.2f} m")
    assert wavelength(kr_hi) < wavelength(kr_lo)

    # CRIM: permittivity rises with water saturation; inversion round-trips
    eps_wet = crim_permittivity(0.25, 0.9)
    eps_oil = crim_permittivity(0.25, 0.2)
    print(f"  CRIM eps wet / oil     = {eps_wet:.1f} / {eps_oil:.1f}")
    assert eps_wet > eps_oil
    assert abs(water_saturation_crim(eps_wet, 0.25) - 0.9) < 1e-9
    print("  PASS")
    return {"sigma_rec": float(sig_rec), "eps_rec": float(eps_rec),
            "eps_wet": float(eps_wet)}


if __name__ == "__main__":
    test_all()
