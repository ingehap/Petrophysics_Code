"""
Article 5: Borehole Measurements of the Complex-Resistivity Dispersion Spectrum:
           A New Logging Method to Identify Low-Resistivity Reservoirs
Jiang, Ke, Kang, Sun, Yin (2017)
Reference: Petrophysics Vol. 58, No. 3 (June 2017), pp. 281-288
DOI: none assigned (this issue predates SPWLA DOI assignment)

A borehole tool measures the complex-resistivity (induced-polarization)
dispersion spectrum, fit with the Cole-Cole model.  The characteristic frequency
(the trough of the imaginary part, Fb = 1/tau) has a power-law relationship with
water-filled porosity, so the spectrum gives water-filled porosity - and hence
water saturation - independent of the resistivity magnitude, identifying
low-resistivity pay.

Implements:

  - Cole-Cole complex resistivity  rho(w) = rho0*[1 - eta*(1 - 1/(1 + (i*w*tau)^c))]
  - Characteristic frequency  Fb = 1/tau  (imaginary-part trough)
  - Power-law water-filled porosity from Fb  phi_w = a*Fb^b
  - Water saturation  Sw = phi_w/phi

Note: this issue's PDF has a text layer; the Cole-Cole variables are defined in
the prose while the typeset formula glyph was dropped, so it is the standard
Cole-Cole reconstruction.  Resistivity in ohm.m, frequency in Hz.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Cole-Cole --------------

def cole_cole(omega, rho0, eta, tau, c):
    """Cole-Cole complex resistivity (Eq. 1)

        rho(w) = rho0*[1 - eta*(1 - 1/(1 + (i*w*tau)^c))],

    rho0 = DC resistivity, eta = chargeability (0-1), tau = relaxation time,
    c = frequency exponent (0-1).
    """
    return petrolib.em_dielectric.cole_cole_resistivity(
        omega / (2.0 * np.pi), rho0=rho0, chargeability=eta, tau=tau, c=c
    )


def characteristic_frequency(tau):
    """Characteristic frequency  Fb = 1/tau (the imaginary-part trough)."""
    return 1.0 / tau


def water_filled_porosity(fb, a, b):
    """Water-filled porosity from the characteristic frequency  phi_w = a*Fb^b (power law)."""
    return a * np.asarray(fb, float) ** b


def water_saturation(phi_w, phi):
    """Water saturation  Sw = phi_w/phi, clipped to [0, 1]."""
    return np.clip(phi_w / phi, 0.0, 1.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Complex-Resistivity Dispersion")
    print("=" * 60)

    rho0, eta, tau, c = 10.0, 0.4, 1e-3, 0.5
    # Low-frequency limit -> rho0 (real); high-frequency limit -> rho0*(1 - eta)
    lo = cole_cole(1e-3, rho0, eta, tau, c)
    hi = cole_cole(1e9, rho0, eta, tau, c)
    print(f"  |rho| low / high f     = {abs(lo):.3f} / {abs(hi):.3f} ohm.m")
    assert np.isclose(lo.real, rho0, atol=1e-2)
    assert np.isclose(hi.real, rho0 * (1 - eta), atol=1e-2)

    # The imaginary part has a polarization trough (negative)
    omega = 2 * np.pi * np.logspace(0, 6, 400)
    imag = cole_cole(omega, rho0, eta, tau, c).imag
    print(f"  min Im(rho)            = {imag.min():.3f}")
    assert imag.min() < 0

    # Characteristic frequency, power-law porosity, and water saturation
    fb = characteristic_frequency(tau)
    phi_w = water_filled_porosity(fb, a=1e-3, b=0.5)
    assert characteristic_frequency(1e-3) == 1000.0
    assert water_filled_porosity(4 * fb, 1e-3, 0.5) > phi_w     # rises with Fb
    sw = water_saturation(0.05, 0.10)
    assert np.isclose(sw, 0.5)
    print("  PASS")
    return {"Fb": float(fb), "Sw": float(sw)}


if __name__ == "__main__":
    test_all()
