"""
Article 6: A Physics-Based Model for the Dielectric Response of Shaly Sands and
           Continuous CEC Logging
Freed, Seleznev, Hou, Fellah, Little, Dumy, Sen (2018)
DOI: 10.30632/PJV59N3-2018a5

A physics-based bimodal dielectric model solves for the polarization of a charged
clay platelet, relates its surface conductivity to the cation exchange capacity
(CEC), and embeds the grain in the rock through differential effective medium
mixing - giving a continuous CEC log from dielectric-dispersion data.  When the
CEC is zero the model reduces to a standard (uncharged) dielectric mixing law.

Implements:

  - Surface conductivity from CEC  sigma_n = u*rho_clay*CEC*(1 - f_Stern)*D
  - CEC from the apparent surface conductivity (the inverse, for logging)
  - Whole-rock CEC  = CEC_clay * clay_fraction
  - Complex rock permittivity  eps* = eps + i*sigma/(omega*eps0)

Note: this issue's PDF has a text layer; the extraction captured the core
physics (Eqs. 1-12) but ended inside the nomenclature.  The CEC<->conductivity
relations are faithful standard-form reconstructions using the constants the
paper reports (Stern-layer fraction f_Stern ~ 0.70-0.86, D = 2e-9 m^2/s, clay
density ~2.8 g/cc).  SI units; eps0 = vacuum permittivity.
"""

import numpy as np

EPS0 = 8.8541878128e-12      # vacuum permittivity (F/m)
F_STERN = 0.70               # Stern-layer cation fraction (paper range 0.70-0.86)
DIFFUSIVITY = 2.0e-9         # counterion diffusion coefficient (m^2/s)
CLAY_DENSITY = 2.8           # clay grain density (g/cm^3)


def _mobility(diffusivity, temp_c=22.0):
    """Counterion mobility from the Nernst-Einstein relation  mu = e*D/(kB*T)."""
    e0, kb = 1.602176634e-19, 1.380649e-23
    return e0 * diffusivity / (kb * (temp_c + 273.15))


# ---------------------------------------------- CEC <-> conductivity --------------

def surface_conductivity(cec, f_stern=F_STERN, clay_density=CLAY_DENSITY,
                         diffusivity=DIFFUSIVITY, temp_c=22.0):
    """Clay surface (quadrature) conductivity from CEC (Eqs. 6-9)

        sigma_n = mu*rho_clay*CEC*(1 - f_Stern),

    with the diffuse-layer fraction (1 - f_Stern) carrying the polarization and
    mu the Nernst-Einstein counterion mobility.  CEC in C/g, rho in g/cm^3.
    """
    return _mobility(diffusivity, temp_c) * clay_density * np.asarray(cec, float) * (1.0 - f_stern)


def cec_from_conductivity(sigma_n, f_stern=F_STERN, clay_density=CLAY_DENSITY,
                          diffusivity=DIFFUSIVITY, temp_c=22.0):
    """Invert the surface-conductivity relation to log CEC continuously (C/g)."""
    return sigma_n / (_mobility(diffusivity, temp_c) * clay_density * (1.0 - f_stern))


def whole_rock_cec(cec_clay, clay_fraction):
    """Whole-rock CEC  = CEC_clay * clay_fraction."""
    return cec_clay * np.asarray(clay_fraction, float)


def complex_permittivity(eps_rel, sigma, omega):
    """Complex permittivity  eps* = eps + i*sigma/(omega*eps0)  (Eqs. 10-12)."""
    return eps_rel + 1j * sigma / (omega * EPS0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Dielectric Response & Continuous CEC")
    print("=" * 60)

    # CEC -> surface conductivity round-trips back to the same CEC
    cec = 0.10 * 96.485 / 1000.0                        # 10 meq/100g -> C/g
    sig_n = surface_conductivity(cec)
    print(f"  sigma_n for CEC        = {sig_n:.4e} S/m")
    assert np.isclose(cec_from_conductivity(sig_n), cec)

    # A larger Stern-layer fraction leaves less diffuse-layer polarization
    assert surface_conductivity(cec, f_stern=0.86) < surface_conductivity(cec, f_stern=0.70)

    # Zero CEC -> zero surface conductivity (model reduces to uncharged case)
    assert surface_conductivity(0.0) == 0.0

    # Whole-rock CEC scales with clay fraction
    assert np.isclose(whole_rock_cec(cec, 0.3), 0.3 * cec)

    # Complex permittivity: conductivity term dominates the imaginary part at low f
    omega = 2 * np.pi * 1e3
    eps = complex_permittivity(25.0, sig_n, omega)
    print(f"  eps* @1kHz             = {eps.real:.1f} + {eps.imag:.3e}j")
    assert eps.real == 25.0 and eps.imag > 0
    print("  PASS")
    return {"sigma_n": float(sig_n), "CEC_C_per_g": float(cec)}


if __name__ == "__main__":
    test_all()
