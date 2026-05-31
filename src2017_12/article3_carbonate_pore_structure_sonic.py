"""
Article 3: Characterization of Pore Structure Variation and Permeability
           Heterogeneity in Carbonate Rocks Using MICP and Sonic Logs: Puguang
           Gas Field, China
Huang, Dou, Sun (2017)
Reference: Petrophysics Vol. 58, No. 6 (December 2017), pp. 576-591
DOI: none assigned (this issue predates SPWLA DOI assignment)

Carbonate permeability is heterogeneous because different pore types follow
different k-phi trends.  This module derives the sonic-log shear frame
flexibility factor (Sun, 2000) that discriminates pore type: elastic moduli come
from Vp, Vs and density; the dry-frame moduli follow a (1-phi)^gamma law, so the
flexibility factors invert from the moduli, and the shear factor classifies the
pore system (moldic vs. intercrystalline).  MICP tortuosity and the Leverett
J-function tie it to capillary data.

Implements:

  - Elastic moduli from velocities  mu = rho*Vs^2,  K = rho*Vp^2 - 4/3*mu
  - P/S velocities  Vp = sqrt((K + 4/3*mu)/rho),  Vs = sqrt(mu/rho)
  - Frame flexibility factors  gamma = ln(Kd/Ks)/ln(1-phi)  (and shear gamma_mu)
  - Pore-type classification from gamma_mu
  - MICP tortuosity  tau = phi*R/(8*K)  and the Leverett J-function

Note: this issue's PDF has a text layer; the velocity/J-function/flexibility
forms survived (decoded), while the inverted gamma equations (Eqs. 8-9) lost
their glyphs and are faithful standard-form reconstructions.  SI units; moduli
in Pa, velocities in m/s, density in kg/m^3.
"""

import numpy as np


# ---------------------------------------------- elastic --------------

def moduli_from_velocity(vp, vs, rho):
    """Bulk and shear moduli from velocities  mu = rho*Vs^2, K = rho*Vp^2 - 4/3*mu."""
    mu = rho * np.asarray(vs, float) ** 2
    kappa = rho * np.asarray(vp, float) ** 2 - 4.0 / 3.0 * mu
    return kappa, mu


def vp_velocity(kappa, mu, rho):
    """Compressional velocity  Vp = sqrt((K + 4/3*mu)/rho)."""
    return np.sqrt((kappa + 4.0 / 3.0 * mu) / rho)


def vs_velocity(mu, rho):
    """Shear velocity  Vs = sqrt(mu/rho)."""
    return np.sqrt(mu / rho)


def flexibility_factor(modulus_dry, modulus_solid, phi):
    """Frame flexibility factor  gamma = ln(M_dry/M_solid)/ln(1 - phi)  (Eqs. 5-9).

    Applies to bulk (gamma) or shear (gamma_mu) moduli; inverts M_dry =
    M_solid*(1 - phi)^gamma.
    """
    return np.log(modulus_dry / modulus_solid) / np.log(1.0 - np.asarray(phi, float))


def dry_modulus(modulus_solid, phi, gamma):
    """Dry-frame modulus  M_dry = M_solid*(1 - phi)^gamma  (Eqs. 5-6)."""
    return modulus_solid * (1.0 - np.asarray(phi, float)) ** gamma


def pore_type(gamma_mu):
    """Classify the carbonate pore system from the shear flexibility factor."""
    if gamma_mu < 4.0:
        return "moldic"
    if gamma_mu <= 8.0:
        return "meso/macro-intercrystalline"
    return "micro-intercrystalline"


# ---------------------------------------------- capillary --------------

def tortuosity(phi, perm, r_throat):
    """MICP tortuosity  tau = phi*R/(8*K)  (Eq. 1); K in same units as implied by R."""
    return phi * r_throat / (8.0 * perm)


def leverett_j(pc, k, phi, sigma, theta_deg):
    """Leverett J-function  J = Pc/(sigma*cos(theta))*sqrt(k/phi)  (Eq. 2)."""
    return np.asarray(pc, float) / (sigma * np.cos(np.radians(theta_deg))) * np.sqrt(k / phi)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Carbonate Pore Structure & Sonic")
    print("=" * 60)

    # Velocities round-trip through the moduli
    kappa, mu = 2.0e10, 1.2e10
    rho = 2500.0
    vp, vs = vp_velocity(kappa, mu, rho), vs_velocity(mu, rho)
    print(f"  Vp / Vs                = {vp:.0f} / {vs:.0f} m/s")
    assert vp > vs
    k2, m2 = moduli_from_velocity(vp, vs, rho)
    assert np.isclose(k2, kappa) and np.isclose(m2, mu)

    # Flexibility factor inverts the (1-phi)^gamma law
    phi = 0.20
    md = dry_modulus(mu, phi, 6.0)
    assert np.isclose(flexibility_factor(md, mu, phi), 6.0)

    # Pore-type classification from gamma_mu
    labels = [pore_type(g) for g in (2.0, 6.0, 12.0)]
    print(f"  pore types g=2/6/12    = {labels}")
    assert labels == ["moldic", "meso/macro-intercrystalline", "micro-intercrystalline"]

    # Capillary diagnostics are positive
    assert tortuosity(0.2, 10.0, 5.0) > 0
    assert leverett_j(2e5, 1e-13, 0.2, 0.03, 30.0) > 0
    print("  PASS")
    return {"Vp": float(vp), "Vs": float(vs)}


if __name__ == "__main__":
    test_all()
