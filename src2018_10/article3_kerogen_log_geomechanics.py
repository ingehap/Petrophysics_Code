"""
Article 3: Integrating Measured Kerogen Properties With Log Analysis for
           Petrophysics and Geomechanics in Unconventional Resources
Craddock, Mosse, Prioul, Miles, Loan, Pirie, Rylander, Lewis, Pomerantz (2018)
DOI: 10.30632/PJV59N5-2018a2

Measured kerogen properties (skeletal density, elastic moduli) are integrated
into log analysis so that organic-rich mudstone petrophysics and geomechanics
are not biased by treating kerogen as part of the mineral matrix.  TOC converts
to a kerogen volume; the bulk density and the elastic moduli are then computed
with kerogen as a soft, light, separate solid component.

Implements:

  - TOC -> kerogen volume fraction
  - Matrix/kerogen bulk-density mixing and density porosity
  - Voigt-Reuss-Hill modulus mixing with a soft kerogen component
  - Dynamic Young's modulus / Poisson's ratio from the mixed moduli

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the kerogen-integrated petrophysics / geomechanics the paper applies.
"""

import numpy as np


# ---------------------------------------------- volumetrics -------------

def kerogen_volume(toc, rho_b, rho_k=1.30, carbon_frac=0.80):
    """Kerogen volume fraction from TOC  V_k = (TOC/carbon_frac)*rho_b/rho_k."""
    return (toc / carbon_frac) * rho_b / rho_k


def bulk_density(phi, vk, rho_f, rho_k, rho_ma):
    """Three-component bulk density (pore fluid + kerogen + mineral matrix)."""
    return phi * rho_f + vk * rho_k + (1.0 - phi - vk) * rho_ma


# ---------------------------------------------- moduli ------------------

def voigt_reuss_hill(fractions, moduli):
    """Voigt-Reuss-Hill average modulus of a composite."""
    f = np.asarray(fractions, float); m = np.asarray(moduli, float)
    voigt = np.sum(f * m)
    reuss = 1.0 / np.sum(f / m)
    return 0.5 * (voigt + reuss)


def youngs_from_k_mu(K, mu):
    """Young's modulus  E = 9*K*mu/(3*K + mu)."""
    return 9.0 * K * mu / (3.0 * K + mu)


def poisson_from_k_mu(K, mu):
    """Poisson's ratio  nu = (3*K - 2*mu)/(2*(3*K + mu))."""
    return (3.0 * K - 2.0 * mu) / (2.0 * (3.0 * K + mu))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Kerogen Properties in Log Analysis & Geomechanics")
    print("=" * 60)

    # Kerogen volume rises with TOC
    vk = kerogen_volume(0.08, 2.40)
    print(f"  kerogen volume (TOC 8%) = {vk:.3f}")
    assert vk > 0 and kerogen_volume(0.12, 2.40) > vk

    # Including light kerogen lowers the bulk density vs an all-mineral matrix
    rho_with_k = bulk_density(0.06, vk, 1.0, 1.30, 2.68)
    rho_no_k = bulk_density(0.06, 0.0, 1.0, 1.30, 2.68)
    print(f"  bulk density w/ , w/o kerogen = {rho_with_k:.3f} / {rho_no_k:.3f}")
    assert rho_with_k < rho_no_k

    # Soft kerogen lowers the composite moduli (VRH between bounds)
    K_rock = voigt_reuss_hill([0.85, 0.15], [40.0, 5.0])     # calcite + kerogen
    K_clean = 40.0
    print(f"  bulk modulus rock/clean = {K_rock:.1f} / {K_clean:.1f} GPa")
    assert 5.0 < K_rock < K_clean

    # Dynamic moduli from K, mu are physical
    E = youngs_from_k_mu(K_rock, 20.0)
    nu = poisson_from_k_mu(K_rock, 20.0)
    print(f"  E / nu                 = {E:.1f} GPa / {nu:.3f}")
    assert E > 0 and 0.0 < nu < 0.5
    # more kerogen softens the rock further -> lower E
    K_more_k = voigt_reuss_hill([0.7, 0.3], [40.0, 5.0])
    assert youngs_from_k_mu(K_more_k, 20.0) < E
    print("  PASS")
    return {"V_kerogen": float(vk), "K_rock": float(K_rock), "E": float(E)}


if __name__ == "__main__":
    test_all()
