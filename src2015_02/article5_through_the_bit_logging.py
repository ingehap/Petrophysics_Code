"""
Article 5: Recharacterization and Validation of Through-the-Bit-Logging Tool
           Measurements
Slocombe, Bammi, Hunka, Reischman, Schmid (2015)
Reference: Petrophysics Vol. 56, No. 1 (February 2015), pp. 58-71
DOI: none assigned (this issue predates SPWLA DOI assignment)

A small-diameter through-the-bit logging (TBL) tool is recharacterized and
validated against standard formations.  This module implements the density
processing the paper documents: the log (apparent) density from the electron
density index, the electron density index from atomic number/mass, and the
spine-and-ribs mudcake (standoff) compensation that combines the long- and
short-spacing densities.

Implements:

  - Log density from electron density index  rho_b = 1.0704*rho_e - 0.188  (Eq. 1)
  - Electron density index  rho_e = (2Z/A)*rho_m  (single element and mixture; Eq. 2)
  - Spine-and-ribs compensated density  rho_b = rho_LS + d_rho_mc  (Eq. 3)
  - Long-short density difference  d_rho = rho_LS - rho_SS  (Eq. 5)
  - Mudcake (standoff) correction as a polynomial of d_rho (rib; Eq. 4)

Note: this issue's PDF has a text layer; the density relations (Eqs. 1-5) are
transcribed from the body, while the typeset glyphs were dropped and
reconstructed in standard form (Ellis & Singer, 2007).  Density in g/cm^3.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- density conversions --------------

def log_density(rho_e):
    """Log (apparent) density from the electron density index (Eq. 1)

        rho_b = 1.0704*rho_e - 0.188.
    """
    return petrolib.porosity_lithology.electron_density_to_bulk(rho_e, a=1.0704, b=-0.188)


def electron_density_index(z, a, rho_m):
    """Electron density index for a single element (Eq. 2)

        rho_e = (2Z/A)*rho_m,

    with Z the atomic number, A the atomic mass and rho_m the true mass density.
    """
    return petrolib.nuclear.electron_density_index(z, a, rho_m)


def electron_density_mixture(z_list, a_list, mass_fractions, rho_m):
    """Electron density index for a mixture (Eq. 2, mass-fraction weighted)

        rho_e = 2*sum_i(w_i*Z_i/A_i)*rho_m.
    """
    return petrolib.nuclear.electron_density_mixture(z_list, a_list, mass_fractions, rho_m)


# ---------------------------------------------- spine-and-ribs --------------

def density_difference(rho_ls, rho_ss):
    """Long-short spacing density difference  d_rho = rho_LS - rho_SS  (Eq. 5)."""
    return rho_ls - rho_ss


def mudcake_correction(delta_rho, coeffs=(0.0, 1.0, 0.0)):
    """Mudcake (standoff) correction from the long-short density difference (Eq. 4)

        d_rho_mc = g(d_rho) = c0 + c1*d_rho + c2*d_rho^2,

    the spine-and-ribs "rib" fitted to characterization data.
    """
    c0, c1, c2 = coeffs
    return c0 + c1 * delta_rho + c2 * delta_rho ** 2


def spine_and_ribs_density(rho_ls, rho_ss, coeffs=(0.0, 1.0, 0.0)):
    """Spine-and-ribs compensated bulk density (Eqs. 3-5)

        rho_b = rho_LS + d_rho_mc(rho_LS - rho_SS),

    correcting the long-spacing density for mudcake/standoff using the
    long-short difference.
    """
    return rho_ls + mudcake_correction(density_difference(rho_ls, rho_ss), coeffs)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Through-the-Bit-Logging Density")
    print("=" * 60)

    # Electron density index and log density: near-equal to true density for
    # common matrices (e.g. quartz Z/A ~ 0.499)
    rho_e = electron_density_index(z=10, a=20.03, rho_m=2.65)   # SiO2-like 2Z/A~0.999
    rho_b = log_density(rho_e)
    print(f"  rho_e / rho_b          = {rho_e:.4f} / {rho_b:.4f} g/cm^3")
    assert abs(rho_b - 2.65) < 0.05

    # Mixture electron density (mass-fraction weighted)
    rho_e_mix = electron_density_mixture([14, 8], [28.09, 16.0], [0.4674, 0.5326], 2.65)
    assert rho_e_mix > 0

    # Spine-and-ribs: with no density difference, no correction
    assert np.isclose(spine_and_ribs_density(2.40, 2.40), 2.40)
    # A long-short difference drives a mudcake correction
    rb = spine_and_ribs_density(2.30, 2.20, coeffs=(0.0, 0.5, 0.0))
    print(f"  compensated density    = {rb:.4f} g/cm^3")
    assert np.isclose(density_difference(2.30, 2.20), 0.10)
    assert np.isclose(rb, 2.30 + 0.5 * 0.10)
    print("  PASS")
    return {"rho_b": float(rho_b), "rho_compensated": float(rb)}


if __name__ == "__main__":
    test_all()
