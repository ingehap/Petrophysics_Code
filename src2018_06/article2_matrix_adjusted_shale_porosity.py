"""
Article 2: Matrix-Adjusted Shale Porosity Measured in Horizontal Wells
Craddock, Mosse, Bernhardt, Ortiz, Gonzalez Tomassini, Pirie, Saldungaray,
Pomerantz (2018)
DOI: 10.30632/PJV59N3-2018a1

Total porosity in organic shales is biased if the grain (matrix) density ignores
kerogen, whose density (~1.1-1.5 g/cc) is far below the mineral grains.  This
module computes a matrix density that mixes the mineral grains and kerogen by
mass (a reciprocal volume average), converts the bulk-density log from its
measured electron density, and combines the two into the density porosity - the
wellsite workflow the paper builds from cuttings spectroscopy.

Implements:

  - Density porosity  phi = (rho_ma - rho_b)/(rho_ma - rho_fl)
  - Matrix density mixing  1/rho_ma = sum(w_i/rho_i) + w_ker/rho_ker
  - Electron-to-bulk density  rho_b = 1.0704*rho_e - 0.1883
  - Kerogen mass fraction from TOC  w_ker ~ 1.2*TOC

Note: this issue's PDF has a text layer; the electron-density conversion (Eq. 5)
survived verbatim, while the porosity / matrix-density relations (Eqs. 1-2) lost
their typeset glyphs and are faithful standard-form reconstructions from the
variable definitions.  Densities in g/cm^3, mass fractions dimensionless.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# Fixed mineral grain densities used in the paper (g/cm^3)
RHO_GRAIN = {"quartz": 2.65, "calcite": 2.71, "illite": 2.78, "pyrite": 5.00}


# ---------------------------------------------- densities --------------

def electron_to_bulk_density(rho_e):
    """Bulk density from logged electron density  rho_b = 1.0704*rho_e - 0.1883 (Eq. 5)."""
    return petrolib.porosity_lithology.electron_density_to_bulk(rho_e)


def matrix_density(mass_fracs, grain_densities, w_ker=0.0, rho_ker=1.43):
    """Matrix (grain) density by reciprocal mass mixing (Eq. 2)

        1/rho_ma = sum_i(w_i/rho_g,i) + w_ker/rho_ker.

    mass_fracs / grain_densities are equal-length arrays of mineral mass
    fractions and grain densities; w_ker, rho_ker are the kerogen mass fraction
    and (skeletal) density.
    """
    return petrolib.porosity_lithology.matrix_density_from_masses(
        mass_fracs, grain_densities, w_kerogen=w_ker, rho_kerogen=rho_ker)


def density_porosity(rho_b, rho_ma, rho_fl=1.0):
    """Total density porosity  phi = (rho_ma - rho_b)/(rho_ma - rho_fl)  (Eq. 1)."""
    return petrolib.porosity_lithology.density_porosity(rho_b, rho_ma, rho_fl)


def kerogen_mass_fraction(toc):
    """Kerogen mass fraction from total organic carbon  w_ker ~ 1.2*TOC."""
    return petrolib.porosity_lithology.kerogen_mass_fraction(toc, k=1.2)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Matrix-Adjusted Shale Porosity")
    print("=" * 60)

    # Electron-to-bulk conversion round-trips a known bulk density
    rho_e = (2.5 + 0.1883) / 1.0704
    assert np.isclose(electron_to_bulk_density(rho_e), 2.5)

    # A quartz/calcite/illite matrix, then the same with 8 wt% kerogen
    minerals = ["quartz", "calcite", "illite"]
    w_min = np.array([0.6, 0.25, 0.15])
    rho_g = np.array([RHO_GRAIN[m] for m in minerals])
    rma_dry = matrix_density(w_min, rho_g)
    w_ker = kerogen_mass_fraction(toc=0.067)            # ~8 wt% kerogen
    rma_ker = matrix_density(w_min * (1 - w_ker), rho_g, w_ker=w_ker, rho_ker=1.43)
    print(f"  matrix density dry/ker = {rma_dry:.3f} / {rma_ker:.3f} g/cc")
    assert rma_ker < rma_dry                            # kerogen lowers rho_ma

    # Ignoring kerogen overstates rho_ma and hence the porosity
    phi_dry = density_porosity(2.45, rma_dry)
    phi_ker = density_porosity(2.45, rma_ker)
    print(f"  porosity dry/ker       = {phi_dry:.3f} / {phi_ker:.3f}")
    assert phi_dry > phi_ker > 0

    # Sensitivity ~ 1 p.u. porosity per 0.1 g/cc kerogen-density error
    d_phi = abs(density_porosity(2.45, matrix_density(w_min * (1 - w_ker), rho_g,
                                                      w_ker=w_ker, rho_ker=1.53)) - phi_ker)
    print(f"  d(phi) for +0.1 rho_ker = {d_phi * 100:.2f} p.u.")
    assert 0.002 < d_phi < 0.02
    print("  PASS")
    return {"rho_ma_ker": float(rma_ker), "phi_ker": float(phi_ker)}


if __name__ == "__main__":
    test_all()
