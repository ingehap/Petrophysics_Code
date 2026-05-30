"""
Article 6: Maximizing the Value of Pulsed-Neutron Logs - A Complex Case Study of
           Gas Pressure Assessment Through Casing
Cavalleri, Brouwer, Kodri, Rose, Brinks (2020)
DOI: 10.30632/PJV61N6-2020a6

Through-casing pulsed-neutron logs respond to gas density through three
measurements (sigma, neutron porosity, and the fast-neutron cross section FNXS)
that are each linear in gas density and follow a linear volumetric mixing law.
With water saturation fixed from openhole, the bulk gas sigma - proportional to
gas density - is inverted for reservoir gas density and hence formation gas
pressure, parameterized against pressure and temperature by a real-gas law.

Implements:

  - Bulk gas sigma  Sigma = rho_bulk * sum_e (w_e * sigma_e)   (Eq. 1)
  - Real-gas density  rho = P*M / (z*R*T)
  - Inversion: measured sigma -> gas density -> gas pressure

Note: this issue's PDF text layer drops the typeset glyph of Eq. 1; the form
here is the faithful standard reconstruction (bulk sigma proportional to gas
density via mass-weighted elemental capture cross sections).  The per-mass
elemental coefficients are set so that H2O returns ~22 c.u. at 1 g/cc, matching
the issue's fresh-water sigma.  Pressure in psi, T in kelvin.
"""

import numpy as np

R_GAS = 8.314            # J/mol/K
PSI_TO_PA = 6894.76

# Per-mass elemental capture coefficients (c.u. per g/cc): calibrated so that
# water (w_H = 0.112) returns ~22 c.u. at 1 g/cc.
ELEM_MASS_SIGMA = {"H": 196.0, "C": 5.0, "O": 0.2}


# ---------------------------------------------- gas sigma ---------------

def mass_capture(weight_fracs):
    """Mass-weighted capture coefficient  sum_e w_e * sigma_e  (c.u. per g/cc)."""
    return sum(w * ELEM_MASS_SIGMA[e] for e, w in weight_fracs.items())


def gas_sigma(rho_gas_gcc, weight_fracs):
    """Bulk gas sigma  Sigma = rho_bulk * sum_e(w_e sigma_e)  (Eq. 1)."""
    return rho_gas_gcc * mass_capture(weight_fracs)


def gas_density_from_sigma(sigma_cu, weight_fracs):
    """Invert Eq. 1 for gas density (g/cc) from a measured bulk sigma."""
    return sigma_cu / mass_capture(weight_fracs)


# ---------------------------------------------- real-gas law ------------

def gas_density(P_psi, T_K, M_kg_mol=0.016, z=0.9):
    """Real-gas density (g/cc)  rho = P*M/(z*R*T).  M default = methane."""
    P = np.asarray(P_psi, float) * PSI_TO_PA
    rho_si = P * M_kg_mol / (z * R_GAS * T_K)        # kg/m^3
    return rho_si / 1000.0                            # g/cc


def pressure_from_density(rho_gcc, T_K, M_kg_mol=0.016, z=0.9):
    """Invert the real-gas law for pressure (psi) from gas density."""
    rho_si = np.asarray(rho_gcc, float) * 1000.0
    P = rho_si * z * R_GAS * T_K / M_kg_mol
    return P / PSI_TO_PA


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Pulsed-Neutron Gas Pressure Through Casing")
    print("=" * 60)

    # Methane composition (CH4): w_H = 4*1.008/16.04, w_C = 12.01/16.04
    M = 16.04
    methane = {"H": 4 * 1.008 / M, "C": 12.011 / M}

    # The bulk gas sigma is proportional to gas density (Eq. 1)
    s_lo = gas_sigma(0.05, methane)
    s_hi = gas_sigma(0.15, methane)
    print(f"  gas sigma @0.05/0.15 g/cc = {s_lo:.2f} / {s_hi:.2f} c.u.")
    assert abs(s_hi / s_lo - 3.0) < 1e-9              # strictly linear

    # Forward: plant the case-study pressure, get density, then the measured
    # sigma; invert sigma -> density -> pressure and recover ~2,785 psi.
    P_true = 2785.0
    T = 350.0                                         # ~77 C Triassic sandstone
    rho = gas_density(P_true, T)
    sigma_meas = gas_sigma(rho, methane)
    print(f"  gas density            = {rho:.4f} g/cc")
    print(f"  measured sigma         = {sigma_meas:.2f} c.u.")

    rho_inv = gas_density_from_sigma(sigma_meas, methane)
    P_inv = pressure_from_density(rho_inv, T)
    print(f"  recovered pressure     = {P_inv:.0f} psi  (true {P_true:.0f})")
    assert abs(P_inv - P_true) < 1.0

    # Sanity: a lower measured sigma implies lower pressure
    P_low = pressure_from_density(gas_density_from_sigma(sigma_meas * 0.8, methane), T)
    assert P_low < P_inv
    print("  PASS")
    return {"gas_density": float(rho), "sigma": float(sigma_meas),
            "pressure": float(P_inv)}


if __name__ == "__main__":
    test_all()
