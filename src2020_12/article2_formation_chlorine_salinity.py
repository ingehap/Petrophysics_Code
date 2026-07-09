"""
Article 2: Formation Chlorine Measurement From Spectroscopy Enables Water
           Salinity Interpretation: Theory, Modeling, and Applications
Miles, Mosse, Grau (2020)
DOI: 10.30632/PJV61N6-2020a2

Pulsed-neutron capture spectroscopy measures a total chlorine yield that mixes a
formation and a borehole contribution.  A chlorine-difference standard (CYDCL)
and an environment factor Phi(env) = 1/f isolate the borehole part; subtracting
it gives the formation dry-weight chlorine (DWCL), which converts to NaCl-
equivalent water salinity, bulk volume water, and water saturation.  A
sigma-consistency model bounds the answer.

Implements:

  - Yields-to-weights  W_i = FY2W * S_i * Y_i                 (Eq. 6)
  - Chlorine yield split  Y_Cl = Y_form + Y_borehole          (Eq. 1)
  - CYDCL borehole standard  CYDCL = f * Y_borehole, 1/f      (Eqs. 4-5)
  - Borehole subtraction  Y_form = Y_total - Y_borehole       (Eqs. 8-9)
  - DWCL -> salinity (molar-mass ratio 1.649), BVW, Sw        (Eqs. 11-14)
  - Macroscopic sigma mixing and Sigma_max                    (Eqs. 19-20)

Note: this issue's PDF text layer preserves equation numbers and variable
definitions but drops the typeset glyphs, so these are faithful standard-form
reconstructions.  Constants are the paper's: molar-mass ratio M_NaCl/M_Cl =
1.649, macroscopic absorption ~567 c.u. per (g/cc) of chlorine, low-GOR
hydrocarbon / fresh-water sigma 22 c.u., shale sigma 29.4 c.u.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

MOLAR_RATIO_NACL_CL = 1.649      # M_NaCl / M_Cl = 58.44 / 35.45
CL_FRAC_OF_NACL = 1.0 / MOLAR_RATIO_NACL_CL   # = 0.6064
SIGMA_PER_G_CL = 567.0           # c.u. per (g/cc) of chlorine
SIGMA_FLUID = 22.0               # c.u., low-GOR HC / fresh water
SIGMA_SHALE = 29.4               # c.u.


# ---------------------------------------------- yields <-> weights ------

def yield_to_weight(fy2w, S, Y):
    """Dry-weight element from yield  W = FY2W * S * Y  (Eq. 6)."""
    return petrolib.nuclear.yields_to_weights(fy2w, S, Y)


def weight_to_yield(fy2w, S, W):
    """Inverse of Eq. 6: expected yield from a known weight  Y = W/(FY2W*S)."""
    return petrolib.nuclear.weights_to_yields(fy2w, S, W)


# ---------------------------------------------- borehole subtraction ----

def borehole_yield_from_cydcl(cydcl, phi_env):
    """Total borehole chlorine yield  Y_bh = CYDCL * Phi(env)  (Eqs. 4-5).

    Phi(env) = 1/f is the calibrated inverse fraction; CYDCL is the measured
    chlorine-difference standard.
    """
    return cydcl * phi_env


def corrected_formation_yield(y_total, y_borehole):
    """Formation chlorine yield  Y_form = Y_total - Y_borehole  (Eqs. 8-9)."""
    return y_total - y_borehole


# ---------------------------------------------- DWCL -> salinity --------

def salinity_from_dwcl(dwcl, rhob, phi_w, rho_w):
    """NaCl-equivalent water salinity (weight fraction) from DWCL  (Eq. 11).

    Chlorine mass per bulk = DWCL*rhob comes from NaCl in the water
    (water mass per bulk = phi_w*rho_w); Sal = (Cl-in-water) * 1.649.
    """
    cl_in_water = dwcl * rhob / (phi_w * rho_w)
    return cl_in_water * MOLAR_RATIO_NACL_CL


def bvw_from_dwcl(dwcl, rhob, sal_w, rho_w):
    """Bulk volume water from DWCL and assumed salinity  (Eq. 12)."""
    return dwcl * rhob * MOLAR_RATIO_NACL_CL / (sal_w * rho_w)


def water_saturation(bvw, phi_total):
    """Water saturation  Sw = BVW / phi_total  (Eq. 14)."""
    return bvw / phi_total


# ---------------------------------------------- sigma model -------------

def sigma_mixture(phi, sigma_matrix, sigma_fluid=SIGMA_FLUID, cl_mass_g_cc=0.0):
    """Macroscopic capture cross section by volume mixing  (Eq. 19).

        Sigma = (1-phi)*Sigma_ma + phi*Sigma_fluid + 567*(g/cc Cl)
    """
    return (1.0 - phi) * sigma_matrix + phi * sigma_fluid \
        + SIGMA_PER_G_CL * cl_mass_g_cc


def sigma_max(phi, sigma_matrix, dwcl, rhob):
    """Maximum sigma: matrix + 22-c.u. pore fluid + measured chlorine  (Eq. 20)."""
    return sigma_mixture(phi, sigma_matrix, SIGMA_FLUID, dwcl * rhob)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Formation Chlorine -> Water Salinity")
    print("=" * 60)

    # Yields-to-weights round-trips through Eq. 6 / its inverse
    fy2w, S = 1.4, 0.85
    W = yield_to_weight(fy2w, S, 0.12)
    assert abs(weight_to_yield(fy2w, S, W) - 0.12) < 1e-12

    # Borehole subtraction recovers a planted formation chlorine yield
    y_form_true, y_bh_true = 0.08, 0.05
    y_total = y_form_true + y_bh_true                     # Eq. 1
    f = 0.40
    cydcl = f * y_bh_true                                 # Eq. 4
    phi_env = 1.0 / f                                     # Eq. 5
    y_bh = borehole_yield_from_cydcl(cydcl, phi_env)
    y_form = corrected_formation_yield(y_total, y_bh)
    print(f"  recovered Y_form       = {y_form:.4f}  (true {y_form_true})")
    assert abs(y_form - y_form_true) < 1e-9

    # DWCL -> salinity and back through BVW is self-consistent
    dwcl, rhob, phi_w, rho_w = 0.004, 2.45, 0.20, 1.05
    sal = salinity_from_dwcl(dwcl, rhob, phi_w, rho_w)
    print(f"  water salinity         = {sal*1000:.1f} ppk NaCl")
    bvw = bvw_from_dwcl(dwcl, rhob, sal, rho_w)
    print(f"  BVW (round-trip)       = {bvw:.4f}  (phi_w {phi_w})")
    assert abs(bvw - phi_w) < 1e-9                        # Eqs. 11 <-> 12 consistent
    sw = water_saturation(bvw, 0.25)
    assert 0.0 < sw <= 1.0

    # Higher salinity -> more chlorine for the same water volume
    assert salinity_from_dwcl(0.008, rhob, phi_w, rho_w) > sal

    # Sigma: chlorine and shaliness both raise the macroscopic cross section;
    # clean fresh-water sand sits near the 22 c.u. fluid value
    sig_clean = sigma_mixture(0.25, sigma_matrix=8.0)
    sig_salty = sigma_mixture(0.25, sigma_matrix=8.0, cl_mass_g_cc=0.01)
    print(f"  sigma clean / salty    = {sig_clean:.1f} / {sig_salty:.1f} c.u.")
    assert sig_salty > sig_clean
    assert sigma_max(0.25, 8.0, dwcl, rhob) > sig_clean
    print("  PASS")
    return {"y_form": y_form, "salinity_ppk": sal * 1000,
            "sw": sw, "sigma_salty": sig_salty}


if __name__ == "__main__":
    test_all()
