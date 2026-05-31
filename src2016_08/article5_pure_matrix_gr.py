"""
Article 5: Pure Matrix GR, an Indicator of Rock Matrix Gamma Radioactivity and
           its Applications
Wang, Zhao (2016)
Reference: Petrophysics Vol. 57, No. 4 (August 2016), pp. 390-396
DOI: none assigned (this issue predates SPWLA DOI assignment)

The gamma-ray log responds not only to lithology but also to porosity, bulk
density and background radiation, which distort it in high-porosity formations.
A new quantity, pure matrix GR (Nm), isolates the radioactivity of the rock
matrix and is, by construction, immune to porosity and bulk density.  Nm is
obtained from the GR, density and porosity logs after removing a (constant)
background by subtracting a low reference reading; converting Nm back to a
density-weighted "matrix GR" allows direct comparison with the original log.

Implements:

  - Homogeneous-formation gamma flux  Psi = n/(rho*mu)  (Eqs. 1-2)
  - Matrix/fluid radioactivity split  n = (1-phi)*nm + phi*nw  (Eq. 3)
  - GR log model  gamma = ((1-phi)*Nm + phi*Nw)/rho + gamma_b  (Eq. 5)
  - Pure matrix GR  Nm = rho*(gamma - gamma0)/(1-phi)  (Eq. 10)
  - Relative error from neglecting fluid radioactivity (Eq. 11)
  - Matrix GR  GRm = Nm/rho_m + gamma_b  (Eq. 12)

Note: this issue's PDF has a text layer; Eqs. 1-12 are transcribed from the
body (variable definitions survived), while the typeset glyphs were dropped and
the integrals/relations are reconstructed in standard form.  GR in GAPI, density
in g/cm^3, porosity as a fraction; Nm in GAPI*g/cm^3.
"""

import numpy as np


# ---------------------------------------------- flux / radioactivity --------------

def gamma_flux(n, mu, rho):
    """Total gamma flux from an infinite homogeneous formation (Eqs. 1-2)

        Psi = n * integral_0^inf exp(-mu*rho*r) dr = n/(rho*mu),

    so the GR counting rate is a direct measure of n/rho.
    """
    return n / (rho * mu)


def formation_radioactivity(nm, nw, phi):
    """Bulk radioactivity per unit volume  n = (1-phi)*nm + phi*nw  (Eq. 3)."""
    return (1.0 - phi) * nm + phi * nw


def gamma_log(nm_gapi, nw_gapi, phi, rho, gamma_b=0.0):
    """Modelled GR reading in GAPI (Eq. 5)

        gamma = ((1-phi)*Nm + phi*Nw)/rho + gamma_b,

    with Nm = nm*k/mu the pure matrix GR and Nw = nw*k/mu the fluid term.
    """
    return ((1.0 - phi) * nm_gapi + phi * nw_gapi) / rho + gamma_b


# ---------------------------------------------- pure matrix GR --------------

def pure_matrix_gr(gamma, gamma0, phi, rho):
    """Pure matrix GR (Eq. 10), with fluid radioactivity neglected

        Nm = rho*(gamma - gamma0)/(1 - phi),

    where gamma0 is a low reference reading close to the background gamma_b
    (e.g. the GR of clean sandstone/carbonate, or the log minimum).  Nm is
    immune to porosity and bulk density.
    """
    return rho * (np.asarray(gamma, float) - gamma0) / (1.0 - phi)


def matrix_gr(nm, rho_m, gamma_b=0.0):
    """Matrix GR for comparison with the original log (Eq. 12)

        GRm = Nm/rho_m + gamma_b,

    the GR a pure-matrix (phi = 0) formation of density rho_m would read.
    """
    return nm / rho_m + gamma_b


def nm_relative_error(nm, nw, phi):
    """Relative error in Nm from neglecting fluid radioactivity Nw (Eq. 11)

        error = phi*Nw / ((1-phi)*Nm),

    the fractional overestimate of Nm when Nw is assumed zero.
    """
    return phi * nw / ((1.0 - phi) * nm)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Pure Matrix GR")
    print("=" * 60)

    # Gamma flux is proportional to n/rho
    assert np.isclose(gamma_flux(10.0, 0.05, 2.0), 10.0 / (2.0 * 0.05))

    # Pure matrix GR is independent of porosity and bulk density:
    # two formations with the same matrix but different phi/rho give the same Nm.
    nm_true, gamma_b = 120.0, 10.0     # GAPI*g/cm^3, GAPI
    rho_m, rho_w = 2.65, 1.0
    for phi in (0.1, 0.4, 0.7):
        rho = (1.0 - phi) * rho_m + phi * rho_w
        gr = gamma_log(nm_true, 0.0, phi, rho, gamma_b)        # Nw = 0
        nm = pure_matrix_gr(gr, gamma_b, phi, rho)
        assert np.isclose(nm, nm_true)
    print(f"  Nm recovered (phi=0.1..0.7) = {nm_true:.1f} GAPI*g/cm^3")

    # Denser/low-porosity rock reads higher raw GR for the same matrix
    gr_lo = gamma_log(nm_true, 0.0, 0.1, (1 - 0.1) * rho_m + 0.1 * rho_w, gamma_b)
    gr_hi = gamma_log(nm_true, 0.0, 0.7, (1 - 0.7) * rho_m + 0.7 * rho_w, gamma_b)
    print(f"  raw GR phi=0.1 / 0.7   = {gr_lo:.1f} / {gr_hi:.1f} GAPI")
    assert gr_lo > gr_hi

    # Matrix GR converts Nm to a comparable GR scale
    grm = matrix_gr(nm_true, rho_m, gamma_b)
    print(f"  matrix GR              = {grm:.1f} GAPI")
    assert grm > gamma_b

    # Neglecting Nw overestimates Nm; the error grows with porosity
    e_lo = nm_relative_error(nm_true, nw=15.0, phi=0.2)
    e_hi = nm_relative_error(nm_true, nw=15.0, phi=0.6)
    print(f"  Nm rel. error phi=0.2/0.6 = {e_lo*100:.1f}% / {e_hi*100:.1f}%")
    assert 0 < e_lo < e_hi
    print("  PASS")
    return {"Nm": nm_true, "GRm": float(grm), "err_hi": float(e_hi)}


if __name__ == "__main__":
    test_all()
