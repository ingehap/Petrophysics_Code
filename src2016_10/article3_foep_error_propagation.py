"""
Article 3: On Error Calculation and Use of First-Order Error Propagation as
           Integral Part of Petrophysical Calculation
Stalheim (2016)
Reference: Petrophysics Vol. 57, No. 5 (October 2016), pp. 465-478
DOI: none assigned (this issue predates SPWLA DOI assignment)

First-order error propagation (FOEP) is presented in matrix form as a practical,
analytical alternative to Monte Carlo for carrying error through petrophysical
calculations.  The error in a result f is sigma_f = sqrt(c' Sigma c), where c is
the vector of partial derivatives of f and Sigma is the variance-covariance
matrix of the inputs; the relative contribution of each input follows from the
same terms.  The method is demonstrated on the density-porosity and Archie
equations.

Implements:

  - FOEP error  sigma_f = sqrt(c' Sigma c)  (Eq. 3)
  - Variance-covariance matrix from standard deviations + correlations (Eqs. 8, 14)
  - Relative contribution of each input error (Eq. 13)
  - Numerical Jacobian (partial-derivative vector c) for nonlinear f (Eq. 12)
  - Petrophysical functions: density porosity, Archie Sw, Vsh, effective porosity
    (Eqs. 15-18) with analytic FOEP demonstrations

Note: this issue's PDF has a text layer; the FOEP relations (Eqs. 1-14) and the
petrophysical functions (Eqs. 15-18) are transcribed from the body, while the
typeset matrix glyphs were dropped and reconstructed in standard form.
"""

import numpy as np


# ---------------------------------------------- FOEP core --------------

def covariance_matrix(stds, corr=None):
    """Variance-covariance matrix from input standard deviations and an optional
    correlation matrix (Eqs. 8, 14)

        Sigma_ij = xi_ij * std_i * std_j,

    with xi_ii = 1.  corr defaults to the identity (independent variables).
    """
    stds = np.asarray(stds, float)
    n = stds.size
    if corr is None:
        corr = np.eye(n)
    corr = np.asarray(corr, float)
    return corr * np.outer(stds, stds)


def foep_error(c, sigma):
    """First-order propagated error  sigma_f = sqrt(c' Sigma c)  (Eq. 3)."""
    c = np.asarray(c, float)
    sigma = np.asarray(sigma, float)
    return float(np.sqrt(c @ sigma @ c))


def relative_contributions(c, sigma):
    """Relative contribution of each input error to the variance (Eq. 13)

        fraction_i = (c_i * (Sigma c)_i) / (c' Sigma c),

    summing to 1.  For independent inputs this reduces to (c_i*std_i)^2/var_f.
    """
    c = np.asarray(c, float)
    sigma = np.asarray(sigma, float)
    var = c @ sigma @ c
    return c * (sigma @ c) / var


def numerical_jacobian(func, x, eps=1e-6):
    """Partial-derivative vector c = [df/dx_i] of a scalar function (Eq. 12),
    by central finite differences (for nonlinear f, e.g. Archie)."""
    x = np.asarray(x, float)
    c = np.zeros_like(x)
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps * max(abs(x[i]), 1.0)
        c[i] = (func(x + dx) - func(x - dx)) / (2.0 * dx[i])
    return c


# ---------------------------------------------- petrophysics --------------

def density_porosity(rho_b, rho_ma, rho_fl):
    """Total porosity from the density log  phi = (rho_ma - rho_b)/(rho_ma - rho_fl)
    (Eq. 15)."""
    return (rho_ma - rho_b) / (rho_ma - rho_fl)


def density_porosity_jacobian(rho_b, rho_ma, rho_fl):
    """Analytic partial-derivative vector of Eq. 15 w.r.t. (rho_b, rho_ma, rho_fl)
    (Eq. 20)."""
    den = rho_ma - rho_fl
    d_rhob = -1.0 / den
    d_rhoma = (rho_b - rho_fl) / den ** 2
    d_rhofl = (rho_ma - rho_b) / den ** 2
    return np.array([d_rhob, d_rhoma, d_rhofl])


def archie_sw(rt, rw, phi, m=2.0, n=2.0):
    """Archie water saturation  Sw = (Rw/(phi^m * Rt))^(1/n)  (Eq. 16)."""
    return (rw / (phi ** m * rt)) ** (1.0 / n)


def vsh_gr(gr, gr_sand, gr_shale):
    """Shale volume from the gamma-ray log (Eq. 17)

        Vsh = (GR - GR_sand)/(GR_shale - GR_sand).
    """
    return (gr - gr_sand) / (gr_shale - gr_sand)


def effective_porosity(phi_t, vsh, phi_sh):
    """Effective porosity  phi_e = phi_t - Vsh*phi_sh  (Eq. 18)."""
    return phi_t - vsh * phi_sh


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: First-Order Error Propagation (FOEP)")
    print("=" * 60)

    # Density porosity and its analytic vs numerical Jacobian agree
    rho_b, rho_ma, rho_fl = 2.45, 2.65, 1.00
    phi = density_porosity(rho_b, rho_ma, rho_fl)
    c_phi = density_porosity_jacobian(rho_b, rho_ma, rho_fl)
    c_num = numerical_jacobian(lambda x: density_porosity(*x), [rho_b, rho_ma, rho_fl])
    print(f"  porosity               = {phi:.4f}")
    assert np.allclose(c_phi, c_num, rtol=1e-4)

    # FOEP porosity error from independent input standard deviations
    sig_phi = covariance_matrix([0.03, 0.02, 0.05])   # std of rho_b, rho_ma, rho_fl
    dphi = foep_error(c_phi, sig_phi)
    print(f"  porosity error +/-     = {dphi:.4f}")
    assert dphi > 0

    # Relative contributions sum to one
    contrib = relative_contributions(c_phi, sig_phi)
    print(f"  contributions          = {np.round(contrib, 3)}")
    assert np.isclose(contrib.sum(), 1.0) and np.all(contrib >= 0)

    # Correlated inputs change the propagated error vs the independent case
    corr = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
    sig_corr = covariance_matrix([0.03, 0.02, 0.05], corr)
    assert not np.isclose(foep_error(c_phi, sig_corr), dphi)

    # FOEP on Archie's (nonlinear) Sw using a numerical Jacobian
    x0 = np.array([20.0, 0.05, phi, 2.0, 2.0])   # Rt, Rw, phi, m, n
    sw = archie_sw(*x0)
    c_sw = numerical_jacobian(lambda x: archie_sw(*x), x0)
    sig_sw = covariance_matrix([2.0, 0.005, dphi, 0.1, 0.1])
    dsw = foep_error(c_sw, sig_sw)
    print(f"  Archie Sw / error      = {sw:.3f} +/- {dsw:.3f}")
    assert 0 < sw <= 1 and dsw > 0

    # Vsh and effective porosity behave as defined
    vsh = vsh_gr(75.0, 30.0, 120.0)
    assert np.isclose(vsh, 0.5)
    assert np.isclose(effective_porosity(0.20, vsh, 0.10), 0.15)
    print("  PASS")
    return {"phi": float(phi), "dphi": dphi, "Sw": float(sw), "dSw": dsw}


if __name__ == "__main__":
    test_all()
