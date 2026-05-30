"""
Article 5: Fracture Characterization Combining Borehole Acoustic Reflection
Imaging and Geomechanical Analyses
Tang, Wang, Li, Xiong, Zhang (2022)
DOI: 10.30632/PJV63N6-2022a5

Implements the 3-D Mohr-Coulomb critically-stressed-fracture analysis and
the SH-wave cross-dipole imaging formula used by the paper:

  - Effective stress tensor                                (Eq. 1)
  - Fracture normal n = (sin t sin a, sin t cos a, cos t) (Eq. 2)
  - Effective normal stress sigma_n on the fracture face   (Eq. 3)
  - Shear stress tau_n                                     (Eq. 4)
  - SH-wave image from 4C cross-dipole data
        SH(alpha) = xx cos^2(a) - sin(a) cos(a) (xy + yx) + yy sin^2(a)
                                                          (Eq. 5)
  - Mohr-Coulomb criticality threshold tau = S_0 + mu sigma_n  (Eq. 6)

Key theoretical claim that we re-derive numerically: the 180-deg cross-dipole
strike ambiguity does NOT change the (tau_n, sigma_n) pair, so the
acoustically-imaged fracture set can be filtered by the geomechanical
criterion without resolving the azimuth ambiguity.
"""

import numpy as np


# ---------------------------------------------- effective stress (Eq. 1) ---

def effective_principal_stresses(sigma_H, sigma_h, sigma_V, p_pore):
    """Return (sigma_H_eff, sigma_h_eff, sigma_V_eff)."""
    return (sigma_H - p_pore, sigma_h - p_pore, sigma_V - p_pore)


# ---------------------------------------------- fracture normal (Eq. 2) ---

def fracture_normal(dip_deg, strike_deg):
    """Unit normal in stress-frame coordinates (theta = dip, alpha = strike)."""
    t = np.deg2rad(dip_deg)
    a = np.deg2rad(strike_deg)
    return np.array([np.sin(t) * np.sin(a),
                     np.sin(t) * np.cos(a),
                     np.cos(t)])


# ---------------------------------------------- traction on fracture face --

def traction(sigma_diag, n):
    """Traction vector T = Sigma n for a diagonal stress tensor."""
    return np.asarray(sigma_diag) * n


def normal_and_shear(sigma_diag, n):
    """sigma_n = n . T (Eq. 3),  tau_n = || T - sigma_n n || (Eq. 4)."""
    T = traction(sigma_diag, n)
    sigma_n = float(np.dot(n, T))
    tau_vec = T - sigma_n * n
    tau_n = float(np.linalg.norm(tau_vec))
    return sigma_n, tau_n


# ---------------------------------------------- Mohr-Coulomb (Eq. 6) -----

def critically_stressed(sigma_n, tau_n, friction_mu=0.6, cohesion_S0=0.0):
    """True if shear traction reaches the Mohr-Coulomb envelope."""
    return tau_n >= (cohesion_S0 + friction_mu * sigma_n)


# ---------------------------------------------- SH cross-dipole image (Eq. 5)

def sh_image(xx, xy, yx, yy, alpha_deg):
    """SH(alpha) for a 4-component dipole-reflection waveform set.

    Inputs are the four xx / xy / yx / yy time series at a given depth and
    delay; alpha is the angle (deg) from the inline (x) dipole direction.
    """
    a = np.deg2rad(alpha_deg)
    return xx * np.cos(a) ** 2 \
           - np.sin(a) * np.cos(a) * (xy + yx) \
           + yy * np.sin(a) ** 2


def best_strike(xx, xy, yx, yy, n_alphas=180):
    """Find the alpha that maximises peak |SH| - returns (alpha_deg, peak)."""
    alphas = np.linspace(0, 180, n_alphas, endpoint=False)
    peaks = [float(np.max(np.abs(sh_image(xx, xy, yx, yy, a)))) for a in alphas]
    i = int(np.argmax(peaks))
    return float(alphas[i]), peaks[i]


# ---------------------------------------------- tests --------------------

def test_all():
    print("=" * 60)
    print("Article 5: Dipole-Shear Imaging + Mohr-Coulomb Filter")
    print("=" * 60)

    # Stress state (MPa) - strike-slip regime
    sigma_H, sigma_h, sigma_V = 70.0, 35.0, 50.0
    p_pore = 15.0
    sigma_eff = effective_principal_stresses(sigma_H, sigma_h, sigma_V, p_pore)
    print(f"  Effective principal stresses (H, h, V) = "
          f"{sigma_eff[0]:.1f}, {sigma_eff[1]:.1f}, {sigma_eff[2]:.1f}  MPa")

    # Random fracture population
    rng = np.random.default_rng(0)
    n_frac = 400
    dips = rng.uniform(30.0, 85.0, n_frac)
    strikes = rng.uniform(0.0, 360.0, n_frac)
    crit = np.zeros(n_frac, dtype=bool)
    sn_arr = np.zeros(n_frac)
    tn_arr = np.zeros(n_frac)
    for i in range(n_frac):
        n = fracture_normal(dips[i], strikes[i])
        sn, tn = normal_and_shear(sigma_eff, n)
        sn_arr[i] = sn
        tn_arr[i] = tn
        crit[i] = critically_stressed(sn, tn, friction_mu=0.6)

    frac_crit = float(crit.mean())
    print(f"  Critically stressed fraction (mu=0.6) = {frac_crit:.3f}")

    # 180-deg-strike-ambiguity check: same fracture, strike + 180 should give
    # the same (sigma_n, tau_n)
    n_a = fracture_normal(60.0, 110.0)
    n_b = fracture_normal(60.0, 110.0 + 180.0)
    sn_a, tn_a = normal_and_shear(sigma_eff, n_a)
    sn_b, tn_b = normal_and_shear(sigma_eff, n_b)
    print(f"  Strike  +110 deg:   sigma_n={sn_a:.2f}  tau_n={tn_a:.2f}")
    print(f"  Strike  +290 deg:   sigma_n={sn_b:.2f}  tau_n={tn_b:.2f}")
    assert np.isclose(sn_a, sn_b) and np.isclose(tn_a, tn_b), \
        "180-deg ambiguity must not change (sigma_n, tau_n)"

    # SH-image best-strike demo on a synthetic single-fracture 4C waveform
    t = np.linspace(0, 1, 401)
    pulse = np.exp(-((t - 0.4) / 0.02) ** 2) * np.sin(2 * np.pi * 18 * t)
    true_strike = 35.0
    a = np.deg2rad(true_strike)
    xx = pulse * (np.cos(a) ** 2)
    yy = pulse * (np.sin(a) ** 2)
    xy = pulse * (-np.sin(a) * np.cos(a))
    yx = xy.copy()
    est, peak = best_strike(xx, xy, yx, yy)
    err = min(abs(est - true_strike), abs(est - true_strike - 180.0))
    print(f"  True dipole strike  = {true_strike:.1f} deg")
    print(f"  Recovered strike    = {est:.1f} deg  (err = {err:.1f} deg)")
    assert err < 2.0, "SH-image strike recovery should be sub-degree"
    print("  PASS")
    return {"frac_critical": frac_crit,
            "ambig_sigma_n_diff": float(abs(sn_a - sn_b)),
            "strike_err_deg": float(err)}


if __name__ == "__main__":
    test_all()
