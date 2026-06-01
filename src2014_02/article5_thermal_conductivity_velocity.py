"""
Article 5: Thermal Conductivity Estimation From Elastic-Wave Velocity -
           Application of a Petrographic-Coded Model
Nina Gegenhuber, Jurgen Schon (2014)
Reference: Petrophysics Vol. 55, No. 1 (February 2014), pp. 51-56
DOI: none assigned (this issue predates SPWLA DOI assignment)

A two-step petrographic-coded model links thermal conductivity to P-wave
velocity through a shared dependence on porosity/cracking.  The solid host
properties are mixed with Hill's average; ellipsoidal inclusions then reduce the
elastic moduli (Budiansky-O'Connell self-consistent cracks) and the thermal
conductivity (Clausius-Mossotti with depolarization), the inclusions being
controlled by the ratio porosity/aspect-ratio.

Implements:

  - Voigt and Reuss bounds and the Hill average of them (Step-1 solid matrix)
  - Best-fit inclusion aspect ratios by petrographic code (Step-2 calibration)
  - Crack density from crack count and radius  eps = (N/V)*r^3  (Eq. 3)
  - Crack density from porosity and aspect ratio  eps = phi/((4/3)*pi*alpha) (Eq. 4)
  - Budiansky-O'Connell self-consistent cracked moduli and P-wave velocity
  - Sen plate-like depolarization exponents  Lc = 1 - (pi/2)*alpha;  La=Lb=(1-Lc)/2
  - The triaxial depolarization function Rmi (a function of La, Lb, Lc; Eqs. 6-7)
  - Clausius-Mossotti (Maxwell-Garnett) effective thermal conductivity
  - The practical Vp -> lambda regression (Table 4) and the paper's <15% /
    0.5 W/m/K prediction-accuracy criterion

Note: this issue's PDF dropped Eqs. 1-9 in extraction; the forms are
reconstructed from the named source theories (Hill, 1952; Budiansky & O'Connell,
1976; Sen, 1981; Berryman, 1995).  Best-fit aspect ratios: granite/gneiss/
sandstone alpha = 0.20, basic magmatic alpha = 0.25.  Moduli in GPa, velocity in
m/s, thermal conductivity in W/m/K.
"""

import numpy as np

# Best-fit inclusion aspect ratios by petrographic code (Gegenhuber & Schon,
# 2014): granite/gneiss and sandstone alpha = 0.20 (axis ratio 1:5), basic
# magmatic rocks alpha = 0.25 (axis ratio 1:4).
ASPECT_RATIO_BY_ROCK = {
    "granite": 0.20,
    "gneiss": 0.20,
    "sandstone": 0.20,
    "basic_magmatic": 0.25,
}

# Typical porosities by petrographic code (Gegenhuber & Schon, 2014): tight
# crystalline rocks ~3%, sandstone ~20%, basic magmatic rocks ~5-15% (midpoint).
TYPICAL_POROSITY_BY_ROCK = {
    "granite": 0.03,
    "gneiss": 0.03,
    "sandstone": 0.20,
    "basic_magmatic": 0.10,
}


# ---------------------------------------------- mixing --------------

def voigt_bound(fractions, moduli):
    """Voigt (iso-strain) upper bound for a mineral mixture  M_V = sum(f_i*M_i)."""
    f = np.asarray(fractions, float)
    m = np.asarray(moduli, float)
    return float(np.sum(f * m))


def reuss_bound(fractions, moduli):
    """Reuss (iso-stress) lower bound for a mineral mixture  1/M_R = sum(f_i/M_i)."""
    f = np.asarray(fractions, float)
    m = np.asarray(moduli, float)
    return float(1.0 / np.sum(f / m))


def hill_average(voigt, reuss):
    """Hill average of the Voigt and Reuss bounds  M = (M_Voigt + M_Reuss)/2.

    The paper's Step 1 builds the dense solid-host moduli (and conductivity) of a
    multi-mineral rock with Hill's average (Hill, 1952) of the ``voigt_bound`` and
    ``reuss_bound`` of its mineral constituents.
    """
    return 0.5 * (voigt + reuss)


def aspect_ratio_for_rock(rock_type):
    """Best-fit inclusion aspect ratio for a petrographic code (the paper's
    Step-2 calibration): granite/gneiss/sandstone 0.20, basic magmatic 0.25."""
    return ASPECT_RATIO_BY_ROCK[rock_type]


# ---------------------------------------------- crack density --------------

def crack_density_from_count(n_per_volume, crack_radius):
    """Crack density parameter from its primary definition (Eq. 3)

        eps = (N/V)*r^3,

    the number of cracks N per unit volume V times the crack radius r cubed
    (Budiansky & O'Connell, 1976).  Eq. 4 re-expresses this same eps through the
    porosity and aspect ratio (see ``crack_density``).
    """
    return n_per_volume * crack_radius ** 3


def crack_density(phi, aspect_ratio):
    """Crack density parameter from porosity and aspect ratio (Eq. 4)

        eps = phi/((4/3)*pi*alpha),

    so the inclusion effect is governed by the ratio phi/alpha.  This is the
    porosity form of the crack density defined by count in Eq. 3
    (``crack_density_from_count``).
    """
    return phi / (4.0 / 3.0 * np.pi * aspect_ratio)


def crack_porosity(crack_density_eps, aspect_ratio):
    """Crack porosity from the crack density and aspect ratio (inverse of Eq. 4)

        phi = (4/3)*pi*alpha*eps.
    """
    return 4.0 / 3.0 * np.pi * aspect_ratio * crack_density_eps


# ---------------------------------------------- Budiansky-O'Connell --------------

def cracked_moduli(k_solid, mu_solid, nu_solid, crack_density_eps):
    """Budiansky-O'Connell self-consistent cracked bulk and shear moduli
    (Eqs. 1-2, 5)

        K_sc/K_s = 1 - (16/9)*((1-nu^2)/(1-2*nu))*eps,
        mu_sc/mu_s = 1 - (32/45)*((1-nu)(5-nu)/(2-nu))*eps,

    with the effective Poisson's ratio approximated as nu*(1-(16/9)*eps).
    Returns (K_sc, mu_sc).
    """
    nu_sc = nu_solid * (1.0 - 16.0 / 9.0 * crack_density_eps)
    k_sc = k_solid * (1.0 - 16.0 / 9.0 * (1.0 - nu_sc ** 2) / (1.0 - 2.0 * nu_sc)
                      * crack_density_eps)
    mu_sc = mu_solid * (1.0 - 32.0 / 45.0 * (1.0 - nu_sc) * (5.0 - nu_sc)
                        / (2.0 - nu_sc) * crack_density_eps)
    return k_sc, mu_sc


def p_wave_velocity(k, mu, rho):
    """P-wave velocity  Vp = sqrt((K + 4/3 mu)/rho), K, mu in GPa, rho in g/cm^3,
    returning m/s."""
    return np.sqrt((k + 4.0 / 3.0 * mu) * 1e9 / (rho * 1e3))


# ---------------------------------------------- Sen depolarization --------------

def sen_depolarization(aspect_ratio):
    """Sen (1981) plate-like depolarization exponents (Eqs. 8-9)

        Lc = 1 - (pi/2)*alpha,   La = Lb = (1 - Lc)/2,

    for an oblate spheroid (a = b >> c).  Returns (La, Lb, Lc).
    """
    lc = 1.0 - np.pi / 2.0 * aspect_ratio
    la = lb = (1.0 - lc) / 2.0
    return la, lb, lc


# ---------------------------------------------- thermal conductivity --------------

def rmi_depolarization_factor(la, lb, lc, lambda_solid, lambda_incl):
    """Depolarization function Rmi for an ellipsoidal inclusion (Eqs. 6-7).

    The paper's Clausius-Mossotti conductivity is written with an Rmi that is a
    function of the full set of depolarization exponents La, Lb, Lc (not Lc
    alone).  Using the standard orientation-averaged Maxwell-Garnett / Berryman
    (1995) form over the three principal axes,

        Rmi = (1/3)*sum_{j=a,b,c} ls/(ls + L_j*(li - ls)),

    so Rmi = 1 when the inclusion matches the solid (li = ls).  The single-Lc
    ``clausius_mossotti_conductivity`` below is the plate-like-crack limit (Sen,
    1981, where Lc dominates); this function restores the general triaxial form.
    """
    dl = lambda_incl - lambda_solid
    return (1.0 / 3.0) * sum(lambda_solid / (lambda_solid + L * dl)
                             for L in (la, lb, lc))


def clausius_mossotti_conductivity(lambda_solid, lambda_incl, phi, depolarization):
    """Effective thermal conductivity from Clausius-Mossotti / Maxwell-Garnett
    with a depolarization factor L (Eqs. 6-7)

        lambda = lambda_s*[1 + phi*(li - ls)/(ls + L*(1-phi)*(li - ls))],

    reducing to the solid conductivity when phi = 0.
    """
    dl = lambda_incl - lambda_solid
    return lambda_solid * (1.0 + phi * dl
                           / (lambda_solid + depolarization * (1.0 - phi) * dl))


# ---------------------------------------------- velocity -> conductivity --------------

def crack_density_from_velocity(vp, k_solid, mu_solid, nu_solid, rho,
                                eps_max=1.0, n_iter=100, tol=1e-10):
    """Invert the Budiansky-O'Connell elastic model for the crack density that
    reproduces a measured P-wave velocity.

    Solves p_wave_velocity(K_sc(eps), mu_sc(eps), rho) = vp for eps by bisection
    on [0, eps_max].  Because velocity decreases monotonically with crack
    density, the inversion is unique.  Returns the crack density eps.
    """
    lo, hi = 0.0, eps_max
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        k_sc, mu_sc = cracked_moduli(k_solid, mu_solid, nu_solid, mid)
        if p_wave_velocity(k_sc, mu_sc, rho) > vp:
            lo = mid          # too few cracks -> velocity too high
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def thermal_conductivity_from_velocity(vp, aspect_ratio, k_solid, mu_solid,
                                       nu_solid, rho, lambda_solid, lambda_incl):
    """Estimate thermal conductivity from P-wave velocity (the paper's two-step
    petrographic-coded model).

    The shared crack density links the elastic and thermal responses: invert the
    elastic model for the crack density that matches the measured vp, convert it
    to a crack porosity (via the aspect ratio), and feed that porosity through
    the Clausius-Mossotti thermal model with the Sen depolarization factor.
    Returns (lambda_eff, phi, crack_density).
    """
    eps = crack_density_from_velocity(vp, k_solid, mu_solid, nu_solid, rho)
    phi = crack_porosity(eps, aspect_ratio)
    _, _, lc = sen_depolarization(aspect_ratio)
    lam = clausius_mossotti_conductivity(lambda_solid, lambda_incl, phi, lc)
    return lam, phi, eps


def fit_velocity_conductivity_regression(vp_values, aspect_ratio, k_solid, mu_solid,
                                         nu_solid, rho, lambda_solid, lambda_incl,
                                         degree=2):
    """Build the practical Vp -> lambda regression (the paper's Table 4).

    Samples the two-step model over a range of P-wave velocities and least-
    squares-fits a polynomial lambda(Vp), giving the per-rock-type "approximated
    regression equation" the paper tabulates so a sonic/acoustic log can be
    converted to a thermal-conductivity log directly.  Returns (coeffs, predict),
    where ``predict(vp)`` evaluates the fitted polynomial.
    """
    vps = np.asarray(vp_values, float)
    lam = np.array([
        thermal_conductivity_from_velocity(v, aspect_ratio, k_solid, mu_solid,
                                           nu_solid, rho, lambda_solid, lambda_incl)[0]
        for v in vps
    ])
    coeffs = np.polyfit(vps, lam, degree)
    return coeffs, (lambda vp: np.polyval(coeffs, vp))


def prediction_within_tolerance(lambda_pred, lambda_true, rel=0.15, abs_tol=0.5):
    """The paper's stated accuracy criterion for the estimated thermal
    conductivity: within 15% (or within 0.5 W/m/K) of the measured value."""
    return bool(abs(lambda_pred - lambda_true) <= max(rel * lambda_true, abs_tol))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Thermal Conductivity from Elastic Velocity")
    print("=" * 60)

    # Hill average lies between the bounds
    assert hill_average(40.0, 30.0) == 35.0

    # Step-1 mineral mixing: Voigt >= Hill >= Reuss for a quartz/feldspar mix
    fr, ks = [0.6, 0.4], [37.0, 75.0]   # bulk moduli (GPa)
    kv, kr = voigt_bound(fr, ks), reuss_bound(fr, ks)
    kh = hill_average(kv, kr)
    print(f"  mixing bounds: Voigt={kv:.2f}  Hill={kh:.2f}  Reuss={kr:.2f} GPa")
    assert kv > kh > kr
    # Petrographic aspect-ratio codes
    assert aspect_ratio_for_rock("sandstone") == 0.20
    assert aspect_ratio_for_rock("basic_magmatic") == 0.25

    # Crack density grows with porosity and falls with aspect ratio
    eps = crack_density(0.03, aspect_ratio=0.20)
    print(f"  crack density (phi=3%, alpha=0.20) = {eps:.4f}")
    assert eps > 0 and crack_density(0.20, 0.20) > eps

    # The count form (Eq. 3) and porosity form (Eq. 4) are the same parameter:
    # match them by choosing the count that reproduces a given crack porosity
    eps_count = crack_density_from_count(n_per_volume=eps / 0.5 ** 3, crack_radius=0.5)
    print(f"  crack density from count = {eps_count:.4f}")
    assert np.isclose(eps_count, eps)

    # Cracks lower the moduli and the P-wave velocity
    k_s, mu_s, nu_s, rho = 37.0, 44.0, 0.07, 2.65   # quartz-rich granite-like
    k_sc, mu_sc = cracked_moduli(k_s, mu_s, nu_s, eps)
    vp0 = p_wave_velocity(k_s, mu_s, rho)
    vp = p_wave_velocity(k_sc, mu_sc, rho)
    print(f"  Vp(solid)={vp0:.0f}  Vp(cracked)={vp:.0f} m/s")
    assert k_sc < k_s and mu_sc < mu_s and vp < vp0

    # Sen depolarization exponents sum to 1
    la, lb, lc = sen_depolarization(0.20)
    print(f"  depolarization La=Lb={la:.3f}  Lc={lc:.3f}")
    assert np.isclose(la + lb + lc, 1.0) and 0 < lc < 1

    # Triaxial Rmi factor (Eqs. 6-7): unity when the inclusion matches the solid;
    # for low-conductivity pores (li < ls) each axis term ls/(ls+L*(li-ls)) > 1
    assert np.isclose(rmi_depolarization_factor(la, lb, lc, 6.0, 6.0), 1.0)
    rmi = rmi_depolarization_factor(la, lb, lc, 6.0, 0.6)
    print(f"  Rmi(li=0.6, ls=6.0) = {rmi:.3f}")
    assert rmi > 1.0
    assert TYPICAL_POROSITY_BY_ROCK["sandstone"] == 0.20

    # Thermal conductivity: low-conductivity pores (air/water) reduce lambda
    lam = clausius_mossotti_conductivity(lambda_solid=6.0, lambda_incl=0.6,
                                         phi=0.03, depolarization=lc)
    lam0 = clausius_mossotti_conductivity(6.0, 0.6, 0.0, lc)
    print(f"  lambda(phi=0)={lam0:.2f}  lambda(phi=3%)={lam:.2f} W/m/K")
    assert np.isclose(lam0, 6.0) and lam < 6.0
    # higher porosity reduces it further
    assert clausius_mossotti_conductivity(6.0, 0.6, 0.20, lc) < lam

    # Crack density inverts from velocity (round-trip) and crack porosity is
    # the inverse of the crack-density relation
    eps_inv = crack_density_from_velocity(vp, k_s, mu_s, nu_s, rho)
    print(f"  crack density from Vp = {eps_inv:.4f}  (true {eps:.4f})")
    assert np.isclose(eps_inv, eps, atol=1e-4)
    assert np.isclose(crack_porosity(eps, 0.20), 0.03)

    # Two-step model: estimate thermal conductivity straight from the velocity.
    # A lower (more cracked) velocity gives a lower estimated conductivity.
    lam_hi, phi_hi, _ = thermal_conductivity_from_velocity(
        vp0 * 0.98, 0.20, k_s, mu_s, nu_s, rho, lambda_solid=6.0, lambda_incl=0.6)
    lam_lo, phi_lo, _ = thermal_conductivity_from_velocity(
        vp0 * 0.90, 0.20, k_s, mu_s, nu_s, rho, lambda_solid=6.0, lambda_incl=0.6)
    print(f"  lambda(Vp high)={lam_hi:.2f}  lambda(Vp low)={lam_lo:.2f} W/m/K")
    assert lam_lo < lam_hi < 6.0 and phi_lo > phi_hi > 0

    # Table-4 regression: fit lambda(Vp) over a velocity range, then check it
    # reproduces the model within the paper's <15% / 0.5 W/m/K tolerance
    vp_grid = np.linspace(vp0 * 0.85, vp0 * 0.99, 12)
    coeffs, predict = fit_velocity_conductivity_regression(
        vp_grid, 0.20, k_s, mu_s, nu_s, rho, lambda_solid=6.0, lambda_incl=0.6)
    vp_test = vp0 * 0.93
    lam_model = thermal_conductivity_from_velocity(
        vp_test, 0.20, k_s, mu_s, nu_s, rho, lambda_solid=6.0, lambda_incl=0.6)[0]
    lam_reg = float(predict(vp_test))
    print(f"  regression lambda={lam_reg:.2f}  model lambda={lam_model:.2f} W/m/K")
    assert prediction_within_tolerance(lam_reg, lam_model)
    assert not prediction_within_tolerance(2.0, 6.0)  # far-off prediction fails
    print("  PASS")
    return {"crack_density": float(eps), "Vp_cracked": float(vp),
            "Lc": float(lc), "lambda": float(lam),
            "lambda_from_Vp": float(lam_hi)}


if __name__ == "__main__":
    test_all()
