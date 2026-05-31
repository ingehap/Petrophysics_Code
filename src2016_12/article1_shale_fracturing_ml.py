"""
Article 1: Shale Fracturing Characterization and Optimization by Using
           Anisotropic Acoustic Interpretation, 3D Fracture Modeling, and
           Supervised Machine Learning
Gu, Gokaraju, Chen, Quirein (2016)
Reference: Petrophysics Vol. 57, No. 6 (December 2016), pp. 573-587
DOI: none assigned (this issue predates SPWLA DOI assignment)

A workflow links anisotropic acoustic interpretation to 3D fracture modeling and
a neural-network surrogate.  The VTI stiffness tensor is closed with the ANNIE
(and modified-ANNIE) relations, the anisotropic moduli are condensed into an
equivalent isotropic Young's modulus for the fracture model, and a
return-on-fracturing-investment objective drives the optimization.

Implements:

  - ANNIE stiffness closure  C11 = 2*(C66 - C44) + C33,  C13 = C11 - 2*C66
  - Modified-ANNIE closure with calibration constants k, k'
  - VTI engineering moduli (Eh, Ev, vh, vv, Gvh, Ghh) from the stiffness tensor
  - Closure (minimum horizontal) stress from the calibrated moduli (Eq. 10)
  - Sneddon-Berry fracture width and the condensed elasticity term f (Eqs. 11, 16)
  - Equivalent isotropic Young's modulus and Poisson's ratio (Eqs. 21-22)
  - Return on fracturing investment (ROFI)

Note: this issue's PDF has a text layer; the ANNIE relations (Eqs. 1-5), the
closure stress (Eq. 10), Sneddon-Berry width (Eq. 11) and the equivalent-modulus
relations (Eqs. 21-22) survived in the body text, while the VTI-moduli formulas
(Eqs. 6-9) and Chertov anisotropic width (Eqs. 12-15) lost their typeset glyphs
and are reconstructed in standard VTI form (Higgins et al., 2008).  These VTI
moduli reduce to the isotropic case when C33 = C11 and C44 = C66.
Stiffnesses/moduli in consistent units (e.g. GPa); stresses/strains as noted.
"""

import numpy as np


# ---------------------------------------------- ANNIE closure --------------

def annie_c11(c33, c44, c66):
    """ANNIE  C11 = 2*(C66 - C44) + C33  (Eq. 3)."""
    return 2.0 * (c66 - c44) + c33


def annie_c13(c11, c66):
    """ANNIE  C13 = C11 - 2*C66  (Eq. 2); equivalently C13 = C33 - 2*C44."""
    return c11 - 2.0 * c66


def mannie_c13(c11, c66, k):
    """Modified-ANNIE  C13 = k*(C11 - 2*C66)  (Eq. 4); k from core data (k=1 -> ANNIE)."""
    return k * (c11 - 2.0 * c66)


def mannie_c11(c33, c44, c66, k_prime):
    """Modified-ANNIE  C11 = k'*(2*(C66 - C44)) + C33  (Eq. 5); k'=1 -> ANNIE."""
    return k_prime * (2.0 * (c66 - c44)) + c33


# ---------------------------------------------- VTI engineering moduli --------------

def vti_elastic_moduli(c11, c33, c44, c66, c12, c13):
    """Horizontal/vertical Young's moduli and Poisson's ratios from the VTI
    stiffness tensor (Eqs. 6-9, Higgins et al., 2008)

        Ev = C33 - 2*C13^2/(C11 + C12)
        Eh = (C11 - C12)*(C11*C33 - 2*C13^2 + C12*C33)/(C11*C33 - C13^2)
        vv = C13/(C11 + C12)                      (vertical Poisson's ratio)
        vh = (C12*C33 - C13^2)/(C11*C33 - C13^2)  (horizontal Poisson's ratio)
        Gvh = C44 (x-z shear), Ghh = C66 (x-y shear).

    Reduces to the isotropic moduli when C33 = C11 and C44 = C66 (so C13 = C12).
    Returns a dict {Ev, Eh, vv, vh, Gvh, Ghh}.
    """
    ev = c33 - 2.0 * c13 ** 2 / (c11 + c12)
    eh = (c11 - c12) * (c11 * c33 - 2.0 * c13 ** 2 + c12 * c33) / (c11 * c33 - c13 ** 2)
    vv = c13 / (c11 + c12)
    vh = (c12 * c33 - c13 ** 2) / (c11 * c33 - c13 ** 2)
    return {"Ev": ev, "Eh": eh, "vv": vv, "vh": vh, "Gvh": c44, "Ghh": c66}


def closure_stress(sigma_v, pore_pressure, biot, e_h, nu_h, eps_min, eps_max):
    """Minimum horizontal (fracture closure) stress (Eq. 10, Thiercelin & Plumb, 1994)

        sHmin = nu/(1-nu)*(sigma_v - alpha*pp) + alpha*pp
                + E/(1-nu^2)*(eps_min + nu*eps_max).

    sHmin is the fracture closure stress; the two principal strains are usually
    fitted to DFIT data.  Use the (calibrated, static) horizontal E and v.
    """
    poro = nu_h / (1.0 - nu_h) * (sigma_v - biot * pore_pressure) + biot * pore_pressure
    tect = e_h / (1.0 - nu_h ** 2) * (eps_min + nu_h * eps_max)
    return poro + tect


# ---------------------------------------------- fracture width / elasticity --------------

def condensed_elasticity_iso(e, nu):
    """Isotropic condensed elasticity term  f = (1 - nu^2)/E  (Eq. 16)."""
    return (1.0 - nu ** 2) / e


def sneddon_berry_width(sigma_net, height, f):
    """Maximum fracture width of an elliptical crack (Sneddon & Berry, 1958; Eq. 11)

        w = 2*sigma_net*h*f,

    with f the condensed elasticity term (f_iso for isotropic, f_aniso for VTI).
    """
    return 2.0 * sigma_net * height * f


def equivalent_poisson_ratio(e_eq, f_aniso):
    """Equivalent isotropic Poisson's ratio (Eq. 22)

    Found by holding the condensed elasticity term constant after converting the
    anisotropic moduli to (Eeq, veq):  (1 - veq^2)/Eeq = f_aniso  ->
        veq = sqrt(1 - Eeq*f_aniso).
    """
    return np.sqrt(1.0 - e_eq * f_aniso)


# ---------------------------------------------- moduli / economics --------------

def equivalent_youngs_modulus(ah, av, avh, e_h, e_v, g_vh, nu_vh):
    """Equivalent isotropic Young's modulus (Eq. 21)

        Eeq = ah*Eh + av*Ev + 2*avh*Gvh*(1 + nu_vh),

    with the weights ah + av + avh = 1.
    """
    return ah * e_h + av * e_v + 2.0 * avh * g_vh * (1.0 + nu_vh)


def rofi(cumulative_production, price, proppant_cost, nonproppant_cost):
    """Return on fracturing investment (Eq. 23)

        ROFI = production*price - (proppant_cost + nonproppant_cost).
    """
    return cumulative_production * price - (proppant_cost + nonproppant_cost)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Shale Fracturing (ANNIE + ML)")
    print("=" * 60)

    # ANNIE closure satisfies the defining identity C13 + 2*C44 - C33 = 0
    c33, c44, c66 = 30.0, 8.0, 12.0
    c11 = annie_c11(c33, c44, c66)
    c13 = annie_c13(c11, c66)
    print(f"  C11 / C13              = {c11:.1f} / {c13:.1f}")
    assert np.isclose(c13 + 2 * c44 - c33, 0.0)

    # Modified-ANNIE reduces to ANNIE when k = k' = 1
    assert np.isclose(mannie_c13(c11, c66, 1.0), c13)
    assert np.isclose(mannie_c11(c33, c44, c66, 1.0), c11)

    # VTI engineering moduli reduce to isotropic when C33=C11 and C44=C66
    lam, mu = 20.0, 12.0
    c11i, c12i = lam + 2 * mu, lam
    iso = vti_elastic_moduli(c11i, c11i, mu, mu, c12i, c12i)
    e_iso = mu * (3 * lam + 2 * mu) / (lam + mu)
    nu_iso = lam / (2 * (lam + mu))
    assert np.isclose(iso["Ev"], e_iso) and np.isclose(iso["Eh"], e_iso)
    assert np.isclose(iso["vv"], nu_iso) and np.isclose(iso["vh"], nu_iso)

    # A genuine VTI medium: horizontal modulus exceeds the vertical (shale stiffer along beds)
    m = vti_elastic_moduli(c11=40.0, c33=30.0, c44=8.0, c66=12.0, c12=16.0, c13=11.0)
    print(f"  VTI  Eh / Ev           = {m['Eh']:.2f} / {m['Ev']:.2f} GPa")
    assert m["Eh"] > m["Ev"] > 0

    # Closure stress is between pore pressure and overburden for typical inputs
    shmin = closure_stress(sigma_v=7250.0, pore_pressure=3600.0, biot=0.7,
                           e_h=4.0e6, nu_h=0.25, eps_min=2e-4, eps_max=4e-4)
    print(f"  closure stress         = {shmin:.0f} psi")
    assert 3600.0 < shmin < 7250.0

    # Equivalent modulus is positive; the equivalent Poisson's ratio reproduces f_aniso
    eeq = equivalent_youngs_modulus(0.4, 0.4, 0.2, e_h=40.0, e_v=25.0, g_vh=12.0, nu_vh=0.25)
    f_aniso = condensed_elasticity_iso(m["Eh"], m["vh"])
    veq = equivalent_poisson_ratio(eeq, f_aniso)
    print(f"  equivalent E / v       = {eeq:.2f} GPa / {veq:.3f}")
    assert eeq > 0 and 0 < veq < 0.5
    assert np.isclose(condensed_elasticity_iso(eeq, veq), f_aniso)

    # Sneddon-Berry width scales with net pressure and the condensed term
    w = sneddon_berry_width(sigma_net=500.0, height=50.0, f=condensed_elasticity_iso(4.0e6, 0.25))
    assert w > 0

    # ROFI is positive when revenue exceeds cost
    assert rofi(1e5, 3.0, 1e5, 5e4) > 0 and rofi(1e3, 3.0, 1e5, 5e4) < 0
    print("  PASS")
    return {"C11": float(c11), "Eh": float(m["Eh"]), "Eeq": float(eeq), "shmin": float(shmin)}


if __name__ == "__main__":
    test_all()
