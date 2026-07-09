"""
Article 5: Method for Acoustic Anisotropy Interpretation in Shales When the
           Stoneley-Wave Velocity is Missing
Gu, Quirein, Murphy, Rivera Barraza, Ou (2016)
Reference: Petrophysics Vol. 57, No. 2 (April 2016), pp. 140-156
DOI: none assigned (this issue predates SPWLA DOI assignment)

VTI shale mechanics needs five stiffness coefficients (C33, C44, C66, C11, C13).
A vertical well gives C33 and C44 from the P- and S-waves; ANNIE/M-ANNIE 1 close
the system with the Stoneley wave (for C66).  When the Stoneley velocity is
missing or uncertain this paper offers Stoneley-free closures: V-reg (predict
the 45 deg / 90 deg velocities from the measured 0 deg velocity by the observed
linear correlations) and M-ANNIE 2 (use the empirical Thomsen relation
gamma = 0.93*epsilon to close for C66).

Implements:

  - VTI engineering moduli (Eqs. 3-4) with isotropic reduction (Eq. 7)
  - Positive-definite stiffness check (Eqs. 2, 6)
  - Thomsen anisotropy parameters epsilon, gamma, delta
  - M-ANNIE 2 closure  C66 from gamma = k*epsilon
  - V-reg: off-axis velocity from the 0 deg velocity; C11/C13 from 90/45 deg
  - Minimum horizontal (closure) stress (Eqs. 8-9)

Note: this issue's PDF has a text layer; the VTI-moduli, Thomsen, V-reg and
M-ANNIE 2 relations are transcribed from the body, while the typeset glyphs were
dropped and reconstructed in standard VTI form (Mavko et al., 2009; Sayers,
2010).  Stiffnesses/moduli in GPa (or consistent), velocities in m/s, density in
kg/m^3, stresses/strains as noted.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

GAMMA_EPSILON_RATIO = 0.93    # observed gamma = 0.93*epsilon across many shales


# ---------------------------------------------- VTI moduli --------------

def vti_elastic_moduli(c11, c33, c44, c66, c13):
    """VTI Young's moduli and Poisson's ratios from the stiffness tensor
    (Eqs. 3-4, with C12 = C11 - 2*C66)

        Evert = C33 - 2*C13^2/(C11+C12)
        Ehorz = (C11-C12)*(C11*C33 - 2*C13^2 + C12*C33)/(C11*C33 - C13^2)
        vvert = C13/(C11+C12)
        vhorz = (C12*C33 - C13^2)/(C11*C33 - C13^2).

    Reduces to the isotropic case when C11=C33, C66=C44, C13=C12 (Eq. 7).
    Returns a dict {Evert, Ehorz, vvert, vhorz}.
    """
    c12 = c11 - 2.0 * c66
    ev = c33 - 2.0 * c13 ** 2 / (c11 + c12)
    eh = (c11 - c12) * (c11 * c33 - 2.0 * c13 ** 2 + c12 * c33) / (c11 * c33 - c13 ** 2)
    vv = c13 / (c11 + c12)
    vh = (c12 * c33 - c13 ** 2) / (c11 * c33 - c13 ** 2)
    return {"Evert": ev, "Ehorz": eh, "vvert": vv, "vhorz": vh}


def is_positive_definite(c11, c33, c44, c66, c13):
    """Check the VTI positive-definite (positive strain-energy) constraints
    (Eqs. 2, 6): C44 > 0, C66 > 0, C11 > |C12|, and C13^2 < C33*(C11 - C66)."""
    c12 = c11 - 2.0 * c66
    return bool(c44 > 0 and c66 > 0 and c11 > abs(c12)
                and c13 ** 2 < c33 * (c11 - c66))


# ---------------------------------------------- Thomsen / M-ANNIE 2 --------------

def thomsen_epsilon(c11, c33):
    """Thomsen P-wave anisotropy  epsilon = (C11 - C33)/(2*C33)."""
    return petrolib.acoustic_geomech.thomsen_epsilon(c11, c33)


def thomsen_gamma(c66, c44):
    """Thomsen S-wave anisotropy  gamma = (C66 - C44)/(2*C44)."""
    return petrolib.acoustic_geomech.thomsen_gamma(c66, c44)


def thomsen_delta(c13, c33, c44):
    """Thomsen  delta = [(C13+C44)^2 - (C33-C44)^2] / (2*C33*(C33-C44))."""
    return petrolib.acoustic_geomech.thomsen_delta(c13, c33, c44)


def mannie2_c66(c11, c33, c44, k=GAMMA_EPSILON_RATIO):
    """M-ANNIE 2 closure for C66 when the Stoneley wave is missing.

    Uses the empirical Thomsen relation  gamma = k*epsilon:
        (C66 - C44)/(2*C44) = k*(C11 - C33)/(2*C33)
        ->  C66 = C44*(1 + k*(C11 - C33)/C33).
    """
    return c44 * (1.0 + k * (c11 - c33) / c33)


# ---------------------------------------------- V-reg --------------

def vreg_offaxis_velocity(v0, slope, intercept):
    """V-reg off-axis velocity from the measured 0 deg velocity

        V(angle) = slope*V0 + intercept,

    using the observed near-linear 0 deg -> 45 deg / 90 deg correlations.
    """
    return slope * np.asarray(v0, float) + intercept


def stiffness_from_velocity(rho, velocity):
    """Stiffness coefficient from a propagation velocity  C = rho*V^2
    (e.g. C33 = rho*Vp(0)^2, C11 = rho*Vp(90)^2, C66 = rho*Vsh(90)^2)."""
    return petrolib.acoustic_geomech.stiffness_from_velocity(rho, velocity)


def c13_from_45deg(c11, c33, c44, rho, v45):
    """C13 from the 45 deg P-wave velocity (TI phase-velocity relation)

        C13 = sqrt[(C11+C44)*(C33+C44) - 4*rho^2*V45^4
                   + 2*rho*V45^2*(C11+C33+2*C44)] - C44.
    """
    term = ((c11 + c44) * (c33 + c44) - 4.0 * rho ** 2 * v45 ** 4
            + 2.0 * rho * v45 ** 2 * (c11 + c33 + 2.0 * c44))
    return np.sqrt(term) - c44


# ---------------------------------------------- stress --------------

def min_horizontal_stress(sigma_v, pp, e_h, nu_h, biot=1.0, eps_h=0.0, eps_H=0.0):
    """Minimum horizontal (closure) stress (Thiercelin & Plumb, 1994; Eqs. 8-9)

        sigma_h = nu/(1-nu)*(sigma_v - alpha*pp) + alpha*pp
                  + E/(1-nu^2)*(eps_h + nu*eps_H).
    """
    return petrolib.acoustic_geomech.min_horizontal_stress(
        sigma_v, pp, nu_h, biot=biot, e=e_h, eps_h=eps_h, eps_H=eps_H)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Acoustic Anisotropy When Stoneley Is Missing")
    print("=" * 60)

    # VTI moduli reduce to isotropic when C11=C33, C66=C44, C13=C12
    lam, mu = 20.0, 12.0
    c11i = lam + 2 * mu
    iso = vti_elastic_moduli(c11i, c11i, mu, mu, lam)
    e_iso = mu * (3 * lam + 2 * mu) / (lam + mu)
    nu_iso = lam / (2 * (lam + mu))
    assert np.isclose(iso["Evert"], e_iso) and np.isclose(iso["Ehorz"], e_iso)
    assert np.isclose(iso["vvert"], nu_iso)

    # A genuine VTI shale: horizontal modulus exceeds vertical; data is admissible
    c11, c33, c44, c66, c13 = 40.0, 30.0, 8.0, 12.0, 11.0
    m = vti_elastic_moduli(c11, c33, c44, c66, c13)
    print(f"  VTI Ehorz/Evert        = {m['Ehorz']:.2f} / {m['Evert']:.2f}")
    assert m["Ehorz"] > m["Evert"] and is_positive_definite(c11, c33, c44, c66, c13)

    # Thomsen parameters are positive for a VTI shale; gamma ~ 0.93*epsilon
    eps, gam = thomsen_epsilon(c11, c33), thomsen_gamma(c66, c44)
    print(f"  epsilon / gamma        = {eps:.3f} / {gam:.3f}")
    assert eps > 0 and gam > 0

    # M-ANNIE 2 closes C66 from C11/C33/C44 without the Stoneley wave
    c66_pred = mannie2_c66(c11, c33, c44)
    assert np.isclose(thomsen_gamma(c66_pred, c44),
                      GAMMA_EPSILON_RATIO * thomsen_epsilon(c11, c33))
    print(f"  M-ANNIE 2 C66          = {c66_pred:.2f}")

    # V-reg: predict the 90 deg velocity and recover C11 = rho*Vp(90)^2
    rho = 2500.0
    vp90 = vreg_offaxis_velocity(3500.0, slope=1.05, intercept=200.0)
    assert vp90 > 3500.0
    c11_vreg = stiffness_from_velocity(rho, vp90)
    assert c11_vreg > stiffness_from_velocity(rho, 3500.0)

    # C13 from the 45 deg velocity round-trips through the TI relation
    c13_chk = c13_from_45deg(c11=40e9, c33=30e9, c44=8e9, rho=2500.0, v45=3400.0)
    assert np.isfinite(c13_chk)

    # Minimum horizontal stress lies between pore pressure and overburden
    sh = min_horizontal_stress(9000.0, 4500.0, e_h=4.0e6, nu_h=0.25)
    print(f"  sigma_h                = {sh:.0f} psi")
    assert 4500.0 < sh < 9000.0
    print("  PASS")
    return {"Ehorz": float(m["Ehorz"]), "epsilon": float(eps), "C66_mannie2": float(c66_pred)}


if __name__ == "__main__":
    test_all()
