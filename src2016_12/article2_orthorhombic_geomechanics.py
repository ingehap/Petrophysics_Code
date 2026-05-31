"""
Article 2: Geomechanics of Orthorhombic Media
Far, Quirein, Mekic (2016)
Reference: Petrophysics Vol. 57, No. 6 (December 2016), pp. 588-596
DOI: none assigned (this issue predates SPWLA DOI assignment)

A geomechanical model for orthorhombic media (a VTI background plus two
orthogonal vertical fracture sets).  Hooke's law links stress and strain through
the stiffness tensor; the compliance is its inverse; and a simplified horizontal-
stress model (with pore pressure) predicts the two horizontal stresses from the
anisotropic Young's moduli and Poisson's ratios.  It reduces to the standard VTI
model when the two horizontal directions are equal.  Shear-wave splitting
measures the anisotropy.

Implements:

  - Hooke's law  sigma = C*epsilon  and compliance  S = inv(C)
  - Orthorhombic horizontal stresses with pore pressure (Eqs. 25-26)
  - VTI reduction of the horizontal-stress model
  - Shear-wave splitting  SWS = (Vs_fast - Vs_slow)/Vs_fast

Note: this issue's PDF has a text layer and this article's key equations
survived as ASCII; the relations below are transcribed.  Stresses in psi, moduli
in psi, ratios/strains dimensionless.
"""

import numpy as np


# ---------------------------------------------- elasticity --------------

def hookes_stress(stiffness, strain):
    """Hooke's law in Voigt notation  sigma = C*epsilon  (Eq. 1)."""
    return np.asarray(stiffness, float) @ np.asarray(strain, float)


def compliance(stiffness):
    """Compliance matrix  S = inv(C)  (Eq. 3)."""
    return np.linalg.inv(np.asarray(stiffness, float))


# ---------------------------------------------- horizontal stress --------------

def horizontal_stress(sigma_v, pore_pressure, biot, e_strike, eps_strike,
                      nu_cross, eps_cross, nu_vertical):
    """Orthorhombic horizontal stress with pore pressure (Eqs. 25-26)

        sigma = nu_V*(sigma_v - alpha*P) + E*(eps + nu_cross*eps_cross) + alpha*P.

    Apply with the H-direction parameters for sigma_H and the h-direction
    parameters for sigma_h; setting the two cross ratios equal recovers VTI.
    """
    return (nu_vertical * (sigma_v - biot * pore_pressure)
            + e_strike * (eps_strike + nu_cross * eps_cross)
            + biot * pore_pressure)


def shear_wave_splitting(vs_fast, vs_slow):
    """Shear-wave splitting  SWS = (Vs_fast - Vs_slow)/Vs_fast  (Eq. 27)."""
    return (vs_fast - vs_slow) / vs_fast


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Geomechanics of Orthorhombic Media")
    print("=" * 60)

    # Compliance is the inverse of the stiffness (S*C = I)
    c = np.diag([30.0, 30.0, 25.0, 8.0, 8.0, 12.0]) + 0.0
    c[0, 1] = c[1, 0] = 10.0
    c[0, 2] = c[2, 0] = c[1, 2] = c[2, 1] = 9.0
    s = compliance(c)
    assert np.allclose(s @ c, np.eye(6), atol=1e-9)

    # Hooke's law returns a 6-vector of stresses
    sig = hookes_stress(c, [1e-4, 1e-4, 2e-4, 0, 0, 0])
    assert sig.shape == (6,)

    # Case-study inputs: the larger applied strain gives the larger horizontal stress
    sv, p, alpha = 7250.0, 3600.0, 0.7
    sH = horizontal_stress(sv, p, alpha, e_strike=4.0e6, eps_strike=4e-4,
                           nu_cross=0.2, eps_cross=2e-4, nu_vertical=0.25)
    sh = horizontal_stress(sv, p, alpha, e_strike=4.0e6, eps_strike=2e-4,
                           nu_cross=0.2, eps_cross=4e-4, nu_vertical=0.25)
    print(f"  sigma_H / sigma_h      = {sH:.0f} / {sh:.0f} psi")
    assert sH > sh

    # Shear-wave splitting is zero with no anisotropy, positive otherwise
    assert shear_wave_splitting(3000.0, 3000.0) == 0.0
    sws = shear_wave_splitting(3000.0, 2700.0)
    print(f"  shear-wave splitting   = {sws * 100:.1f} %")
    assert 0 < sws < 1
    print("  PASS")
    return {"sigma_H": float(sH), "SWS": float(sws)}


if __name__ == "__main__":
    test_all()
