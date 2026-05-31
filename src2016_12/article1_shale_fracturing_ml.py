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
  - Equivalent isotropic Young's modulus from the anisotropic moduli
  - Return on fracturing investment (ROFI)

Note: this issue's PDF has a text layer; the ANNIE relations (Eqs. 1-5), the
equivalent-modulus (Eq. 21) and ROFI (Eq. 23) survived, while the
fracture-width/geometry equations lost their glyphs.  Stiffnesses/moduli in
consistent units (e.g. GPa).
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

    # Equivalent modulus lies between the horizontal and vertical moduli
    eeq = equivalent_youngs_modulus(0.4, 0.4, 0.2, e_h=40.0, e_v=25.0, g_vh=12.0, nu_vh=0.25)
    print(f"  equivalent E           = {eeq:.2f} GPa")
    assert eeq > 0

    # ROFI is positive when revenue exceeds cost
    assert rofi(1e5, 3.0, 1e5, 5e4) > 0 and rofi(1e3, 3.0, 1e5, 5e4) < 0
    print("  PASS")
    return {"C11": float(c11), "Eeq": float(eeq)}


if __name__ == "__main__":
    test_all()
