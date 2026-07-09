"""
Article 6: Improving Dielectric Interpretation by Calibrating Matrix Permittivity
           and Solving Dielectric Mixing Laws With a New Graphical Method
Wang, Wang, Toumelin, Brown, Crousse (2018)
DOI: 10.30632/PJV59N2-2018a5

Dielectric interpretation needs an accurate matrix permittivity, to which the
answer is very sensitive.  This module calibrates the matrix permittivity from
mineralogy with a CRIM (complex refractive index) mixing law - adding kerogen as
a matrix component in organic shales - and inverts the CRIM equation for
water-filled porosity and salinity.  It also quantifies the sensitivity that
motivates the paper's Complex-Domain Analysis (a graphical method that removes
the matrix permittivity as a required input).

Implements:

  - CRIM matrix permittivity  sqrt(eps_mat) = sum_i psi_i*sqrt(eps_i)
  - CRIM mixing of matrix + fluids  sqrt(eps) = sum phase sqrt(eps)
  - Simplified-CRIM inversion for water-filled porosity (phi_T = phi_w)
  - Matrix-permittivity -> water-filled-porosity sensitivity

Note: this issue's PDF has a text layer but the mixing-law expressions (Eqs.
2-7) lost their typeset glyphs in extraction, so the CRIM relations are faithful
standard-form reconstructions from the surviving definitions; the CDA inversion
itself is graphical in the paper.  Permittivities are relative (dimensionless).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- CRIM --------------

def crim_matrix_permittivity(fractions, eps_minerals):
    """CRIM matrix permittivity  sqrt(eps_mat) = sum_i psi_i*sqrt(eps_i)  (Eq. 3).

    fractions = volumetric mineral fractions (summing to 1, kerogen included as
    a matrix component), eps_minerals = their dielectric constants.
    """
    return petrolib.em_dielectric.mix_power_law(fractions, eps_minerals, alpha=0.5)


def crim_permittivity(phi, eps_mat, eps_w, sw=1.0, eps_hc=2.2):
    """CRIM bulk permittivity (Eq. 4)

        sqrt(eps) = (1-phi)*sqrt(eps_mat) + phi*Sw*sqrt(eps_w) + phi*(1-Sw)*sqrt(eps_hc).
    """
    return petrolib.em_dielectric.crim(
        phi, sw, eps_w=eps_w, eps_hc=eps_hc, eps_matrix=eps_mat
    )


def water_filled_porosity(eps_meas, eps_mat, eps_w):
    """Simplified-CRIM water-filled porosity (phi_T = phi_w, no oil/air) (Eq. 5)

        phi_w = (sqrt(eps_meas) - sqrt(eps_mat))/(sqrt(eps_w) - sqrt(eps_mat)).
    """
    return petrolib.em_dielectric.water_filled_porosity(
        eps_meas, eps_matrix=eps_mat, eps_w=eps_w, clip=False
    )


def matrix_sensitivity(eps_meas, eps_mat, eps_w, d_eps_mat=1.0):
    """Water-filled-porosity error from a matrix-permittivity error (p.u. per unit)."""
    base = water_filled_porosity(eps_meas, eps_mat, eps_w)
    perturbed = water_filled_porosity(eps_meas, eps_mat + d_eps_mat, eps_w)
    return abs(perturbed - base)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Dielectric Matrix Calibration (CRIM/CDA)")
    print("=" * 60)

    # Calibrate matrix permittivity from quartz/calcite/clay (+ kerogen)
    eps_mat = crim_matrix_permittivity([0.60, 0.20, 0.15, 0.05],
                                       [4.65, 8.05, 5.21, 3.23])
    print(f"  CRIM matrix permittivity = {eps_mat:.3f}")
    assert 4.0 < eps_mat < 6.0

    # Forward CRIM then invert recovers the planted water-filled porosity
    eps_w = 60.0
    eps_meas = crim_permittivity(0.18, eps_mat, eps_w, sw=1.0)
    phi_w = water_filled_porosity(eps_meas, eps_mat, eps_w)
    print(f"  recovered phi_w        = {phi_w:.3f}  (true 0.180)")
    assert np.isclose(phi_w, 0.18)

    # Matrix-permittivity error of +/-1 maps to ~2 p.u. water-filled porosity
    sens = matrix_sensitivity(eps_meas, eps_mat, eps_w, d_eps_mat=1.0)
    print(f"  sensitivity (+1 eps_mat) = {sens * 100:.1f} p.u.")
    assert 0.01 < sens < 0.04
    print("  PASS")
    return {"eps_mat": float(eps_mat), "phi_w": float(phi_w), "sensitivity": float(sens)}


if __name__ == "__main__":
    test_all()
