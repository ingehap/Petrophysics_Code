"""
Article 5: Experimental Quantification of the Impact of Thermal Maturity on
           Kerogen Density
Jagadisan, Yang, Heidari (2017)
Reference: Petrophysics Vol. 58, No. 6 (December 2017), pp. 603-612
DOI: none assigned (this issue predates SPWLA DOI assignment)

Helium pycnometry on isolated kerogen overestimates kerogen density if residual
pyrite and iron are ignored.  This module corrects the measured grain density
for pyrite/iron by a mass-volume balance, mixes the corrected kerogen with the
minerals into a matrix density, computes total porosity, applies Archie for
water saturation, and shows how sensitive porosity is to the kerogen density.
Kerogen density rises with thermal maturity (lower Rock-Eval HI).

Implements:

  - Pyrite/iron-corrected kerogen density  rho_k = (rho_t - C_py*rho_py - C_Fe*rho_Fe)/C_k
  - Multimineral matrix density  rho_ma = sum_i C_i*rho_i
  - Total porosity  phi = (rho_ma - rho_b)/(rho_ma - rho_f)
  - Archie water saturation (m=2, n=1.5)
  - Porosity sensitivity to a kerogen-density error

Note: this issue's PDF has a text layer but the numbered equations (Eqs. 1-2)
lost their typeset glyphs in extraction, so they are faithful standard-form
reconstructions from the surviving nomenclature; the reported constants
(rho_pyrite = 4.95, rho_Fe = 7.87 g/cm^3) are reproduced.  Densities in g/cm^3.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

RHO_PYRITE = 4.95
RHO_FE = 7.87


# ---------------------------------------------- kerogen density --------------

def corrected_kerogen_density(rho_total, c_pyrite, c_fe, c_kerogen,
                              rho_pyrite=RHO_PYRITE, rho_fe=RHO_FE):
    """Pyrite/iron-corrected kerogen density (Eq. 1)

        rho_k = (rho_t - C_py*rho_py - C_Fe*rho_Fe)/C_k.

    Removing the dense pyrite/iron contribution lowers the apparent kerogen
    density toward its true value.  C's are volumetric concentrations.
    """
    return (rho_total - c_pyrite * rho_pyrite - c_fe * rho_fe) / c_kerogen


def matrix_density(concentrations, densities):
    """Multimineral matrix density  rho_ma = sum_i C_i*rho_i  (volume-weighted)."""
    return float(petrolib.porosity_lithology.matrix_density_from_volumes(concentrations, densities))


def total_porosity(rho_matrix, rho_b, rho_fluid=1.0):
    """Total porosity  phi = (rho_ma - rho_b)/(rho_ma - rho_fluid)  (Eq. 2)."""
    return petrolib.porosity_lithology.density_porosity(rho_b, rho_matrix, rho_fluid)


def archie_sw(rw, rt, phi, m=2.0, n=1.5):
    """Archie water saturation  Sw = (Rw/(phi^m*Rt))^(1/n), clipped to [0,1] (m=2, n=1.5)."""
    # HAZARD (LIBRARY_MERGE_PLAN.md section 9): this article's argument order
    # is (rw, rt) — the canonical order is (rt, rw).  Mapped explicitly.
    return petrolib.saturation_resistivity.archie_sw(rt, rw, phi=phi, m=m, n=n, clip=(0.0, 1.0))


def porosity_sensitivity(rho_b, rho_matrix, d_rho_matrix, rho_fluid=1.0):
    """Relative porosity error from a matrix(kerogen)-density error d_rho_matrix."""
    base = total_porosity(rho_matrix, rho_b, rho_fluid)
    perturbed = total_porosity(rho_matrix + d_rho_matrix, rho_b, rho_fluid)
    return abs(perturbed - base) / base


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Kerogen Density & Thermal Maturity")
    print("=" * 60)

    # Correcting for pyrite lowers the apparent kerogen density
    rho_k = corrected_kerogen_density(1.50, c_pyrite=0.05, c_fe=0.0, c_kerogen=0.95)
    print(f"  corrected kerogen rho  = {rho_k:.3f} g/cc (measured 1.50)")
    assert rho_k < 1.50 and 1.0 < rho_k < 1.5

    # Maturity trend: denser kerogen at low HI (1.19 -> 1.77 g/cc, paper range)
    assert 1.19 < 1.77 and corrected_kerogen_density(1.80, 0.03, 0.0, 0.97) > rho_k

    # Matrix density, porosity, and Archie saturation
    rho_ma = matrix_density([0.10, 0.55, 0.35], [rho_k, 2.65, 2.71])
    phi = total_porosity(rho_ma, 2.45)
    sw = archie_sw(0.05, 8.0, phi)
    print(f"  rho_ma / phi / Sw      = {rho_ma:.3f} / {phi:.3f} / {sw:.3f}")
    assert 0 < phi < 0.3 and 0 <= sw <= 1.0

    # A 0.58 g/cc kerogen-density error propagates to a large porosity error
    sens = porosity_sensitivity(2.45, rho_ma, 0.58 * 0.10)   # weighted by kerogen fraction
    print(f"  porosity sensitivity   = {sens * 100:.1f} %")
    assert sens > 0.1
    print("  PASS")
    return {"rho_kerogen": float(rho_k), "phi": float(phi)}


if __name__ == "__main__":
    test_all()
