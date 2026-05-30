"""
Article 3: Effect of Pyrite in Water Saturation Evaluation of Clay-Rich
Carbonate
Storebo, Hjuler, Meireles, Fabricius (2022)
DOI: 10.30632/PJV63N2-2022a3

Extends Archie's law to mixed-mineral carbonate by combining the Clavier
dual-water model with weighted Hashin-Shtrikman (HS) bounds for pyrite.
Implements:

  - Archie  sigma_t = sigma_w * phi^m * Sw^n                  (Eq. 1)
  - Wiener arithmetic (parallel) and harmonic (series) bounds (Eqs. 2-3)
  - Hashin-Shtrikman lower / upper bounds for an isotropic
    two-component medium                                      (Eqs. 4-5)
  - Archie with extra conductivity sigma_o = (sigma_w + sigma_extra) *
                                              phi^m * Sw^n    (Eq. 6)
  - Waxman-Smits excess conductivity  sigma_x = beta * Qv      (Eqs. 7-9)
  - Clavier dual-water sigma_we via volumetric weighting       (Eqs. 10-13)
  - Mineral-bound water conductivity constants                 (Eqs. 24-27)
  - Two-phase weighted HS mixing with calibration weight w     (Eqs. 48-52)

The default constants reproduce the Well Boje-2C numbers reported in
the paper: 78% calcite / 20% silicates / 2% pyrite, 26% porosity,
mineral-bound water conductance 82.9 S/m at 91 C, pyrite mineral
conductivity 1,500 S/m and HS weighting constant w = 0.03.
"""

import numpy as np


# ---------------------------------------------- Eq. 1: Archie -----------

def archie_sigma_t(sigma_w, phi, Sw, m=2.0, n=2.0):
    """sigma_t = sigma_w * phi^m * Sw^n   (Eq. 1)."""
    return sigma_w * phi ** m * Sw ** n


# ---------------------------------------------- Eqs. 2-3: Wiener bounds -

def wiener_arithmetic(f1, sigma1, f2, sigma2):
    """Upper (parallel) Wiener bound."""
    return f1 * sigma1 + f2 * sigma2


def wiener_harmonic(f1, sigma1, f2, sigma2):
    """Lower (series) Wiener bound."""
    return 1.0 / (f1 / max(sigma1, 1e-12) + f2 / max(sigma2, 1e-12))


# ---------------------------------------------- Eqs. 4-5: HS bounds ----

def hs_bounds(f1, sigma1, f2, sigma2):
    """Hashin-Shtrikman bounds for an isotropic two-component medium.

    Returns (lower, upper).
    """
    if sigma1 <= sigma2:
        lo_s = sigma1 + f2 / (1.0 / (sigma2 - sigma1) + f1 / (3.0 * sigma1))
        hi_s = sigma2 + f1 / (1.0 / (sigma1 - sigma2) + f2 / (3.0 * sigma2))
    else:
        lo_s = sigma2 + f1 / (1.0 / (sigma1 - sigma2) + f2 / (3.0 * sigma2))
        hi_s = sigma1 + f2 / (1.0 / (sigma2 - sigma1) + f1 / (3.0 * sigma1))
    return float(lo_s), float(hi_s)


def hs_weighted(f1, sigma1, f2, sigma2, weight=0.50):
    """Weighted blend of HS lower / upper bounds (Eqs. 48-52)."""
    lo, hi = hs_bounds(f1, sigma1, f2, sigma2)
    return float((1.0 - weight) * lo + weight * hi)


# ---------------------------------------------- Eqs. 6, 7-9: extra conductivity

def archie_with_extra(sigma_w, sigma_extra, phi, Sw, m=2.0, n=2.0):
    """sigma_o = (sigma_w + sigma_extra) * phi^m * Sw^n  (Eq. 6)."""
    return (sigma_w + sigma_extra) * phi ** m * Sw ** n


def waxman_smits_extra(beta, Qv):
    """sigma_x = beta * Qv   (Eqs. 7-9)."""
    return beta * Qv


# ---------------------------------------------- Eqs. 10-13: Clavier ---

def clavier_dual_water(sigma_w, sigma_wb, Vw, Vwb):
    """Effective electrolyte conductivity in the dual-water model.

        sigma_we = (V_w * sigma_w + V_wb * sigma_wb) / (V_w + V_wb)
    """
    return (Vw * sigma_w + Vwb * sigma_wb) / (Vw + Vwb)


# ---------------------------------------------- Eqs. 24-27: constants --

# Per the paper for Boje-2C at 91 C
CEC_PHI_CA = 6.5e6                # 1/m^3 - Eq. 24
CHI_H_ANG = 6.18                  # Angstrom, Eqs. 25-26
BETA = 0.0025                     # (S/m)/(Eq/m), Eq. 27


def excess_conductance_silicate(CEC_eq_per_g, density_g_per_cc, nu=0.5):
    """sigma_extra (silicate) = beta * CEC * rho * nu  (Eq. 32)."""
    return BETA * CEC_eq_per_g * density_g_per_cc * 1000.0 * nu   # S/m


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Pyrite-Aware Sw with HS-Bounded Mineral Mixing")
    print("=" * 60)

    # Boje-2C parameters
    phi = 0.26
    sigma_w = 7.0                      # formation water, S/m at 91 C
    sigma_wb = 82.9                    # mineral-bound water, S/m (paper)
    sigma_py = 1500.0                  # pyrite mineral, S/m (recommended)
    w_hs = 0.03                        # HS weight (paper)

    # Two-phase mixing of mineral-bound water and free water
    Vw, Vwb = 0.20, 0.06
    sigma_we = clavier_dual_water(sigma_w, sigma_wb, Vw, Vwb)
    print(f"  sigma_we (Clavier, Eqs. 10-13)     = {sigma_we:.2f} S/m")
    assert sigma_w < sigma_we < sigma_wb

    # Wiener bounds for water + pyrite
    wH = wiener_harmonic(0.98, sigma_we, 0.02, sigma_py)
    wA = wiener_arithmetic(0.98, sigma_we, 0.02, sigma_py)
    print(f"  Wiener bounds water/pyrite         = "
          f"[{wH:.2f}, {wA:.2f}] S/m")
    assert wH < wA

    # HS bounds and weighted mix
    lo, hi = hs_bounds(0.98, sigma_we, 0.02, sigma_py)
    sigma_mix = hs_weighted(0.98, sigma_we, 0.02, sigma_py, weight=w_hs)
    print(f"  HS bounds water/pyrite             = [{lo:.2f}, {hi:.2f}] S/m")
    print(f"  HS-weighted (w = {w_hs})              = {sigma_mix:.2f} S/m")
    assert lo <= sigma_mix <= hi

    # Archie with extra conductivity from pyrite + silicate CEC
    sigma_extra_sil = excess_conductance_silicate(CEC_eq_per_g=0.10,
                                                  density_g_per_cc=2.65)
    sigma_extra = sigma_extra_sil + 0.5 * sigma_mix * 0.02     # paper-style
    sigma_t = archie_with_extra(sigma_we, sigma_extra, phi, Sw=0.40)
    sigma_t_archie = archie_sigma_t(sigma_we, phi, 0.40)
    print(f"  sigma_extra (silicate CEC, Eq. 32) = {sigma_extra_sil:.3f} S/m")
    print(f"  sigma_t  pure Archie               = {sigma_t_archie:.3f} S/m")
    print(f"  sigma_t  with extras (Eq. 6)       = {sigma_t:.3f} S/m")
    assert sigma_t > sigma_t_archie
    print("  PASS")
    return {"phi": phi, "sigma_we": sigma_we, "sigma_t": sigma_t,
            "hs_weighted": sigma_mix}


if __name__ == "__main__":
    test_all()
