"""
Article 8: Review of the Intercept Method for Relative Permeability Correction a
           Variety of Case Study Data
Reed, Maas (2019)
DOI: 10.30632/PJV60N2-2019a6

Steady-state relative-permeability measurements are biased by capillary end
effects, which depend on flow rate.  The intercept method plots the apparent
(uncorrected) result against the reciprocal flow rate and extrapolates to
infinite rate (1/Q -> 0), where the end-effect contribution vanishes, giving the
corrected relative permeability.

Implements:

  - Darcy apparent relative permeability from a steady-state pressure drop
  - End-effect-biased apparent kr vs flow rate
  - Intercept extrapolation to 1/Q = 0 (end-effect-free kr)

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard end-effect / intercept-correction method the
paper reviews.
"""

import numpy as np


# ---------------------------------------------- Darcy kr ----------------

def apparent_kr(q, mu, L, A, dP, k_abs):
    """Apparent relative permeability from Darcy's law  kr = q*mu*L/(A*dP*k_abs)."""
    return q * mu * L / (A * np.asarray(dP, float) * k_abs)


# ---------------------------------------------- intercept method --------

def end_effect_apparent_kr(q, kr_true, end_effect_coeff):
    """Apparent kr biased by a rate-dependent capillary end effect.

        1/kr_app = 1/kr_true + end_effect_coeff/q
    The end-effect term decays as 1/Q and vanishes at infinite rate.
    """
    q = np.asarray(q, float)
    return 1.0 / (1.0 / kr_true + end_effect_coeff / q)


def intercept_correction(q, kr_app):
    """Recover the end-effect-free kr by extrapolating 1/kr_app vs 1/Q to 1/Q=0.

    Returns kr_true (the reciprocal of the intercept).
    """
    x = 1.0 / np.asarray(q, float)
    y = 1.0 / np.asarray(kr_app, float)
    slope, intercept = np.polyfit(x, y, 1)
    return 1.0 / intercept


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Intercept Method for Rel-Perm Correction")
    print("=" * 60)

    # Darcy apparent kr scales with rate and inversely with pressure drop
    assert apparent_kr(2e-6, 1e-3, 0.05, 1e-4, 5e4, 1e-13) > \
        apparent_kr(1e-6, 1e-3, 0.05, 1e-4, 5e4, 1e-13)

    # End effect biases the apparent kr below the true value, more so at low rate
    kr_true = 0.30
    q = np.array([1e-6, 2e-6, 4e-6, 8e-6])
    kr_app = end_effect_apparent_kr(q, kr_true, end_effect_coeff=1e-6)
    print(f"  kr_app vs rate         = {np.array2string(kr_app, precision=3)}")
    assert np.all(kr_app < kr_true)               # end effect lowers apparent kr
    assert np.all(np.diff(kr_app) > 0)            # bias shrinks as rate rises

    # Intercept method recovers the true (end-effect-free) kr
    kr_corr = intercept_correction(q, kr_app)
    print(f"  corrected kr           = {kr_corr:.3f}  (true {kr_true})")
    assert abs(kr_corr - kr_true) < 1e-6
    print("  PASS")
    return {"kr_app_lowQ": float(kr_app[0]), "kr_corrected": float(kr_corr)}


if __name__ == "__main__":
    test_all()
