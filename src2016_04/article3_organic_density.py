"""
Article 3: A New Approach to Measuring Organic Density
Dang, Sondergeld, Rai (2016)
Reference: Petrophysics Vol. 57, No. 2 (April 2016), pp. 112-120
DOI: none assigned (this issue predates SPWLA DOI assignment)

Organic (kerogen) grain density is a critical, hard-to-measure input to shale
porosity from logs.  This paper measures it without hazardous kerogen isolation:
low-pressure (helium) pycnometer total grain density is combined with stepwise
low-temperature plasma ashing and TOC measurement.  Because specific volume
(1/density) mixes linearly by mass fraction, a plot of 1/rho_gt vs. TOC is a
straight line whose intercept gives the mineral grain density and whose slope
(with the TOC-to-organic-matter factor K) gives the organic grain density.

Implements:

  - Bulk density mass balance (Eq. 1)
  - Total grain density from mineral + kerogen (mass-balance mixing, Eq. 4)
  - Total organic matter  TOM = TOC/K  (Eq. 5)
  - Organic-density regression: fit 1/rho_gt vs. TOC -> (rho_mineral, rho_kerogen)

Note: this issue's PDF has a text layer; the mass-balance and TOM relations
(Eqs. 1, 4-5) are transcribed from the body, while the typeset glyphs were
dropped and reconstructed in standard form.  Densities in g/cm^3, TOC/TOM as
mass fractions.
"""

import numpy as np


# ---------------------------------------------- mass balance --------------

def bulk_density(rho_gm, v_m, rho_gk, v_k, rho_fm, phi_m, rho_fk, phi_k):
    """Bulk density mass balance (Eq. 1)

        rho_b = rho_gm*Vm + rho_gk*Vk + rho_fm*phi_m + rho_fk*phi_k,

    over grain-mineral, grain-kerogen, fluid-in-matrix and fluid-in-OM terms.
    """
    return rho_gm * v_m + rho_gk * v_k + rho_fm * phi_m + rho_fk * phi_k


def total_grain_density(rho_gm, rho_gk, w_kerogen):
    """Total grain density from mineral and kerogen grains (Eq. 4)

        1/rho_gt = (1 - w_k)/rho_gm + w_k/rho_gk,

    the mass-fraction (specific-volume) mixing rule; w_kerogen is the kerogen
    mass fraction of the grain.
    """
    return 1.0 / ((1.0 - w_kerogen) / rho_gm + w_kerogen / rho_gk)


def tom_from_toc(toc, k):
    """Total organic matter  TOM = TOC/K  (Eq. 5), K = TOC mass fraction of TOM."""
    return toc / k


# ---------------------------------------------- regression --------------

def organic_density_regression(toc, inv_rho_gt, k):
    """Fit organic and mineral grain densities from the 1/rho_gt vs. TOC line.

    Since 1/rho_gt = 1/rho_gm + (TOC/K)*(1/rho_gk - 1/rho_gm), a straight-line
    fit of 1/rho_gt against TOC gives
        intercept = 1/rho_gm
        slope     = (1/K)*(1/rho_gk - 1/rho_gm).
    Returns (rho_mineral, rho_kerogen).
    """
    toc = np.asarray(toc, float)
    y = np.asarray(inv_rho_gt, float)
    slope, intercept = np.polyfit(toc, y, 1)
    rho_gm = 1.0 / intercept
    inv_rho_gk = intercept + k * slope
    return rho_gm, 1.0 / inv_rho_gk


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: A New Approach to Measuring Organic Density")
    print("=" * 60)

    # Bulk density sums the mass-balance contributions
    rb = bulk_density(2.70, 0.80, 1.30, 0.10, 1.0, 0.05, 0.7, 0.05)
    print(f"  bulk density            = {rb:.3f} g/cm^3")
    assert 1.0 < rb < 3.0

    # Total grain density lies between the kerogen and mineral grain densities
    rgt = total_grain_density(rho_gm=2.70, rho_gk=1.30, w_kerogen=0.15)
    print(f"  total grain density     = {rgt:.3f} g/cm^3")
    assert 1.30 < rgt < 2.70

    # TOM exceeds TOC (K < 1)
    assert tom_from_toc(0.10, k=0.8) > 0.10

    # Regression recovers the mineral and kerogen grain densities used to
    # synthesize a 1/rho_gt vs. TOC line (with K converting TOC to organic mass)
    rho_gm_true, rho_gk_true, k = 2.70, 1.30, 0.80
    toc = np.array([0.02, 0.05, 0.08, 0.12, 0.16])
    w_k = toc / k                                 # kerogen mass fraction
    inv_rgt = (1.0 - w_k) / rho_gm_true + w_k / rho_gk_true
    rho_gm_fit, rho_gk_fit = organic_density_regression(toc, inv_rgt, k)
    print(f"  fitted rho_mineral/kerogen = {rho_gm_fit:.3f} / {rho_gk_fit:.3f}")
    assert np.isclose(rho_gm_fit, rho_gm_true) and np.isclose(rho_gk_fit, rho_gk_true)
    print("  PASS")
    return {"rho_gt": float(rgt), "rho_kerogen": float(rho_gk_fit)}


if __name__ == "__main__":
    test_all()
