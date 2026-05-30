"""
Article 2: Effects of Clay Minerals and Pore-Water Conductivity on Saturation
           Exponent of Clay-Bearing Sandstones Based on Digital Rock
Fan, Pan, Guo, Lei (2020)
DOI: 10.30632/PJV61N4-2020a2

A segmented digital rock (Berea sandstone) is augmented with clay at the pore
walls to build clay-bearing models of fixed porosity; rock resistivity at each
water saturation is obtained from a finite-element conductivity simulation and
fitted with Archie's resistivity index to extract the saturation exponent n.
Clay (via the Waxman-Smits excess conductivity B*Qv) and pore-water conductivity
Cw both reshape the resistivity-index curve, so the apparent n is not constant.

Implements:

  - Archie formation factor and resistivity index  F = a*phi^-m, I = Sw^-n  (Eqs. 1-2)
  - Waxman-Smits saturated conductivity  C0 = (Cw + B*Qv)/F*               (Eqs. 3-4)
  - Cation mobility B(Cw) at 25 C (rises, then saturates)                  (Eq. 5)
  - Qv from CEC  Qv = rhoG*CEC*(1-phiT)/(100*phiT)                         (Eq. 6)
  - Partial-saturation Waxman-Smits conductivity and apparent n from the
    log I vs log Sw slope

Note: this issue's PDF text layer dropped the typeset glyphs, so these are the
standard Archie / Waxman-Smits forms anchored to the preserved definitions.
The finite-element solve (Eq. 7) is represented by the Waxman-Smits
conductivity model.  Paper anchors: clean-rock n ~ 1.5, clay lowers n (toward
~0.4 for montmorillonite), porosity 19.86%.
"""

import numpy as np


# ---------------------------------------------- Archie ------------------

def formation_factor(phi, a=1.0, m=2.0):
    """Archie formation factor  F = a*phi^-m  (Eq. 1)."""
    return a * np.asarray(phi, float) ** (-m)


def resistivity_index(sw, n=2.0, b=1.0):
    """Archie resistivity index  I = b*Sw^-n  (Eq. 2)."""
    return b * np.asarray(sw, float) ** (-n)


# ---------------------------------------------- Waxman-Smits ------------

def cation_mobility(Cw, B_max=4.6, Cw0=1.0):
    """Cation mobility B(Cw) at 25 C: rises with salinity, then saturates (Eq. 5).

        B = B_max * (1 - exp(-Cw/Cw0))   (S/m per (mmol/cm^3))
    """
    return B_max * (1.0 - np.exp(-np.asarray(Cw, float) / Cw0))


def qv_from_cec(cec, phi_t, rho_grain=2.65):
    """Qv from cation-exchange capacity  Qv = rhoG*CEC*(1-phiT)/(100*phiT)  (Eq. 6).

    CEC in mmol/100 g, Qv in mmol/cm^3.
    """
    return rho_grain * cec * (1.0 - phi_t) / (100.0 * phi_t)


def waxman_smits_c0(Cw, B, Qv, Fstar):
    """Brine-saturated rock conductivity  C0 = (Cw + B*Qv)/F*  (Eqs. 3-4)."""
    return (Cw + B * Qv) / Fstar


def waxman_smits_ct(sw, Cw, B, Qv, Fstar, n_star=2.0):
    """Partially saturated Waxman-Smits conductivity.

        Ct = (Sw^n* * Cw + B*Qv*Sw^(n*-1)) / F*
    The clay term B*Qv adds conductivity that decays more slowly than the
    brine term as Sw drops, flattening the resistivity index at low Sw.
    """
    sw = np.asarray(sw, float)
    return (sw ** n_star * Cw + B * Qv * sw ** (n_star - 1.0)) / Fstar


def apparent_n(sw, Cw, B, Qv, Fstar, n_star=2.0):
    """Apparent saturation exponent: slope of log(I) vs log(1/Sw)."""
    c0 = waxman_smits_c0(Cw, B, Qv, Fstar)
    ct = waxman_smits_ct(sw, Cw, B, Qv, Fstar, n_star)
    I = c0 / ct
    return float(np.polyfit(np.log(1.0 / np.asarray(sw, float)), np.log(I), 1)[0])


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Saturation Exponent of Clay-Bearing Sandstone")
    print("=" * 60)

    phi = 0.1986                       # fixed digital-rock porosity
    F = formation_factor(phi, m=2.0)
    print(f"  formation factor       = {F:.1f}")
    assert F > 1

    # Archie resistivity index round-trip: clean rock -> n recovered exactly
    sw = np.linspace(0.25, 1.0, 16)
    I_clean = resistivity_index(sw, n=2.0)
    n_fit = float(np.polyfit(np.log(1.0 / sw), np.log(I_clean), 1)[0])
    assert abs(n_fit - 2.0) < 1e-9

    # Cation mobility rises with Cw then saturates
    assert cation_mobility(0.5) < cation_mobility(5.0) < cation_mobility(50.0)
    assert cation_mobility(100.0) < 4.6 + 1e-9

    # Qv from CEC is positive and rises with CEC
    qv = qv_from_cec(cec=6.0, phi_t=0.1986)
    print(f"  Qv (CEC=6)             = {qv:.3f} mmol/cm^3")
    assert qv > 0 and qv_from_cec(20.0, 0.1986) > qv

    # Clay lowers the apparent saturation exponent below the clean value
    Cw, Fstar = 1.0, F
    n_clean = apparent_n(sw, Cw, cation_mobility(Cw), 0.0, Fstar, n_star=2.0)
    n_clay = apparent_n(sw, Cw, cation_mobility(Cw), 1.0, Fstar, n_star=2.0)
    n_mont = apparent_n(sw, Cw, cation_mobility(Cw), 5.0, Fstar, n_star=2.0)
    print(f"  apparent n clean/clay/montmor = {n_clean:.2f} / {n_clay:.2f} / {n_mont:.2f}")
    assert abs(n_clean - 2.0) < 1e-6              # no clay -> Archie n
    assert n_clay < n_clean                        # clay lowers apparent n
    assert n_mont < n_clay < 1.5                   # high-CEC clay drives n down

    # Higher pore-water conductivity dilutes the clay effect (n back toward n*)
    n_lowCw = apparent_n(sw, 0.5, cation_mobility(0.5), 1.0, Fstar)
    n_hiCw = apparent_n(sw, 13.0, cation_mobility(13.0), 1.0, Fstar)
    print(f"  apparent n  lowCw/hiCw = {n_lowCw:.2f} / {n_hiCw:.2f}")
    assert n_hiCw > n_lowCw
    print("  PASS")
    return {"F": F, "Qv": qv, "n_clean": n_clean, "n_clay": n_clay,
            "n_montmor": n_mont}


if __name__ == "__main__":
    test_all()
