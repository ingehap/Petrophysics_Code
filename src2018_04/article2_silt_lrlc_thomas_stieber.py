"""
Article 2: The Problem With Silt in Low-Resistivity Low-Contrast (LRLC) Pay
           Reservoirs
Belevich, Bal (2018)
DOI: 10.30632/PJV59N2-2018a1

In laminated sand-silt-shale thin beds the binary Thomas-Stieber model breaks
down because silt is a third end member.  This module implements the
shale-distribution porosity relations (dispersed, laminated, structural), the
Thomas-Stieber sand-lamina porosity that redistributes the bulk porosity over
the laminar shale fraction, and the resistivity-anisotropy (Rv/Rh) discriminator
that reveals hydrocarbons hidden in thin sands.

Implements:

  - Dispersed/laminated/structural shale effective porosity
  - Total porosity  phi_T = phi_Sd + Vsh*phi_Sh
  - Thomas-Stieber sand-lamina porosity  phi_sand = (phi_bulk - Vlam*phi_TSH)/(1 - Vlam)
  - Resistivity anisotropy ratio  Rv/Rh

Note: this issue's PDF has a text layer but its typeset display-equation glyphs
were partly dropped in extraction, so the numbered relations (Eqs. 1-5) are
faithful standard-form reconstructions from the surviving variable definitions
and the modeling example (phi_Sd=0.3, Vsh=0.3, phi_Sh=0.2).  Fractions.
"""

import numpy as np


# ---------------------------------------------- shale distribution --------------

def dispersed_clay_porosity(phi_sd, vsh, phi_sh):
    """Dispersed-clay effective porosity  phi_e = phi_Sd - Vsh*phi_Sh  (Eq. 1)."""
    return phi_sd - vsh * phi_sh


def laminated_shale_porosity(phi_sd, vsh, phi_sh):
    """Laminated-shale effective porosity  phi_e = phi_Sd - Vsh*phi_Sd + Vsh*phi_Sh."""
    return phi_sd - vsh * phi_sd + vsh * phi_sh


def total_porosity(phi_sd, vsh, phi_sh):
    """Total porosity  phi_T = phi_Sd + Vsh*phi_Sh  (Eq. 3)."""
    return phi_sd + vsh * phi_sh


def structural_shale_porosity(phi_t, vsh, phi_sh):
    """Structural-shale effective porosity  phi_e = phi_T - Vsh*phi_Sh  (Eq. 4)."""
    return phi_t - vsh * phi_sh


# ---------------------------------------------- thomas-stieber --------------

def thomas_stieber_sand_porosity(phi_bulk, vlam, phi_tsh):
    """Thomas-Stieber sand-lamina total porosity (Eq. 5)

        phi_sand = (phi_bulk - Vlam*phi_TSH)/(1 - Vlam).

    Redistributing the bulk porosity over the laminar shale fraction Vlam: a
    larger Vlam boosts the recovered clean-sand porosity.
    """
    return (phi_bulk - vlam * phi_tsh) / (1.0 - vlam)


def anisotropy_ratio(rv, rh):
    """Resistivity anisotropy  Rv/Rh (>= 1 in laminated pay; Rv sees thin sands)."""
    return rv / rh


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Silt in LRLC Pay (Thomas-Stieber)")
    print("=" * 60)

    phi_sd, vsh, phi_sh = 0.30, 0.30, 0.20
    # Ordering of the shale-distribution models at the same volumetrics
    disp = dispersed_clay_porosity(phi_sd, vsh, phi_sh)
    lam = laminated_shale_porosity(phi_sd, vsh, phi_sh)
    tot = total_porosity(phi_sd, vsh, phi_sh)
    print(f"  phi disp/lam/total     = {disp:.3f} / {lam:.3f} / {tot:.3f}")
    assert disp < lam < tot
    assert np.isclose(structural_shale_porosity(tot, vsh, phi_sh), phi_sd)

    # Thomas-Stieber: a larger laminar fraction boosts the recovered sand porosity
    ts_lo = thomas_stieber_sand_porosity(0.22, 0.20, 0.10)
    ts_hi = thomas_stieber_sand_porosity(0.22, 0.40, 0.10)
    print(f"  TS sand phi Vlam .2/.4 = {ts_lo:.3f} / {ts_hi:.3f}")
    assert ts_hi > ts_lo

    # Anisotropy: vertical resistivity exceeds horizontal in laminated pay
    assert anisotropy_ratio(8.0, 2.0) == 4.0 and anisotropy_ratio(2.0, 2.0) == 1.0
    print("  PASS")
    return {"phi_total": float(tot), "TS_sand_hi": float(ts_hi)}


if __name__ == "__main__":
    test_all()
