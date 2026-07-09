"""
Article 1 (Tutorial): What is it about Shaly Sands? Shaly Sand Tutorial No. 3
                      of 3
Thomas (2018)
DOI: 10.30632/PJV59N3-2018t1

The third shaly-sand tutorial derives the Waxman-Smits conductivity model in
conductivity space: a clean sand plots C0 vs Cw as a line of slope 1/F through
the origin, but a shaly sand needs an extra clay-conductivity term so the line
no longer passes through the origin.  The extra conductivity comes from the
cation exchange capacity per pore volume (Qv) and the counterion conductance
(B), and the hydrocarbon-bearing case extends to a saturation equation.  A
Thomas-Stieber porosity example handles laminated shale.

Implements:

  - Clean-sand Archie in conductivity space  C0 = Cw/F
  - Shaly-sand modification  C0 = (Cw + CXTRA)/F*  with  CXTRA = B*Qv
  - Formation factor  F* = phi_t^(-m*)
  - Waxman-Smits saturation conductivity  Ct = (Sw^n*/F*)*(Cw + B*Qv/Sw)
  - Thomas-Stieber porosity for laminated shale

Note: this issue's PDF has a text layer but its typeset display-equation glyphs
were dropped in extraction, so the numbered relations (Eqs. 1-6) are faithful
standard-form reconstructions of the Waxman-Smits model the tutorial teaches.
Conductivities in S/m (or mho/cm consistently); Qv in equiv/L.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- conductivity space --------------

def clean_sand_c0(cw, formation_factor):
    """Clean-sand water-zone conductivity  C0 = Cw/F  (Eq. 1).

    A line of slope 1/F through the origin in (Cw, C0) space.
    """
    return np.asarray(cw, float) / formation_factor


def excess_conductivity(b, qv):
    """Clay excess conductivity  CXTRA = B*Qv  (Eq. 3).

    B = equivalent counterion conductance, Qv = cation exchange capacity per
    unit pore volume.
    """
    return b * np.asarray(qv, float)


def shaly_sand_c0(cw, formation_factor_star, cxtra):
    """Shaly-sand water-zone conductivity  C0 = (Cw + CXTRA)/F*  (Eq. 2).

    1/F* multiplies both the free-water and the clay term (parallel-circuit
    interpretation), so the line is offset from the origin by CXTRA/F*.
    """
    return (np.asarray(cw, float) + cxtra) / formation_factor_star


def formation_factor_star(phi_t, m_star):
    """Shaly-sand formation factor  F* = phi_t^(-m*)  (Eq. 4)."""
    # F* uses the shaly-rock exponent m* — same formula as the Archie F,
    # different calibration (see petrolib.saturation_resistivity docstring).
    return petrolib.saturation_resistivity.formation_factor(phi_t, m=m_star)


def waxman_smits_ct(cw, formation_factor_star, b, qv, sw, n_star=2.0):
    """Waxman-Smits conductivity of a partially saturated shaly sand (Eqs. 5-6)

        Ct = (Sw^n*/F*)*(Cw + B*Qv/Sw).

    Reduces to the water-zone C0 = (Cw + B*Qv)/F* at Sw = 1.
    """
    sw = np.asarray(sw, float)
    return (sw ** n_star / formation_factor_star) * (cw + b * qv / sw)


# ---------------------------------------------- laminated shale --------------

def thomas_stieber_porosity(phi_cl, x_sh, phi_shgr, x_sd=0.0, phi_sdgr=0.0):
    """Thomas-Stieber porosity  phi = phi_cl - x_sd*phi_sdgr + x_sh*phi_shgr.

    phi_cl = clean-sand porosity, x_sh/x_sd = shale/dispersed-sand volume
    fractions, phi_shgr/phi_sdgr = their grain porosities.
    """
    return phi_cl - x_sd * phi_sdgr + x_sh * phi_shgr


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1 (Tutorial): Shaly Sand No. 3 of 3")
    print("=" * 60)

    cw, F = 5.0, 10.0
    # The shaly line sits above the clean line by CXTRA/F* (offset from origin)
    cxtra = excess_conductivity(b=0.045, qv=0.5)
    c0_clean = clean_sand_c0(cw, F)
    c0_shaly = shaly_sand_c0(cw, F, cxtra)
    print(f"  C0 clean / shaly       = {c0_clean:.4f} / {c0_shaly:.4f} S/m")
    assert c0_shaly > c0_clean

    # F* rises as porosity falls
    assert formation_factor_star(0.10, 2.0) > formation_factor_star(0.25, 2.0)

    # Waxman-Smits: Ct at Sw=1 matches the shaly water-zone C0; drops as Sw falls
    Fs = formation_factor_star(0.20, 2.0)
    ct_sw1 = waxman_smits_ct(cw, Fs, 0.045, 0.5, sw=1.0)
    assert np.isclose(ct_sw1, shaly_sand_c0(cw, Fs, cxtra))
    assert waxman_smits_ct(cw, Fs, 0.045, 0.5, 0.5) < ct_sw1

    # Thomas-Stieber laminated example: phi_cl=0.30, 15% shale, phi_shgr=0.10
    phi = thomas_stieber_porosity(phi_cl=0.30, x_sh=0.15, phi_shgr=0.10)
    print(f"  Thomas-Stieber porosity = {phi:.3f}")
    assert np.isclose(phi, 0.315)
    print("  PASS")
    return {"C0_shaly": float(c0_shaly), "phi_str": float(phi)}


if __name__ == "__main__":
    test_all()
