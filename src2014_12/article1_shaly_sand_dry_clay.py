"""
Article 1: Review of Existing Shaly-Sand Models and Introduction of a New
           Method Based on Dry-Clay Parameters
Max Peeters and Antony Holmes (2014)
Reference: Petrophysics Vol. 55, No. 6 (December 2014), pp. 543-553
DOI: none assigned (this issue predates SPWLA DOI assignment)

The classic shaly-sand conductivity models (Waxman-Smits, dual-water, modified
Simandoux) are reviewed and shown to be algebraically equivalent once the
bound-water term is written consistently.  The paper then introduces a "dry-clay
parameter" method: the dry-clay volume is read from the neutron-density
separation, and the Waxman-Smits cation-exchange term Qv is computed from
tabulated per-clay dry-clay parameters rather than from a measured CEC.

Implements:

  - Bound-water saturation  Sb = (phit - phieff)/phit  (Eq. 1) and
    Sb = Vsh*phitsh/phit  (Eq. 2)
  - Effective porosity  phieff = phit - Vsh*phitsh  (Eq. 3)
  - NaCl conductivity in meq  Co = 0.017*salinity_ppm  (Eq. 5)
  - Waxman-Smits conductivity (Eq. 9)
  - Juhasz Qv from dry-clay volume (Eq. 10)
  - Difference-method shale and dry-clay volumes (Eqs. 18, 19) and Qv (Eq. 20)
  - Dual-water conductivity (Eq. 22) and bound-water conductivity (Eq. 23)
  - Modified Simandoux conductivity (Eq. 25)
  - Waxman-Smits apparent cementation exponent (Eq. A1-1)

Note: this issue's PDF has a text layer; Eqs. 9, 10, 18-20, 22-23, 25 and A1-1
are transcribed from the body, while the typeset glyphs were dropped and
reconstructed in standard form (Waxman & Smits, 1968; Juhasz, 1981; Clavier et
al., 1984).  Conductivities in mho/m (S/m), porosities and volumes as fractions.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- bound water & porosity --------------

def bound_water_saturation_porosity(phit, phieff):
    """Bound-water saturation from total and effective porosity (Eq. 1)

        Sb = (phit - phieff)/phit.
    """
    return (phit - phieff) / phit


def bound_water_saturation_shale(vsh, phitsh, phit):
    """Bound-water saturation from shale volume (Eq. 2)

        Sb = Vsh*phitsh/phit,

    with phitsh the total porosity of a 100% shale layer adjacent to the sand.
    """
    return vsh * phitsh / phit


def effective_porosity(phit, vsh, phitsh):
    """Effective porosity from total porosity and shale volume (Eq. 3)

        phieff = phit - Vsh*phitsh.
    """
    return petrolib.porosity_lithology.effective_porosity(phit, vsh, phitsh, clip=None)


def co_from_salinity(salinity_ppm):
    """Formation-water concentration in meq/l from NaCl-equivalent salinity (Eq. 5)

        Co [meq/l] = 0.017*salinity [ppm NaCl eq.].
    """
    return 0.017 * np.asarray(salinity_ppm, float)


# ---------------------------------------------- Waxman-Smits --------------

def waxman_smits_conductivity(sw, cw, phit, m, n, b, qv):
    """Waxman-Smits total conductivity (Eq. 9)

        Ct = Sw^n*Cw*phit^m + B*Qv*phit^m*Sw^(n-1),

    the pure-sand Archie term plus the shale (cation-exchange) term.  B is the
    equivalent cation conductance (mho*ml/(meq*m)) and Qv the cation exchange
    capacity per unit pore volume (meq/ml).  The Sw^(n-1) factor means this is
    not a pure parallel-conductor model.
    """
    return petrolib.saturation_resistivity.waxman_smits_conductivity(
        sw, cw=cw, qv=qv, b=b, phi=phit, m_star=m, n_star=n)


def juhasz_qv(vcldry, rho_cldry, cec_cl, phit):
    """Juhasz cation exchange capacity per pore volume from dry clay (Eq. 10)

        Qv = Vcldry*rho_cldry*CECcl/phit,

    with the dry-clay volume Vcldry, dry-clay density rho_cldry (g/cm^3) and the
    clay CEC in meq/g.
    """
    return petrolib.saturation_resistivity.qv_juhasz(vcldry, rho_clay=rho_cldry, cec_clay=cec_cl, phit=phit)


# ---------------------------------------------- difference (dry-clay) method --------------

def shale_volume_difference(phin, phid, phin_sh, phid_sh):
    """Shale volume from the neutron-density separation (Eq. 18)

        Vsh = (phiN - phiD)/(phiN_sh - phiD_sh).
    """
    return petrolib.porosity_lithology.vshale_neutron_density(phin, phid, phin_sh, phid_sh)


def dry_clay_volume_difference(phin, phid, phin_cldry, phid_cldry):
    """Dry-clay volume from the neutron-density separation (Eq. 19)

        Vcldry = (phiN - phiD)/(phiN_cldry - phiD_cldry).
    """
    return petrolib.porosity_lithology.vshale_neutron_density(phin, phid, phin_cldry, phid_cldry)


def qv_dry_clay_method(phin, phid, phin_cldry, phid_cldry, rho_cldry, cec_cl, phit):
    """Qv from the dry-clay parameter method (Eq. 20, Eq. 19 into Eq. 10)

        Qv = (phiN - phiD)/(phiN_cldry - phiD_cldry)*rho_cldry*CECcl/phit.

    Because the dry-clay neutron-density difference (phiN_cldry - phiD_cldry)
    spans only ~0.26-0.42 across the four main clays, the clay CEC dominates.
    """
    vcldry = dry_clay_volume_difference(phin, phid, phin_cldry, phid_cldry)
    return juhasz_qv(vcldry, rho_cldry, cec_cl, phit)


# ---------------------------------------------- dual water & Simandoux --------------

def dual_water_conductivity(sw, cw, cb, sb, phit, m, n):
    """Dual-water total conductivity (Clavier et al., 1984; Eq. 22)

        Ct = Sw^n*phit^m*Cw + Sw^(n-1)*Sb*phit^m*(Cb - Cw),

    with the bound-water saturation Sb and bound-water conductivity Cb.
    """
    return petrolib.saturation_resistivity.dual_water_conductivity(sw, cw=cw, cb=cb, swb=sb, phi=phit, m=m, n=n)


def bound_water_conductivity(csh, phitsh):
    """Bound-water conductivity from the shale conductivity (Eq. 23, Archie m=2)

        Cb = Csh/phitsh^2.
    """
    return csh / phitsh ** 2


def modified_simandoux_conductivity(sw, cw, csh, vsh, phi, m):
    """Modified Simandoux total conductivity (Poupon et al., 1971; Eq. 25)

        Ct = Sw^2*phi^m*Cw + Vsh*Csh*Sw,

    where the extra Sw on the shale term is the Poupon modification of the
    original Simandoux equation.
    """
    sw = np.asarray(sw, float)
    return sw ** 2 * phi ** m * cw + vsh * csh * sw


def cementation_exponent_ws(m_star, rw, b, qv, phi):
    """Apparent Archie cementation exponent in the presence of clay (Eq. A1-1)

        m = m* + log(1 + Rw*B*Qv)/log(phi),

    relating the Waxman-Smits intrinsic exponent m* to the apparent Archie
    exponent m.  Because log(phi) < 0, the cation-exchange conductivity lowers
    the apparent exponent below m* (the rock looks more conductive, so its
    apparent formation factor and cementation exponent are reduced).
    """
    return m_star + np.log10(1.0 + rw * b * qv) / np.log10(phi)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Shaly-Sand Models & Dry-Clay Parameters")
    print("=" * 60)

    # Volumetric model: effective porosity and bound water are consistent
    phit, vsh, phitsh = 0.25, 0.30, 0.20
    phieff = effective_porosity(phit, vsh, phitsh)
    sb1 = bound_water_saturation_porosity(phit, phieff)
    sb2 = bound_water_saturation_shale(vsh, phitsh, phit)
    print(f"  phieff={phieff:.4f}  Sb={sb1:.4f}")
    assert np.isclose(phieff, 0.19) and np.isclose(sb1, sb2)

    # NaCl meq concentration scales linearly with salinity
    assert np.isclose(co_from_salinity(50000), 850.0)

    # Difference method: dry-clay volume and Qv are positive and consistent
    phin, phid = 0.30, 0.18
    vsh_d = shale_volume_difference(phin, phid, phin_sh=0.40, phid_sh=0.10)
    vcl = dry_clay_volume_difference(phin, phid, phin_cldry=0.37, phid_cldry=0.05)
    qv = qv_dry_clay_method(phin, phid, 0.37, 0.05, rho_cldry=2.53, cec_cl=0.11, phit=phit)
    print(f"  Vsh={vsh_d:.3f}  Vcldry={vcl:.3f}  Qv={qv:.4f}")
    assert 0 < vcl < 1 and qv > 0
    # ratio Vsh/Vcldry equals the dry-clay/shale separation ratio (Eq. 21)
    assert np.isclose(vsh_d / vcl, (0.37 - 0.05) / (0.40 - 0.10))

    # The three models all increase with Sw and reduce to Archie when clean
    cw, m, n, b = 5.0, 2.0, 2.0, 3.14
    cb = bound_water_conductivity(csh=2.0, phitsh=phitsh)
    ct_ws = waxman_smits_conductivity(0.6, cw, phit, m, n, b, qv)
    ct_dw = dual_water_conductivity(0.6, cw, cb, sb2, phit, m, n)
    ct_si = modified_simandoux_conductivity(0.6, cw, csh=2.0, vsh=vsh, phi=phit, m=m)
    print(f"  Ct: WS={ct_ws:.4f}  DW={ct_dw:.4f}  Simandoux={ct_si:.4f}")
    for ct, args in [(ct_ws, (0.9, cw, phit, m, n, b, qv))]:
        assert waxman_smits_conductivity(*args) > ct  # rises with Sw
    # clean-sand limit (no clay) reduces Waxman-Smits to Archie
    assert np.isclose(waxman_smits_conductivity(0.6, cw, phit, m, n, b, 0.0),
                      0.6 ** n * cw * phit ** m)

    # The cation-exchange term lowers the apparent Archie exponent below m*
    m_app = cementation_exponent_ws(m_star=1.8, rw=0.33, b=b, qv=qv, phi=phit)
    print(f"  apparent m = {m_app:.3f}")
    assert m_app < 1.8
    assert cementation_exponent_ws(1.8, 0.33, b, 0.0, phit) == 1.8  # no clay -> m*
    print("  PASS")
    return {"phieff": float(phieff), "Qv": float(qv), "m_app": float(m_app)}


if __name__ == "__main__":
    test_all()
