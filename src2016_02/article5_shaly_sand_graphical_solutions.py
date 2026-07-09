"""
Article 5: Graphical Solutions for Laminated and Dispersed Shaly Sands
Bootle (2016)
Reference: Petrophysics Vol. 57, No. 1 (February 2016), pp. 51-59
DOI: none assigned (this issue predates SPWLA DOI assignment)

Regular submission.  Shaly-sand saturation equations are recast as graphical
solutions and their impact relative to Archie is quantified.  For dispersed clay
the Waxman-Smits/Juhasz equation adds a clay-conductivity term (Rw*B*Qv) to
Archie, with Qv from cation exchange capacity (Juhasz) and exponents m*, n* that
vary with clay content (reverting to Archie at Vcl = 0).  For laminated shale
the horizontal/vertical resistivity (Rh-Rv) anisotropy model solves jointly for
the laminated-shale volume and the sand water saturation.

Implements:

  - Archie water saturation (Eq. 3)
  - Waxman-Smits/Juhasz conductivity and Sw solve (Eqs. 2, 5-6)
  - Juhasz Qv from CEC (Eq. 4); clay-dependent variable exponents
  - Laminated-shale Rh (parallel) / Rv (series) and anisotropy
  - Joint Rh-Rv solution for laminated-shale volume and sand resistivity

Note: this issue's PDF has a text layer; the Archie / Waxman-Smits / Juhasz
relations (Eqs. 2-6) and the Rh-Rv laminated-shale model are transcribed from the
body, while the typeset glyphs were dropped and reconstructed in standard form.
Resistivities in ohm-m, Qv in meq/cm^3, CEC in meq/g, density in g/cm^3.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Archie --------------

def archie_sw(rt, rw, phi, m=2.0, n=2.0, a=1.0):
    """Archie water saturation (Eq. 3)

        Sw = (a*Rw/(phi^m * Rt))^(1/n).
    """
    return petrolib.saturation_resistivity.archie_sw(rt, rw, phi=phi, a=a, m=m, n=n)


# ---------------------------------------------- Waxman-Smits / Juhasz --------------

def juhasz_qv(cec, rho_dry_clay, vcl, phi):
    """Juhasz (1981) Qv from cation exchange capacity (Eq. 4)

        Qv = CEC*rho_dry_clay*Vcl/phi,

    the effective clay-counterion concentration per unit pore volume.
    """
    return petrolib.saturation_resistivity.qv_juhasz(vcl, rho_clay=rho_dry_clay, cec_clay=cec, phit=phi)


def variable_exponent(base, clay_coeff, vcl):
    """Clay-dependent Waxman-Smits exponent  e* = e + E*Vcl  (Eqs. 5-6),

    reverting to the Archie exponent at Vcl = 0.
    """
    return base + clay_coeff * vcl


def waxman_smits_ct(sw, rw, phi, qv, m_star, n_star, b, a=1.0):
    """Waxman-Smits conductivity (Eq. 2)

        Ct = (phi^m*/a)*(Sw^n*/Rw + B*Qv*Sw^(n*-1)),

    the Archie term plus the clay term Rw*B*Qv (here as conductivity).
    """
    return petrolib.saturation_resistivity.waxman_smits_conductivity(
        sw, cw=1.0 / rw, qv=qv, b=b, phi=phi, m_star=m_star, n_star=n_star) / a


def waxman_smits_sw(rt, rw, phi, qv, m_star, n_star, b, a=1.0, tol=1e-10):
    """Solve the Waxman-Smits equation for water saturation (given Ct = 1/Rt),
    by bisection on Sw in (0, 1]."""
    target = 1.0 / rt
    lo, hi = 1e-6, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if waxman_smits_ct(mid, rw, phi, qv, m_star, n_star, b, a) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------- laminated Rh-Rv --------------

def laminated_rh(rsand, rshale, vlam):
    """Horizontal (parallel) resistivity of a laminated sand-shale sequence

        1/Rh = (1 - Vlam)/Rsand + Vlam/Rshale.
    """
    return 1.0 / ((1.0 - vlam) / rsand + vlam / rshale)


def laminated_rv(rsand, rshale, vlam):
    """Vertical (series) resistivity of a laminated sand-shale sequence

        Rv = (1 - Vlam)*Rsand + Vlam*Rshale.
    """
    return (1.0 - vlam) * rsand + vlam * rshale


def laminated_anisotropy(rh, rv):
    """Resistivity anisotropy  lambda = sqrt(Rv/Rh)  (>= 1 for laminated shale)."""
    return np.sqrt(rv / rh)


def solve_laminated(rh, rv, rshale):
    """Joint Rh-Rv solution for the laminated-shale volume and sand resistivity.

    Given the measured horizontal/vertical resistivities and the shale
    resistivity, solves the parallel/series pair for (Vlam, Rsand) by bisection
    on Vlam in [0, 1).  Returns (Vlam, Rsand).
    """
    ch, csh = 1.0 / rh, 1.0 / rshale

    def rv_model(v):
        rsd = (1.0 - v) / (ch - v * csh)      # from the parallel (Rh) equation
        return (1.0 - v) * rsd + v * rshale   # series (Rv) equation

    lo, hi = 0.0, 1.0 - 1e-6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        # Rv increases with Vlam (more resistive shale in series); invert
        if rv_model(mid) < rv:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    vlam = 0.5 * (lo + hi)
    rsand = (1.0 - vlam) / (ch - vlam * csh)
    return vlam, rsand


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Shaly-Sand Graphical Solutions")
    print("=" * 60)

    # Archie baseline saturation
    sw_archie = archie_sw(20.0, 0.05, 0.13, m=2.0, n=2.0)
    print(f"  Archie Sw              = {sw_archie:.3f}")
    assert 0 < sw_archie < 1

    # Waxman-Smits lowers Sw vs Archie (the clay term adds conductivity)
    qv = juhasz_qv(cec=0.2, rho_dry_clay=2.7, vcl=0.2, phi=0.13)
    m_star = variable_exponent(2.0, 0.0, 0.2)
    n_star = variable_exponent(2.0, 0.0, 0.2)
    sw_ws = waxman_smits_sw(20.0, 0.05, 0.13, qv, m_star, n_star, b=4.0)
    print(f"  Qv / Waxman-Smits Sw   = {qv:.3f} / {sw_ws:.3f}")
    assert sw_ws < sw_archie
    # With no clay (Qv = 0) Waxman-Smits reverts to Archie
    assert np.isclose(waxman_smits_sw(20.0, 0.05, 0.13, 0.0, 2.0, 2.0, b=4.0),
                      sw_archie, atol=1e-3)

    # Laminated model: Rv (series) >= Rh (parallel), anisotropy >= 1
    rh = laminated_rh(5.0, 2.0, 0.3)
    rv = laminated_rv(5.0, 2.0, 0.3)
    lam = laminated_anisotropy(rh, rv)
    print(f"  Rh / Rv / anisotropy   = {rh:.3f} / {rv:.3f} / {lam:.3f}")
    assert rv > rh and lam > 1.0

    # The joint Rh-Rv solver recovers the laminated-shale volume and sand resistivity
    vlam, rsand = solve_laminated(rh, rv, rshale=2.0)
    print(f"  solved Vlam / Rsand    = {vlam:.3f} / {rsand:.3f}")
    assert np.isclose(vlam, 0.3, atol=1e-3) and np.isclose(rsand, 5.0, atol=1e-2)
    print("  PASS")
    return {"Sw_archie": float(sw_archie), "Sw_ws": float(sw_ws), "Vlam": float(vlam)}


if __name__ == "__main__":
    test_all()
