"""
Article 1: Nanoindentation of Shale Cuttings and Its Application to Core
           Measurements
Esatyana, Sakhaee-Pour, Sadooni, Al-Kuwari (2020)
DOI: 10.30632/PJV61N5-2020a1

Nanoindentation on small drill cuttings (not core plugs) extracts the Young's
modulus of shale and upscales it to the core scale.  The Oliver-Pharr method
gives hardness from the maximum load and the indentation modulus from the
unloading contact stiffness; a plastic-zone size sets the lower bound on indent
spacing so the cutting-scale average equals the core-scale property.

Implements:

  - Hardness  H = Pmax / Ac                                       (Eq. 1a)
  - Oliver-Pharr indentation modulus  M = (sqrt(pi)/2)*S/(alpha*sqrt(Ac))  (Eq. 1b)
  - Specimen Young's modulus from M  Es = M*(1 - nu_s^2)          (Eq. 2)
  - Ideal Berkovich projected contact area  Ac = 24.5*hc^2
  - Johnson plastic-zone radius (indent-spacing lower bound)      (Eq. 4)

Note: this issue's PDF text layer preserved the equation numbers and variable
definitions but dropped the typeset glyphs, so the closed forms here are the
standard Oliver-Pharr / Johnson expressions anchored to those definitions.
Constants are the paper's: Berkovich alpha = 1.03, semi-angle 70.3 deg,
500 mN load, nu = 0.25, E = 20 GPa, yield 20 MPa.  SI units internally.
"""

import numpy as np

ALPHA_BERKOVICH = 1.03           # tip shape factor
BERKOVICH_SEMI_ANGLE = 70.3      # degrees (equivalent cone)


# ---------------------------------------------- Oliver-Pharr ------------

def berkovich_area(hc_m):
    """Ideal Berkovich projected contact area  Ac = 24.5 * hc^2  (m^2)."""
    return 24.5 * np.asarray(hc_m, float) ** 2


def hardness(p_max_N, Ac_m2):
    """Indentation hardness  H = Pmax / Ac  (Pa)  (Eq. 1a)."""
    return np.asarray(p_max_N, float) / np.asarray(Ac_m2, float)


def indentation_modulus(stiffness_N_m, Ac_m2, alpha=ALPHA_BERKOVICH):
    """Oliver-Pharr indentation modulus  M = (sqrt(pi)/2)*S/(alpha*sqrt(Ac))  (Eq. 1b).

    S = dP/dh is the unloading contact stiffness (N/m); returns M in Pa.
    """
    return (np.sqrt(np.pi) / 2.0) * np.asarray(stiffness_N_m, float) \
        / (alpha * np.sqrt(np.asarray(Ac_m2, float)))


def youngs_modulus(M_pa, nu_s=0.25):
    """Specimen Young's modulus  Es = M*(1 - nu_s^2)  (Eq. 2).

    The indenter compliance term is neglected because Ei >> Es.
    """
    return M_pa * (1.0 - nu_s ** 2)


# ---------------------------------------------- plastic zone ------------

def plastic_zone_radius(a_m, E_pa, sigma_y_pa, nu=0.25, semi_angle=BERKOVICH_SEMI_ANGLE):
    """Johnson (1970) expanding-cavity plastic-zone radius  c  (Eq. 4).

        (c/a)^3 = E*tan(theta) / (3*sigma_y*(1 - nu))
    a = contact-area radius; indents must be spaced > 2c to avoid interference.
    """
    theta = np.radians(semi_angle)
    ratio3 = E_pa * np.tan(theta) / (3.0 * sigma_y_pa * (1.0 - nu))
    return a_m * ratio3 ** (1.0 / 3.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Nanoindentation of Shale Cuttings")
    print("=" * 60)

    # Berkovich area and hardness at the paper's 500 mN load
    hc = 3.5e-6                       # ~3.5 um penetration
    Ac = berkovich_area(hc)
    H = hardness(0.5, Ac)             # 500 mN
    print(f"  contact area           = {Ac*1e12:.1f} um^2")
    print(f"  hardness               = {H/1e9:.3f} GPa")
    assert Ac > 0 and H > 0

    # Indentation modulus from a stiffness that should yield a shale-like Es.
    # Pick S so Es lands near the paper's 20 GPa basis.
    Es_target = 20e9
    M_target = Es_target / (1.0 - 0.25 ** 2)
    S = M_target * (ALPHA_BERKOVICH * np.sqrt(Ac)) / (np.sqrt(np.pi) / 2.0)
    M = indentation_modulus(S, Ac)
    Es = youngs_modulus(M, nu_s=0.25)
    print(f"  indentation modulus M  = {M/1e9:.2f} GPa")
    print(f"  Young's modulus Es     = {Es/1e9:.2f} GPa")
    assert abs(Es - Es_target) < 1e3      # round-trips to ~20 GPa

    # Poisson's-ratio sensitivity: changing nu by 0.1 alters Es by < ~6%
    Es_lo = youngs_modulus(M, nu_s=0.20)
    Es_hi = youngs_modulus(M, nu_s=0.30)
    rel = abs(Es_hi - Es_lo) / Es
    print(f"  Es sensitivity to nu   = {rel*100:.2f} %")
    assert rel < 0.06

    # Plastic zone is much larger than the contact radius (E >> yield)
    a = np.sqrt(Ac / np.pi)
    c = plastic_zone_radius(a, 20e9, 20e6)
    print(f"  contact a / plastic c  = {a*1e6:.1f} / {c*1e6:.1f} um")
    assert c > a
    # indent spacing of 2.2 mm safely exceeds 2c
    assert 2.2e-3 > 2 * c
    print("  PASS")
    return {"H_GPa": float(H/1e9), "Es_GPa": float(Es/1e9),
            "c_um": float(c*1e6)}


if __name__ == "__main__":
    test_all()
