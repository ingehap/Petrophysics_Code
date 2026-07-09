"""
Article 8: Experimental Study on the Relationship Between Capillary Pressure
           and Resistivity Index in Tight Sandstone Rocks
Xiao, Yang, Li, Yang, Bernabe, Zhao, Li, Ren (2021)
DOI: 10.30632/PJV62N4-2021a7

Simultaneous gas-water capillary-pressure (Pc) and resistivity-index (I)
measurements on clay-bearing (Junggar) and clay-free (Fontainebleau) tight
sandstones.  Both follow power-law I-Sw (Archie) behaviour, but the Pc-I
relationship differs: clay-free rocks follow the Li & Williams power law,
while clayey rocks follow Szabo's linear model.

Implements:

  - Archie resistivity index   I = Sw^(-n)                       (Eq. 1)
  - Archie formation factor     F = phi^(-m)                      (Eq. 2)
  - Waxman-Smits resistivity index (clay-corrected)              (Eq. 5)
  - Normalized water saturation  Sw* = (Sw-Swr)/(1-Swr)          (Eq. 8)
  - Li & Williams power law  Pc = Pe * I^beta                    (Eq. 9)
  - Szabo linear model  I = a + b*Pc                             (Eq. 6)
  - Toledo fractal capillary pressure  Pc = Pew*Sw*^(-1/lambda)  (Eq. 18)
  - Washburn pore-throat radius  r = 2*sigma*cos(theta)/Pc
  - beta(k) and b(k) permeability regressions

Note: the journal's Eqs. 1-18 were image-rendered and not in the text; the
forms here are standard reconstructions consistent with the paper's prose and
its reported parameters (Fontainebleau n~1.17, Junggar n~1.69, D~2.62).
Pc in MPa, resistivities/indices dimensionless, saturation as fraction.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Eqs. 1-2: Archie --------

def resistivity_index(sw, n):
    """Archie second law  I = Rt/Ro = Sw^(-n)  (Eq. 1)."""
    return petrolib.saturation_resistivity.resistivity_index_from_sw(sw, n=n)


def formation_factor(phi, m, a=1.0):
    """Archie first law  F = a * phi^(-m)  (Eq. 2)."""
    return petrolib.saturation_resistivity.formation_factor(phi, a=a, m=m)


def saturation_from_index(I, n):
    """Invert Archie:  Sw = I^(-1/n)."""
    return petrolib.saturation_resistivity.sw_from_resistivity_index(I, n=n)


# ---------------------------------------------- Eq. 5: Waxman-Smits -----

def waxman_smits_index(sw, n_star, rw, B, qv):
    """Clay-corrected resistivity index (Eq. 5).

    I = Sw^(-n*) * (1 + Rw*B*Qv) / (1 + Rw*B*Qv/Sw).
    """
    sw = np.asarray(sw, float)
    f = rw * B * qv
    return sw ** (-n_star) * (1.0 + f) / (1.0 + f / sw)


# ---------------------------------------------- Eq. 8: normalized Sw ----

def normalized_saturation(sw, swr):
    """Sw* = (Sw - Swr) / (1 - Swr)  (Eq. 8)."""
    return (np.asarray(sw, float) - swr) / (1.0 - swr)


# ---------------------------------------------- Eqs. 9, 6: Pc-I ---------

def li_williams(I, pe, beta):
    """Li & Williams power-law Pc-I  Pc = Pe * I^beta  (Eq. 9)."""
    return pe * np.asarray(I, float) ** beta


def szabo_linear(Pc, a, b):
    """Szabo linear model  I = a + b*Pc  (Eq. 6)."""
    return a + b * np.asarray(Pc, float)


# ---------------------------------------------- Eq. 18: Toledo fractal --

def toledo_capillary(sw_star, pew, D):
    """Fractal capillary pressure  Pc = Pew * Sw*^(-1/lambda), lambda=3-D (Eq. 18)."""
    # Fractal Brooks-Corey Pc: lambda = 3 - D (fractal dimension); Sw* is already
    # normalized (swirr=0).  The library's exponent is -1/lam.
    lam = 3.0 - D
    return petrolib.capillary_pressure.brooks_corey_pc(
        sw_star, pc_entry=pew, lam=lam, swirr=0.0)


# ---------------------------------------------- Washburn ----------------

def washburn_radius(Pc_mpa, sigma=0.072, theta_deg=0.0):
    """Pore-throat radius  r = 2*sigma*cos(theta)/Pc.  Pc in MPa -> r in um."""
    # Signed cos; MPa->Pa on Pc and m->um on r are kept here in the facade.
    r_m = petrolib.capillary_pressure.washburn_radius(
        np.asarray(Pc_mpa, float) * 1e6, sigma=sigma, theta_deg=theta_deg, absolute=False)
    return r_m * 1e6


# ---------------------------------------------- permeability regressions

def beta_from_permeability(k_md):
    """Reported regression  beta = 5.083 * k^(-0.188)."""
    return 5.083 * np.asarray(k_md, float) ** (-0.188)


def szabo_slope_from_permeability(k_md):
    """Reported regression  b = 0.1648 * exp(4.58 * k)."""
    return 0.1648 * np.exp(4.58 * np.asarray(k_md, float))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Capillary Pressure vs Resistivity Index")
    print("=" * 60)

    sw = np.linspace(0.15, 1.0, 40)

    # Archie index: I=1 at Sw=1, increases as Sw drops; round-trips
    n_fb = 1.173            # Fontainebleau average
    I = resistivity_index(sw, n_fb)
    assert abs(I[-1] - 1.0) < 1e-9 and np.all(np.diff(I) < 0)
    assert np.allclose(saturation_from_index(I, n_fb), sw)
    print(f"  Archie I at Sw=0.5     = {resistivity_index(0.5, n_fb):.3f}")

    # Clayey rock has a higher saturation exponent
    n_jg = 1.686            # Junggar average
    assert resistivity_index(0.3, n_jg) > resistivity_index(0.3, n_fb)

    # Waxman-Smits index is reduced vs Archie in clayey rock (excess conductance)
    I_ws = waxman_smits_index(0.5, n_star=n_jg, rw=0.23, B=3.83, qv=0.023)
    I_ar = resistivity_index(0.5, n_jg)
    print(f"  I Archie / Waxman-Smits @Sw=0.5 = {I_ar:.3f} / {I_ws:.3f}")
    assert I_ws < I_ar

    # Formation factor
    assert abs(formation_factor(0.10, 2.0) - 100.0) < 1e-9

    # Li & Williams power law: Pc rises with I
    pc_lw = li_williams(I, pe=0.015, beta=6.619)
    assert np.all(np.diff(pc_lw) < 0) or np.all(np.diff(pc_lw) > 0)
    assert np.all(pc_lw > 0)

    # Toledo fractal Pc increases as water saturation drops
    sw_star = normalized_saturation(sw, swr=0.10)
    pc_frac = toledo_capillary(sw_star, pew=0.015, D=2.617)
    assert pc_frac[0] > pc_frac[-1] and np.all(pc_frac > 0)
    print(f"  fractal Pc range       = {pc_frac[-1]:.4f} .. {pc_frac[0]:.3f} MPa")

    # Washburn: higher Pc -> smaller throat radius
    r_lo = washburn_radius(0.1)
    r_hi = washburn_radius(5.0)
    print(f"  throat radius @0.1/5 MPa = {r_lo:.3f} / {r_hi:.4f} um")
    assert r_lo > r_hi > 0

    # Permeability regressions
    assert abs(beta_from_permeability(10.0) - 5.083 * 10.0 ** -0.188) < 1e-9
    assert szabo_slope_from_permeability(0.05) > 0
    print("  PASS")
    return {"I_at_0.5": float(resistivity_index(0.5, n_fb)),
            "n_fb": n_fb, "n_jg": n_jg, "D": 2.617}


if __name__ == "__main__":
    test_all()
