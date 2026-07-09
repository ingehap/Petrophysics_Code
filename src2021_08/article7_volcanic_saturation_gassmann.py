"""
Article 7: Experimental Study on the Saturation Model of Volcanic Rock Based
           on Fluid Distribution
Pan, Zhou, Guo, Si, Lin (2021)
DOI: 10.30632/PJV62N4-2021a6

A *velocity*-based (NOT electrical) saturation model for tight volcanic rock.
P-wave velocity is related to water saturation via the Gassmann equation and
fluid-distribution mixing laws.  Pore-fluid distribution shifts from patchy
(high Sw) toward uniform (low Sw) as the rock dries, so the authors blend the
uniform (Gassmann-Brie) and patchy (White) moduli into one Gassmann-Brie-
Patchy (G-B-P) law tuned by a coefficient f.

Implements:

  - Gassmann saturated bulk modulus                              (Eq. 1)
  - Wood-Lindsay (Reuss) and Domenico (Voigt) fluid moduli       (Eqs. 2-3)
  - Brie et al. empirical fluid modulus                          (Eq. 4)
  - White patchy modulus (Berryman-Milton form)                  (Eq. 5)
  - Gassmann-Brie-Patchy blend                                   (Eq. 6)
  - P-wave velocity from (K, mu, rho)

Note: the journal's Eqs. 1-9 were image-rendered and not in the text; the
forms here are standard rock-physics reconstructions (Gassmann 1951; Wood &
Lindsay 1956; Domenico 1976; Brie 1995; White 1975).  Moduli in GPa,
velocity in m/s, density in g/cc.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

K_WATER = 2.2          # GPa
K_GAS = 0.000142       # GPa
RHO_WATER = 1.0        # g/cc
RHO_GAS = 0.001        # g/cc


# ---------------------------------------------- Eq. 1: Gassmann ---------

def gassmann_ksat(k_dry, k0, k_fl, phi):
    """Gassmann saturated bulk modulus (Eq. 1).  Shear modulus is unchanged."""
    return petrolib.acoustic_geomech.gassmann_ksat(
        k_dry=k_dry, k_mineral=k0, k_fluid=k_fl, phi=phi)


# ---------------------------------------------- Eqs. 2-4: fluid moduli --

def wood_lindsay(sw, kw=K_WATER, kg=K_GAS):
    """Reuss (harmonic) fluid modulus  1/Kfl = Sw/Kw + (1-Sw)/Kg  (Eq. 2)."""
    return petrolib.acoustic_geomech.wood_fluid_modulus([sw, 1.0 - sw], [kw, kg])


def domenico(sw, kw=K_WATER, kg=K_GAS):
    """Voigt (arithmetic) fluid modulus  Kfl = Sw*Kw + (1-Sw)*Kg  (Eq. 3)."""
    return petrolib.acoustic_geomech.voigt([sw, 1.0 - sw], [kw, kg])


def brie(sw, e=8.0, kw=K_WATER, kg=K_GAS):
    """Brie et al. empirical fluid modulus  Kfl = (Kw-Kg)*Sw^e + Kg  (Eq. 4)."""
    return petrolib.acoustic_geomech.brie_fluid_modulus(sw, kw, kg, e=e)


# ---------------------------------------------- Eq. 5: White patchy -----

def white_patchy(sw, k1, k2, mu):
    """White patchy modulus (Berryman-Milton form, Eq. 5).

    1/(Kpat + 4/3 mu) = Sw/(K1 + 4/3 mu) + (1-Sw)/(K2 + 4/3 mu),
    with K1, K2 the fully water- and fully gas-saturated bulk moduli.
    """
    m = 4.0 / 3.0 * mu
    inv = sw / (k1 + m) + (1.0 - sw) / (k2 + m)
    return 1.0 / inv - m


# ---------------------------------------------- Eq. 6: G-B-P blend ------

def gbp_blend(sw, k_patchy, k_uniform, f=2.0):
    """Gassmann-Brie-Patchy blend  K = (Kpat - Kuni)*Sw^f + Kuni  (Eq. 6)."""
    return (k_patchy - k_uniform) * np.asarray(sw, float) ** f + k_uniform


# ---------------------------------------------- velocity ----------------

def p_velocity(k_gpa, mu_gpa, rho_gcc):
    """P-wave velocity  Vp = sqrt((K + 4/3 mu)/rho).  GPa, g/cc -> m/s."""
    K = np.asarray(k_gpa, float) * 1e9
    mu = mu_gpa * 1e9
    rho = np.asarray(rho_gcc, float) * 1e3
    return np.sqrt((K + 4.0 / 3.0 * mu) / rho)


def saturated_density(phi, sw, rho_grain):
    """Bulk density  rho = rho_grain*(1-phi) + (Sw*rho_w + (1-Sw)*rho_g)*phi."""
    rho_fl = sw * RHO_WATER + (1.0 - sw) * RHO_GAS
    return rho_grain * (1.0 - phi) + rho_fl * phi


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Volcanic-Rock Saturation Model (acoustic)")
    print("=" * 60)

    # Representative tight basalt
    k0, mu, phi, rho_grain = 70.0, 30.0, 0.10, 2.9
    k_dry = 20.0
    sw = np.linspace(0.0, 1.0, 21)

    # Brie fluid modulus lies between Reuss and Voigt bounds (at Sw=0.5)
    kfl_b = brie(0.5)
    assert wood_lindsay(0.5) < kfl_b < domenico(0.5)
    print(f"  Kfl(0.5) Reuss/Brie/Voigt = {wood_lindsay(0.5):.4f} / "
          f"{kfl_b:.4f} / {domenico(0.5):.4f} GPa")

    # Uniform (Gassmann-Brie) and patchy (White) modulus curves
    k_uniform = gassmann_ksat(k_dry, k0, brie(sw), phi)
    k1 = gassmann_ksat(k_dry, k0, K_WATER, phi)     # full water
    k2 = gassmann_ksat(k_dry, k0, K_GAS, phi)       # full gas
    k_patchy = white_patchy(sw, k1, k2, mu)
    k_gbp = gbp_blend(sw, k_patchy, k_uniform, f=2.0)

    rho = saturated_density(phi, sw, rho_grain)
    vp_uniform = p_velocity(k_uniform, mu, rho)
    vp_patchy = p_velocity(k_patchy, mu, rho)
    vp_gbp = p_velocity(k_gbp, mu, rho)

    # Patchy is the upper velocity bound; uniform the lower; G-B-P between
    interior = slice(1, -1)
    assert np.all(vp_patchy[interior] >= vp_uniform[interior] - 1.0)
    assert np.all(vp_gbp[interior] >= vp_uniform[interior] - 1.0)
    assert np.all(vp_gbp[interior] <= vp_patchy[interior] + 1.0)
    print(f"  Vp uniform/patchy @Sw=0.5 = {vp_uniform[10]:.0f} / "
          f"{vp_patchy[10]:.0f} m/s")

    # End members converge (Sw=0 and Sw=1)
    assert abs(vp_patchy[0] - vp_uniform[0]) < 1.0
    assert abs(vp_patchy[-1] - vp_uniform[-1]) < 1.0
    print(f"  Vp at Sw=0 / Sw=1      = {vp_gbp[0]:.0f} / {vp_gbp[-1]:.0f} m/s")
    print("  PASS")
    return {"vp_uniform_50": float(vp_uniform[10]),
            "vp_patchy_50": float(vp_patchy[10])}


if __name__ == "__main__":
    test_all()
