"""
Article 7: Effect of Fluids on the Elastic Properties of 3D-Printed
           Anisotropic Rock Models
Dande, Stewart, Dyaur (2021)
DOI: 10.30632/PJV62N5-2021a7

Two 3D-printed (FDM) 1-in. cube rock analogs - a layered (VTI) model and a
layered model with embedded penny-shaped inclusions - are saturated with air,
water, oils and glycerol while measuring ultrasonic P- and two S-wave
velocities along x/y/z.  Saturation raises Vp (up to ~57% for glycerol),
collapses P-wave anisotropy epsilon, and produces shear-wave splitting.

Implements:

  - Saturated bulk density  rho = rho_m(1-phi) + rho_f*phi        (Eq. 1)
  - Velocity from traveltime  V = L / t
  - Isotropic moduli K, G, E, nu, M from Vp, Vs, rho
  - Thomsen (1986) anisotropy parameters epsilon, gamma
  - Gassmann fluid substitution (isotropic form)
  - Vp/Vs ratio and acoustic impedance

Note: only Eq. 1 is transcribed in the paper; the velocity/modulus/Thomsen/
Gassmann forms are standard reconstructions (Thomsen 1986, Gassmann 1951,
Russell 2003), flagged as such.  Velocities m/s, density kg/m^3, moduli Pa.
"""

import numpy as np


# ---------------------------------------------- Eq. 1: density ----------

def saturated_density(rho_matrix, phi, rho_fluid):
    """Saturated bulk density  rho = rho_m*(1-phi) + rho_f*phi  (Eq. 1)."""
    return rho_matrix * (1.0 - phi) + rho_fluid * phi


# ---------------------------------------------- velocities & moduli -----

def velocity(length, traveltime):
    """Velocity from first-break traveltime  V = L / t."""
    return length / traveltime


def shear_modulus(rho, vs):
    """G = rho * Vs^2."""
    return rho * vs ** 2


def bulk_modulus(rho, vp, vs):
    """K = rho * (Vp^2 - 4/3 Vs^2)."""
    return rho * (vp ** 2 - 4.0 / 3.0 * vs ** 2)


def youngs_modulus(rho, vp, vs):
    """E = rho * Vs^2 * (3 Vp^2 - 4 Vs^2)/(Vp^2 - Vs^2)."""
    return rho * vs ** 2 * (3 * vp ** 2 - 4 * vs ** 2) / (vp ** 2 - vs ** 2)


def poisson_ratio(vp, vs):
    """nu = (Vp^2 - 2 Vs^2) / (2 (Vp^2 - Vs^2))."""
    return (vp ** 2 - 2 * vs ** 2) / (2.0 * (vp ** 2 - vs ** 2))


def acoustic_impedance(rho, v):
    """Acoustic impedance  Z = rho * v."""
    return rho * v


# ---------------------------------------------- Thomsen parameters ------

def thomsen_epsilon(vp_parallel, vp_perp):
    """P-wave anisotropy  epsilon = (Vp90^2 - Vp0^2)/(2 Vp0^2)."""
    return (vp_parallel ** 2 - vp_perp ** 2) / (2.0 * vp_perp ** 2)


def thomsen_gamma(vs_fast, vs_slow):
    """S-wave anisotropy (splitting)  gamma = (Vs1^2 - Vs2^2)/(2 Vs2^2)."""
    return (vs_fast ** 2 - vs_slow ** 2) / (2.0 * vs_slow ** 2)


# ---------------------------------------------- Gassmann ----------------

def gassmann_ksat(k_dry, k_mineral, k_fluid, phi):
    """Gassmann saturated bulk modulus (shear unchanged)."""
    num = (1.0 - k_dry / k_mineral) ** 2
    den = phi / k_fluid + (1.0 - phi) / k_mineral - k_dry / k_mineral ** 2
    return k_dry + num / den


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Fluids in 3D-Printed Anisotropic Rock Models")
    print("=" * 60)

    # Saturated density (Eq. 1): inclusion model, water
    rho = saturated_density(1.04, phi=0.24, rho_fluid=1.0)   # g/cc
    print(f"  saturated density      = {rho:.4f} g/cc")
    assert abs(rho - 1.0304) < 1e-4

    # Moduli from velocities (SI): rho=790, Vp=2000, Vs=700
    rho_si, vp, vs = 790.0, 2000.0, 700.0
    G = shear_modulus(rho_si, vs) / 1e9
    K = bulk_modulus(rho_si, vp, vs) / 1e9
    nu = poisson_ratio(vp, vs)
    vpvs = vp / vs
    print(f"  G / K                  = {G:.3f} / {K:.3f} GPa")
    print(f"  Poisson / Vp:Vs        = {nu:.3f} / {vpvs:.3f}")
    assert abs(G - 0.387) < 1e-2
    assert abs(K - 2.654) < 0.02
    assert abs(nu - 0.432) < 1e-2
    assert abs(vpvs - 2.857) < 1e-2

    # Velocity from traveltime
    assert abs(velocity(0.0254, 0.0254 / 2000.0) - 2000.0) < 1e-6

    # Thomsen epsilon collapses as fluid replaces air (0.26 -> ~0.01)
    eps_air = thomsen_epsilon(vp_parallel=2760.0, vp_perp=2240.0)   # ~0.26
    print(f"  Thomsen epsilon (air)  = {eps_air:.2f}")
    assert eps_air > 0.20
    gam = thomsen_gamma(vs_fast=820.0, vs_slow=720.0)
    assert gam > 0

    # Gassmann: a stiffer fluid raises K_sat
    k_water = gassmann_ksat(k_dry=2.3e9, k_mineral=5.0e9, k_fluid=2.25e9, phi=0.24)
    k_air = gassmann_ksat(k_dry=2.3e9, k_mineral=5.0e9, k_fluid=1.4e5, phi=0.24)
    print(f"  K_sat air / water      = {k_air/1e9:.2f} / {k_water/1e9:.2f} GPa")
    assert k_water > k_air
    print("  PASS")
    return {"rho_sat": rho, "G_GPa": G, "K_GPa": K, "poisson": nu,
            "epsilon_air": eps_air}


if __name__ == "__main__":
    test_all()
