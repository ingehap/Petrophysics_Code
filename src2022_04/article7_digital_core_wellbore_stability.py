"""
Article 7: Application of Digital Core Technology in Wellbore Stability
Research
Zhou, Ye, Zhu, Cheng, Song, Wang, Cai (2022)
DOI: 10.30632/PJV63N2-2022a7

Body text was not present in the available PDF extract, so this module
is a *methodology proxy* guided by the editor's letter: reconstructs
rock skeleton and mineral facies from drill-cuttings via digital-core
CT scanning; computes rock-mechanics properties for wellbore-stability
analysis; evaluates drilling-fluid (water-immersion) effects on rock
strength; benchmarks against triaxial-compression and log-based values.

Implements:

  - Synthetic voxel sand-pack with two mineral phases.
  - Per-voxel Voigt-Reuss-Hill elastic-modulus assignment.
  - Static-equivalent Young's modulus E from K, G:
        E = 9 K G / (3 K + G)
  - Uniaxial compressive strength (UCS) estimator from Plumb-Allen-
    style empirical correlation:
        UCS [MPa] = a * E_GPa - b * phi_pct
  - Water-immersion weakening factor: tau-day exposure reduces UCS by
        UCS(tau) = UCS_dry * exp(-tau / tau_decay)
  - Mohr-Coulomb wellbore-stability check at a given mud-weight ECD
    against minimum / maximum horizontal stresses.
"""

import numpy as np


# ---------------------------------------------- digital-core builder ----

def build_voxel_core(size=48, frac_quartz=0.75, seed=0):
    """0 = pore, 1 = quartz, 2 = clay."""
    rng = np.random.default_rng(seed)
    cube = rng.choice([0, 1, 2], size=(size, size, size),
                      p=[0.18, frac_quartz * 0.82, (1.0 - frac_quartz) * 0.82])
    return cube


def voxel_porosity(cube):
    return float((cube == 0).sum() / cube.size)


def vrh_solid(frac_q, K_q=37.0, G_q=44.0, K_c=21.0, G_c=7.0):
    """VRH average of two minerals."""
    f_c = 1.0 - frac_q
    K_v = frac_q * K_q + f_c * K_c
    G_v = frac_q * G_q + f_c * G_c
    K_r = 1.0 / (frac_q / K_q + f_c / K_c)
    G_r = 1.0 / (frac_q / G_q + f_c / G_c)
    return 0.5 * (K_v + K_r), 0.5 * (G_v + G_r)


def youngs_modulus(K_GPa, G_GPa):
    return 9.0 * K_GPa * G_GPa / (3.0 * K_GPa + G_GPa)


# ---------------------------------------------- UCS predictor ---------

def ucs_plumb_allen(E_GPa, phi_pct, a=2.5, b=0.20):
    return max(0.0, a * E_GPa - b * phi_pct)


# ---------------------------------------------- water-immersion -------

def ucs_water_immersion(ucs_dry, t_days, tau_decay_days=14.0,
                       floor=0.3):
    """UCS_wet(t) = UCS_dry * (floor + (1 - floor) * exp(-t / tau))."""
    return ucs_dry * (floor + (1.0 - floor) * np.exp(-t_days / tau_decay_days))


# ---------------------------------------------- Mohr-Coulomb -----

def mohr_coulomb_critical_mw(sigma_h_MPa, sigma_H_MPa, ucs_MPa,
                             pore_p_MPa, mu_friction=0.6):
    """Return the critical mud-weight ECD (MPa) at which the wellbore
    wall reaches the Mohr-Coulomb shear envelope.

    Uses the Kirsch-stress formulation for a vertical well in
    isotropic horizontal stresses:
        sigma_theta_max = 3 sigma_H - sigma_h - P_mud
    Failure occurs when (sigma_theta_max - pore_p)
                       >= ucs + 2 mu sigma_n.
    """
    sigma_theta_max = 3.0 * sigma_H_MPa - sigma_h_MPa
    # Linear closed form for P_mud at failure threshold
    P_mud_crit = sigma_theta_max - ucs_MPa - pore_p_MPa
    return float(max(P_mud_crit, 0.0))


# ---------------------------------------------- tests ------------

def test_all():
    print("=" * 60)
    print("Article 7: Digital-Core Wellbore Stability (proxy)")
    print("=" * 60)

    cube = build_voxel_core(size=48, frac_quartz=0.80)
    phi = voxel_porosity(cube)
    print(f"  Voxel-core porosity                = {phi:.3f}")

    K_min, G_min = vrh_solid(0.80)
    E_min = youngs_modulus(K_min, G_min)
    print(f"  VRH solid moduli  K = {K_min:.1f} GPa,  G = {G_min:.1f} GPa,  "
          f"E = {E_min:.1f} GPa")

    # Naive porosity-softening for the static modulus (Krief-like)
    soften = (1.0 - phi) ** 3
    E_dry = E_min * soften
    ucs_dry = ucs_plumb_allen(E_dry, phi * 100.0)
    print(f"  E_dry (Krief soften)               = {E_dry:.2f} GPa")
    print(f"  UCS dry                            = {ucs_dry:.2f} MPa")

    # Water-immersion weakening at 3 and 14 days
    for t in (3, 7, 14):
        u = ucs_water_immersion(ucs_dry, t)
        print(f"  UCS after {t:2d} days water immersion  = {u:.2f} MPa")

    # Critical mud weight
    P_mud_crit = mohr_coulomb_critical_mw(sigma_h_MPa=30.0,
                                         sigma_H_MPa=45.0,
                                         ucs_MPa=ucs_dry,
                                         pore_p_MPa=20.0)
    print(f"  Critical mud pressure (Kirsch + MC) = {P_mud_crit:.2f} MPa")
    assert ucs_dry > 0.0 and E_dry > 0.0
    assert ucs_water_immersion(ucs_dry, 30) < ucs_dry, \
        "Water-immersion must reduce UCS"
    print("  PASS")
    return {"phi": phi, "E_GPa": E_dry, "UCS_dry_MPa": ucs_dry,
            "P_mud_crit_MPa": P_mud_crit}


if __name__ == "__main__":
    test_all()
