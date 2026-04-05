#!/usr/bin/env python3
"""
Article 1: Log Interpretation for Petrophysical and Elastic Properties
       of Fine-Grained Sedimentary Rocks
Authors: Ermis Proestakis and Ida Lykke Fabricius
Ref: Petrophysics, Vol. 66, No. 5 (October 2025), pp. 705-727.
     DOI: 10.30632/PJV66N5-2025a1

Implements:
  - Kozeny's permeability equation with porosity-dependent Kozeny factor
  - Archie's formation factor and m-exponent from surface area
  - Parallel conduction model (bulk + bound water) for formation conductivity
  - Free-water and bound-water (adsorbed layer) porosity partitioning
  - Irreducible water saturation and water saturation in hydrocarbon zones
  - Iso-frame elastic model (Fabricius 2003) with Gassmann fluid substitution
  - Biot's coefficient and vertical elastic strain
"""

import numpy as np


# ---------------------------------------------------------------------------
# Kozeny permeability model (Eqs. 1-3)
# ---------------------------------------------------------------------------

def kozeny_factor(porosity):
    """Kozeny factor c as a function of porosity (Mortensen et al., 1998, Eq. 2)."""
    phi = np.asarray(porosity, dtype=float)
    return 0.5 * (1.0 + phi * (1.0 + 2.0 * (1.0 - phi)))


def kozeny_permeability(porosity, s_bulk):
    """Permeability from Kozeny's equation (Eq. 1).

    Parameters
    ----------
    porosity : float or array  – total porosity (fraction)
    s_bulk   : float or array  – specific surface area per bulk volume (1/m)

    Returns
    -------
    k : permeability in m^2
    """
    phi = np.asarray(porosity, dtype=float)
    sb = np.asarray(s_bulk, dtype=float)
    c = kozeny_factor(phi)
    k = phi ** 3 / (c * sb ** 2)
    return k


def kozeny_surface_area(porosity, permeability):
    """Kozeny surface area S_KZ (Eq. 3)."""
    phi = np.asarray(porosity, dtype=float)
    k = np.asarray(permeability, dtype=float)
    c = kozeny_factor(phi)
    s_kz = np.sqrt(phi ** 3 / (c * k))
    return s_kz


# ---------------------------------------------------------------------------
# Archie's m-exponent from Kozeny surface area (Eq. 21-22)
# ---------------------------------------------------------------------------

def archie_m_from_skz(s_kz):
    """Archie's m-exponent from Kozeny surface area (Eq. 21).
    m = 1.50 + 0.248 * ln(S_KZ)
    """
    return 1.50 + 0.248 * np.log(np.asarray(s_kz, dtype=float))


def skz_from_archie_m(m):
    """Inverse of Eq. 21 -> Eq. 22: S_KZ from m."""
    return np.exp((np.asarray(m, dtype=float) - 1.50) / 0.248)


# ---------------------------------------------------------------------------
# Formation conductivity – parallel conduction model (Eqs. 4-7)
# ---------------------------------------------------------------------------

def archie_formation_factor(porosity, m):
    """Classic Archie formation factor F = phi^{-m} (Eq. 4)."""
    return np.asarray(porosity, dtype=float) ** (-np.asarray(m, dtype=float))


def free_water_porosity(porosity, sigma_o, sigma_w, p=1.5):
    """Free-water porosity from Eq. 6.
    phi_w = (sigma_o / sigma_w)^{1/p}
    """
    return (np.asarray(sigma_o, dtype=float)
            / np.asarray(sigma_w, dtype=float)) ** (1.0 / p)


def bound_water_porosity(porosity, phi_w):
    """Bound-water porosity (adsorbed layer) from Eq. 7.
    phi_AL = phi - phi_w
    """
    return np.asarray(porosity, dtype=float) - np.asarray(phi_w, dtype=float)


def formation_conductivity(sigma_w, phi_w, sigma_al, phi_al, p=1.5):
    """Parallel conduction model (Eq. 5).
    sigma_o = sigma_w * phi_w^p + sigma_AL * phi_AL^p
    """
    sw = np.asarray(sigma_w, dtype=float)
    pw = np.asarray(phi_w, dtype=float)
    sa = np.asarray(sigma_al, dtype=float)
    pa = np.asarray(phi_al, dtype=float)
    return sw * pw ** p + sa * pa ** p


# ---------------------------------------------------------------------------
# Irreducible water saturation (Eq. 24)
# ---------------------------------------------------------------------------

def irreducible_water_saturation(phi_al, porosity):
    """S_ir = phi_AL / phi  (Eq. 24)."""
    return np.asarray(phi_al, dtype=float) / np.asarray(porosity, dtype=float)


# ---------------------------------------------------------------------------
# Water saturation in hydrocarbon zones (Eqs. 25-26)
# ---------------------------------------------------------------------------

def water_saturation_hc(sigma_t, sigma_w, sigma_al, phi_al, porosity, p=1.5):
    """Water saturation in a hydrocarbon-bearing zone (Eqs. 25-26).

    Solves:  sigma_t = sigma_w * phi_w^p + sigma_AL * phi_AL^p
    for phi_w, then  Sw = (phi_w + phi_AL) / porosity.
    """
    st = np.asarray(sigma_t, dtype=float)
    sw = np.asarray(sigma_w, dtype=float)
    sa = np.asarray(sigma_al, dtype=float)
    pa = np.asarray(phi_al, dtype=float)
    phi = np.asarray(porosity, dtype=float)
    # phi_w from Eq. 26
    numerator = st - sa * pa ** p
    numerator = np.maximum(numerator, 0.0)
    phi_w = (numerator / sw) ** (1.0 / p)
    s_w = (phi_w + pa) / phi
    return np.clip(s_w, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Elastic properties – Iso-frame model (Eqs. 11-20)
# ---------------------------------------------------------------------------

def gassmann_bulk_modulus(k_dry, k_s, k_fl, porosity):
    """Gassmann saturated bulk modulus (Eq. 11)."""
    phi = np.asarray(porosity, dtype=float)
    kd = np.asarray(k_dry, dtype=float)
    ks = np.asarray(k_s, dtype=float)
    kf = np.asarray(k_fl, dtype=float)
    num = (1.0 - kd / ks) ** 2
    den = phi / kf + (1.0 - phi) / ks - kd / ks ** 2
    return kd + num / den


def suspension_modulus(k_fl, k_s, g_s, porosity):
    """Lower Hashin-Shtrikman bound = suspension modulus K_sus (Eq. 18)."""
    phi = np.asarray(porosity, dtype=float)
    kf = np.asarray(k_fl, dtype=float)
    ks = np.asarray(k_s, dtype=float)
    gs = np.asarray(g_s, dtype=float)
    k_sus = kf + (1.0 - phi) / (1.0 / (ks - kf) + phi / (kf + 4.0 * gs / 3.0))
    return k_sus


def zeta(k_s, g_s, k_sus):
    """Helper zeta parameter (Eq. 19)."""
    return (g_s / 6.0) * (9.0 * k_s + 8.0 * g_s) / (k_s + 2.0 * g_s)


def isoframe_bulk_modulus(k_s, g_s, k_fl, porosity, iso_frame):
    """Iso-frame bulk modulus (Eq. 16).

    iso_frame : fraction of solid forming shells [0, 1].
    """
    phi = np.asarray(porosity, dtype=float)
    IF = np.asarray(iso_frame, dtype=float)
    k_sus = suspension_modulus(k_fl, k_s, g_s, phi)
    z = zeta(k_s, g_s, k_sus)
    phi_eff = 1.0 - IF * (1.0 - phi)  # effective "fluid" fraction
    k_if = k_sus + (1.0 - phi_eff) / (
        1.0 / (k_s - k_sus) + phi_eff / (k_sus + 4.0 * g_s / 3.0)
    )
    return k_if


def isoframe_shear_modulus(k_s, g_s, k_fl, porosity, iso_frame):
    """Iso-frame shear modulus (Eq. 17)."""
    phi = np.asarray(porosity, dtype=float)
    IF = np.asarray(iso_frame, dtype=float)
    k_sus = suspension_modulus(k_fl, k_s, g_s, phi)
    z = zeta(k_s, g_s, k_sus)
    phi_eff = 1.0 - IF * (1.0 - phi)
    g_if = 0.0 + (1.0 - phi_eff) / (1.0 / g_s + phi_eff * 6.0 / (5.0 * g_s)
                                      * (k_sus + 2.0 * g_s)
                                      / (3.0 * k_sus + 4.0 * g_s + 1e-30))
    return np.maximum(g_if, 0.0)


def pwave_modulus(k, g):
    """P-wave modulus M = K + 4G/3  (Eq. 20)."""
    return np.asarray(k, dtype=float) + 4.0 / 3.0 * np.asarray(g, dtype=float)


def biot_coefficient(k_dry, k_s):
    """Biot's coefficient alpha = 1 - K_dry / K_s  (Eq. 10)."""
    return 1.0 - np.asarray(k_dry, dtype=float) / np.asarray(k_s, dtype=float)


def vertical_elastic_strain(sigma_total, pore_pressure, m_dry, alpha):
    """Vertical elastic strain epsilon (Eqs. 8-9).
    sigma' = sigma_total - alpha * P
    epsilon = sigma' / M_dry
    """
    sp = np.asarray(sigma_total, dtype=float) - np.asarray(alpha, dtype=float) * np.asarray(pore_pressure, dtype=float)
    return sp / np.asarray(m_dry, dtype=float)


def overburden_stress(depth, rho_avg=1980.0, g=9.81):
    """Total overburden stress (Eq. 28): sigma = rho_avg * g * depth."""
    return rho_avg * g * np.asarray(depth, dtype=float)


# ---------------------------------------------------------------------------
# Arps temperature correction for conductivity (Eq. 23)
# ---------------------------------------------------------------------------

def arps_temperature_correction(sigma_1, t1, t2):
    """Convert conductivity from temperature T1 to T2 (Arps, 1953, Eq. 23).
    sigma_2 = sigma_1 * (T2 + 21.5) / (T1 + 21.5)   [T in Celsius]
    """
    return sigma_1 * (t2 + 21.5) / (t1 + 21.5)


# ---------------------------------------------------------------------------
# Main / demo
# ---------------------------------------------------------------------------

def demo():
    """Run a short demonstration of the key functions."""
    print("=== Article 1: Log Interpretation Demo ===\n")

    # Synthetic well-log data
    depth = np.linspace(1500, 1700, 20)
    porosity = np.linspace(0.35, 0.20, 20)
    s_bulk = np.linspace(5e4, 2e5, 20)

    k = kozeny_permeability(porosity, s_bulk)
    print(f"Porosity range     : {porosity.min():.2f} – {porosity.max():.2f}")
    print(f"Permeability range : {k.min():.2e} – {k.max():.2e} m^2")

    s_kz = kozeny_surface_area(porosity, k)
    m = archie_m_from_skz(s_kz)
    print(f"Archie m range     : {m.min():.2f} – {m.max():.2f}")

    # Elastic properties
    k_s, g_s = 70e9, 30e9   # calcite-like (GPa)
    k_fl = 2.3e9
    IF = 0.5
    k_if = isoframe_bulk_modulus(k_s, g_s, k_fl, porosity, IF)
    g_if = isoframe_shear_modulus(k_s, g_s, k_fl, porosity, IF)
    m_sat = pwave_modulus(k_if, g_if)
    print(f"Iso-frame Msat     : {m_sat.min()/1e9:.1f} – {m_sat.max()/1e9:.1f} GPa")

    alpha = biot_coefficient(k_if * 0.6, k_s)
    sigma_ov = overburden_stress(depth)
    pp = 0.45 * sigma_ov
    eps = vertical_elastic_strain(sigma_ov, pp, m_sat * 0.6, alpha)
    print(f"Elastic strain     : {eps.min():.4f} – {eps.max():.4f}")
    print()


if __name__ == "__main__":
    demo()
