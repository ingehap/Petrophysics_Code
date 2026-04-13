"""
article_01_laronga_ccs_evaluation.py
====================================
Implementation of ideas from:

    Laronga, R., Borchardt, E., Hill, B., Velez, E., Klemin, D., Haddad, S.,
    Haddad, E., Chadwick, C., Mahmoodaghdam, E., and Hamichi, F. (2023).
    "Integrated Formation Evaluation for Site-Specific Evaluation,
    Optimization, and Permitting of Carbon Storage Projects."
    Petrophysics, 64(5), 580-620. DOI: 10.30632/PJV64N5-2023a1

The paper reviews the petrophysics of CCS site evaluation around three
"pillars":
    1. Storage capacity (effective pore volume available to CO2)
    2. Containment (caprock seal capacity / capillary entry pressure)
    3. Injectivity (rate of injection that can be sustained)

This module implements simple, transparent quantitative estimates for
each pillar using standard equations from the CCS / petrophysics
literature cited in the paper.

Run as a standalone script:  ``python article_01_laronga_ccs_evaluation.py``
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# CO2 properties (very simple correlations sufficient for screening)
# ---------------------------------------------------------------------------
def co2_density(p_pa: float, T_K: float) -> float:
    """Approximate supercritical CO2 density (kg/m3) using a simple cubic-EoS
    style correlation valid for ~50-300 bar / 300-400 K.  Adequate for
    screening; not for design."""
    # Fit to NIST tables: rho(p,T) ~ a*p / (R*T) with non-ideal correction
    R = 188.92  # J/(kg.K) for CO2
    Z = 0.95 - 1.6e-8 * p_pa + 0.0009 * (T_K - 320.0)
    Z = max(0.25, min(Z, 1.05))
    return p_pa / (Z * R * T_K)


def brine_density(salinity_ppm: float, T_K: float) -> float:
    """Brine density (kg/m3) - Batzle & Wang (1992) simplified."""
    S = salinity_ppm * 1e-6
    T_C = T_K - 273.15
    rho_w = 1000.0 * (1.0 - (T_C - 4.0) ** 2 / 178000.0)
    return rho_w + S * (300.0 - 2.0 * T_C - 0.6 * S * 100.0)


# ---------------------------------------------------------------------------
# Pillar 1 - Storage capacity
# ---------------------------------------------------------------------------
def storage_capacity_mass(area_m2: float,
                          thickness_m: float,
                          porosity: float,
                          net_to_gross: float,
                          efficiency: float,
                          rho_co2: float) -> float:
    """Effective CO2 storage mass (kg) for a saline aquifer.

    M_CO2 = A * h * NTG * phi * E * rho_CO2

    Equation (1) of the US-DOE/NETL methodology referenced by Laronga et al.
    """
    return area_m2 * thickness_m * net_to_gross * porosity * efficiency * rho_co2


# ---------------------------------------------------------------------------
# Pillar 2 - Containment (caprock entry pressure & sealing column height)
# ---------------------------------------------------------------------------
def capillary_entry_pressure(ift_Nm: float,
                             contact_angle_deg: float,
                             pore_throat_radius_m: float) -> float:
    """Young-Laplace capillary entry pressure (Pa).

        Pc = 2 * sigma * cos(theta) / r
    """
    return 2.0 * ift_Nm * math.cos(math.radians(contact_angle_deg)) / pore_throat_radius_m


def max_co2_column_height(pc_caprock_Pa: float,
                          rho_brine: float,
                          rho_co2: float,
                          g: float = 9.81) -> float:
    """Maximum buoyant CO2 column (m) the caprock can hold before breakthrough."""
    drho = rho_brine - rho_co2
    if drho <= 0:
        return 0.0
    return pc_caprock_Pa / (drho * g)


# ---------------------------------------------------------------------------
# Pillar 3 - Injectivity index (simple radial Darcy)
# ---------------------------------------------------------------------------
def injectivity_index(k_m2: float,
                      h_m: float,
                      mu_Pas: float,
                      r_e_m: float,
                      r_w_m: float,
                      skin: float = 0.0) -> float:
    """Steady-state radial injectivity index, II = q / dP  (m3/s/Pa).

        II = 2*pi*k*h / ( mu * (ln(re/rw) + S) )
    """
    return 2.0 * math.pi * k_m2 * h_m / (mu_Pas * (math.log(r_e_m / r_w_m) + skin))


# ---------------------------------------------------------------------------
# Bundling all three pillars
# ---------------------------------------------------------------------------
@dataclass
class CCSScreeningResult:
    storage_mass_Mt: float       # megatonnes CO2
    max_column_m: float          # m of CO2 the seal can hold
    injectivity_kg_s_per_bar: float


def screen_site(area_km2: float, thickness_m: float, porosity: float,
                ntg: float, efficiency: float,
                p_res_bar: float, T_res_K: float,
                ift_mNm: float, contact_angle_deg: float,
                throat_radius_um: float,
                k_mD: float, mu_cp: float,
                r_e_m: float, r_w_m: float,
                salinity_ppm: float = 100_000.0) -> CCSScreeningResult:
    """End-to-end screening assessment for a candidate CCS site."""
    p_pa = p_res_bar * 1e5
    rho_co2 = co2_density(p_pa, T_res_K)
    rho_b = brine_density(salinity_ppm, T_res_K)
    M = storage_capacity_mass(area_km2 * 1e6, thickness_m, porosity, ntg,
                              efficiency, rho_co2) / 1e9      # Mt
    pc = capillary_entry_pressure(ift_mNm * 1e-3, contact_angle_deg,
                                  throat_radius_um * 1e-6)
    h_col = max_co2_column_height(pc, rho_b, rho_co2)
    II = injectivity_index(k_mD * 9.869233e-16, thickness_m, mu_cp * 1e-3,
                           r_e_m, r_w_m)
    II_kg_s_bar = II * 1e5 * rho_co2     # convert m3/s/Pa -> kg/s/bar
    return CCSScreeningResult(M, h_col, II_kg_s_bar)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    rng = np.random.default_rng(0)
    # Generic Gulf-Coast-like saline aquifer
    res = screen_site(area_km2=25.0, thickness_m=80.0, porosity=0.22,
                      ntg=0.7, efficiency=0.05,
                      p_res_bar=180.0, T_res_K=350.0,
                      ift_mNm=30.0, contact_angle_deg=30.0,
                      throat_radius_um=0.05,
                      k_mD=200.0, mu_cp=0.05,
                      r_e_m=1500.0, r_w_m=0.1)
    assert 0.5 < res.storage_mass_Mt < 1000.0, res
    assert res.max_column_m > 100.0
    assert res.injectivity_kg_s_per_bar > 0.0

    # Vary porosity, capacity should scale linearly
    M = [screen_site(25, 80, p, 0.7, 0.05, 180, 350, 30, 30, 0.05,
                     200, 0.05, 1500, 0.1).storage_mass_Mt
         for p in (0.1, 0.2, 0.3)]
    assert abs(M[1] / M[0] - 2.0) < 1e-6 and abs(M[2] / M[0] - 3.0) < 1e-6

    # Random sanity sweep
    for _ in range(20):
        r = screen_site(rng.uniform(1, 100), rng.uniform(20, 200),
                        rng.uniform(0.05, 0.3), rng.uniform(0.3, 1.0),
                        rng.uniform(0.01, 0.1), rng.uniform(80, 300),
                        rng.uniform(310, 400), rng.uniform(20, 40),
                        rng.uniform(20, 60), rng.uniform(0.01, 0.2),
                        rng.uniform(10, 1000), rng.uniform(0.03, 0.1),
                        1500, 0.1)
        assert r.storage_mass_Mt > 0
        assert r.max_column_m >= 0
        assert r.injectivity_kg_s_per_bar > 0
    print("article_01_laronga_ccs_evaluation: OK  ({})".format(res))


if __name__ == "__main__":
    test_all()
