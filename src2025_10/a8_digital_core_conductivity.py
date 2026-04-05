#!/usr/bin/env python3
"""
Article 8: Digital-Core-Based Numerical Simulation of Conductivity in
           Low-Permeability Sandstone Reservoirs
Authors: Guoqing Feng and Junhui Zou
Ref: Petrophysics, Vol. 66, No. 5 (October 2025), pp. 858-871.
     DOI: 10.30632/PJV66N5-2025a8

Implements:
  - Archie's first law: F = a * phi^{-m}  with directional anisotropy
  - Archie's second law: I = R_t / R_0 = Sw^{-n}  with bimodal behaviour
  - Bimodal saturation exponent model (breakpoint at ~45% Sw)
  - Wettability effect on saturation exponent
  - Salinity influence on formation water resistivity
  - Simple 3-D resistivity simulation on a digital core grid
"""

import numpy as np


# ---------------------------------------------------------------------------
# Archie's laws (Eqs. 3-4 in article, classical form)
# ---------------------------------------------------------------------------

def formation_factor(porosity, a=1.0, m=2.0):
    """Archie's first law: F = a * phi^{-m} (Eq. 3)."""
    phi = np.maximum(np.asarray(porosity, dtype=float), 1e-6)
    return a * phi ** (-m)


def resistivity_index(sw, n=2.0):
    """Archie's second law: I = Sw^{-n} (Eq. 4)."""
    sw = np.clip(np.asarray(sw, dtype=float), 0.01, 1.0)
    return sw ** (-n)


def true_resistivity(rw, F, I):
    """Rt = Rw * F * I."""
    return rw * F * I


# ---------------------------------------------------------------------------
# Directional Archie parameters (from article results)
# ---------------------------------------------------------------------------

ARCHIE_PARAMS = {
    'x': {'a': 0.8226, 'm': 2.428},
    'y': {'a': 0.8112, 'm': 2.346},
    'z': {'a': 1.313,  'm': 2.164},
}


def formation_factor_directional(porosity, direction='x'):
    """Directional formation factor using article-calibrated parameters."""
    p = ARCHIE_PARAMS[direction]
    return formation_factor(porosity, a=p['a'], m=p['m'])


# ---------------------------------------------------------------------------
# Bimodal saturation exponent (breakpoint at Sw ~ 0.45)
# ---------------------------------------------------------------------------

def bimodal_saturation_exponent(sw, n_high=2.0, n_low=3.5, sw_break=0.45):
    """Returns n as a function of Sw with bimodal behaviour.
    Below sw_break the exponent is higher (steeper I-Sw).
    """
    sw = np.asarray(sw, dtype=float)
    n = np.where(sw >= sw_break, n_high, n_low)
    return n


def resistivity_index_bimodal(sw, n_high=2.0, n_low=3.5, sw_break=0.45):
    """I with bimodal n."""
    n = bimodal_saturation_exponent(sw, n_high, n_low, sw_break)
    sw = np.clip(sw, 0.01, 1.0)
    return sw ** (-n)


# ---------------------------------------------------------------------------
# Wettability effect
# ---------------------------------------------------------------------------

def saturation_exponent_wettability(contact_angle_deg, n_water_wet=2.0,
                                     n_oil_wet=4.0):
    """Interpolate n based on contact angle.
    60° (water-wet) -> n_water_wet; 120° (oil-wet) -> n_oil_wet.
    """
    theta = np.clip(contact_angle_deg, 0, 180)
    frac = (theta - 60.0) / 60.0
    frac = np.clip(frac, 0, 1)
    return n_water_wet + frac * (n_oil_wet - n_water_wet)


# ---------------------------------------------------------------------------
# Formation water resistivity from salinity
# ---------------------------------------------------------------------------

def rw_from_salinity(salinity_ppm, temperature_c=25.0):
    """Approximate Rw (ohm·m) from NaCl salinity and temperature.
    Uses Arps-like empirical relation.
    """
    sal = np.maximum(salinity_ppm, 1.0)
    rw_25 = 1.0 / (0.0123 + 0.0000364 * sal)
    rw = rw_25 * (25.0 + 21.5) / (temperature_c + 21.5)
    return rw


# ---------------------------------------------------------------------------
# Simple 3-D digital core resistivity simulation
# ---------------------------------------------------------------------------

def generate_digital_core(nx=50, ny=50, nz=50, porosity_target=0.15,
                           seed=42):
    """Generate a binary 3-D digital core (0=pore, 1=matrix).

    Returns
    -------
    core : (nx, ny, nz) int array
    actual_porosity : float
    """
    rng = np.random.default_rng(seed)
    core = (rng.random((nx, ny, nz)) > porosity_target).astype(int)
    actual = 1.0 - core.mean()
    return core, actual


def simulate_resistivity_1d(core_slice, rw, r_matrix=1e6):
    """Simulate effective resistivity of a 2-D slice along x-direction.

    Uses simple series-parallel resistor network.
    core_slice: 2-D binary (ny, nz), 0=pore (Rw), 1=matrix (R_matrix).
    """
    ny, nz = core_slice.shape
    r_col = np.zeros(nz)
    for j in range(nz):
        col = core_slice[:, j]
        # series: each cell is rw or r_matrix
        r_series = np.where(col == 0, rw, r_matrix).sum() / ny
        r_col[j] = r_series
    # parallel combination across z columns
    r_eff = 1.0 / np.sum(1.0 / r_col) * nz
    return r_eff


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== Article 8: Digital Core Conductivity Demo ===\n")

    porosity = np.linspace(0.05, 0.35, 20)
    for d in ['x', 'y', 'z']:
        F = formation_factor_directional(porosity, d)
        print(f"F ({d}-axis) at phi=0.10 : {F[np.argmin(np.abs(porosity-0.10))]:.1f}")

    sw = np.linspace(0.1, 1.0, 50)
    I_std = resistivity_index(sw, n=2.0)
    I_bi = resistivity_index_bimodal(sw)
    print(f"\nI at Sw=0.30 (standard n=2) : {I_std[np.argmin(np.abs(sw-0.30))]:.2f}")
    print(f"I at Sw=0.30 (bimodal)      : {I_bi[np.argmin(np.abs(sw-0.30))]:.2f}")

    # Digital core
    core, phi_actual = generate_digital_core(30, 30, 30, 0.12)
    rw = rw_from_salinity(50000, 80)
    r_eff = simulate_resistivity_1d(core[15, :, :], rw)
    print(f"\nDigital core porosity  : {phi_actual:.3f}")
    print(f"Rw at 50k ppm, 80°C   : {rw:.3f} ohm·m")
    print(f"Effective resistivity  : {r_eff:.2f} ohm·m")
    print()


if __name__ == "__main__":
    demo()
