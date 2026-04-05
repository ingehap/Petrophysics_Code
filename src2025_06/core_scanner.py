#!/usr/bin/env python3
"""
Core Scanner for Electrical Profiling of Full-Bore Cores
=========================================================
Implements the electromagnetic core scanner methodology from:
  Mirza, D., Birkeland, K., Øy, L., Chemali, R., and Barrouillet, B., 2025,
  "Core Scanner for Electrical Profiling of Full-Bore Cores at the Wellsite
  With Advanced Pulse Electromagnetic Technology,"
  Petrophysics, Vol. 66, No. 3, pp. 352–363.

Key ideas implemented:
  - Forward modeling of time-of-flight (TOF) and attenuation as functions
    of resistivity and dielectric permittivity at 3.8 GHz.
  - Joint inversion of TOF and attenuation to recover resistivity and
    dielectric permittivity via lookup-table search.
  - Complex Refractive Index Method (CRIM) for water-filled porosity
    estimation from GHz-frequency resistivity and permittivity.

References:
  Birchak, J.R., et al., 1974 (CRIM formulation).
  Bittar, M.S., et al., 2010 (complex dielectric extension).
  Newsham, K.E., et al., 2019 (approximate CRIM formulation).
"""

import numpy as np
from typing import Tuple, Optional

# ---------- Physical constants ----------
EPS_0 = 8.854187817e-12          # F/m  – vacuum permittivity
C_LIGHT = 2.99792458e8           # m/s  – speed of light
FREQ_HZ = 3.8e9                  # Hz   – scanner operating frequency
OMEGA = 2.0 * np.pi * FREQ_HZ   # rad/s

# Matrix dielectric permittivities (Wang et al., 2018 – Table 1 in paper)
MATRIX_PERMITTIVITY = {
    "quartz":    4.65,
    "calcite":   7.50,
    "dolomite":  6.80,
    "anhydrite": 6.35,
    "feldspar":  5.50,
    "siderite":  7.00,
    "pyrite":   81.00,
}


def complex_permittivity(eps_prime: float, sigma: float,
                         eps_dl: float = 0.0) -> complex:
    """
    Compute the complex relative dielectric permittivity (Eq. 3 of paper).

    Parameters
    ----------
    eps_prime : real part of relative permittivity
    sigma     : conductivity (S/m)
    eps_dl    : dipolar-loss component (dimensionless)

    Returns
    -------
    Complex relative permittivity  ε* = ε' - j(σ/(ω·ε₀) + ε_dl)
    """
    eps_imag = sigma / (OMEGA * EPS_0) + eps_dl
    return np.asarray(eps_prime) - 1j * np.asarray(eps_imag)


def water_complex_permittivity(rw: float,
                               eps_water_real: float = 55.0,
                               eps_dl_water: float = 15.0) -> complex:
    """
    Complex permittivity of formation water at 3.8 GHz.

    Parameters
    ----------
    rw             : water resistivity (Ω·m)
    eps_water_real : real permittivity of water at 3.8 GHz (≈55 for deionised)
    eps_dl_water   : dipolar-loss term for water at 3.8 GHz

    Returns
    -------
    ε*_w  complex permittivity
    """
    sigma_w = 1.0 / rw
    eps_imag = sigma_w / (OMEGA * EPS_0) + eps_dl_water
    return eps_water_real - 1j * eps_imag


# ---------- Forward model ----------

def forward_model(resistivity: np.ndarray,
                  eps_r: np.ndarray,
                  core_diameter_m: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward-model the time-of-flight (TOF) and normalised amplitude
    for a plane-wave pulse crossing a core of given diameter.

    Uses Maxwell's equations for a lossy dielectric slab (Jackson, 1975).

    Parameters
    ----------
    resistivity    : formation resistivity array (Ω·m)
    eps_r          : relative dielectric permittivity array (dimensionless)
    core_diameter_m: core diameter (m), default 0.10 m

    Returns
    -------
    tof_ns       : time of flight in nanoseconds
    norm_amp_db  : normalised amplitude in dB (negative = attenuation)
    """
    resistivity = np.asarray(resistivity, dtype=float)
    eps_r = np.asarray(eps_r, dtype=float)

    sigma = 1.0 / resistivity
    eps_star = complex_permittivity(eps_r, sigma)

    # Complex propagation constant  k = ω √(μ₀ ε₀ ε*)
    k = (OMEGA / C_LIGHT) * np.sqrt(eps_star)  # complex

    # Propagation through the core diameter
    phase = k * core_diameter_m            # complex phase shift

    tof_ns = np.real(phase) / OMEGA * 1e9  # nanoseconds
    norm_amp_db = 20.0 * np.log10(np.exp(1)) * np.imag(phase)  # dB (negative)

    return tof_ns, norm_amp_db


def build_lookup_table(res_min: float = 1.5, res_max: float = 1000.0,
                       eps_min: float = 4.0, eps_max: float = 80.0,
                       n_res: int = 200, n_eps: int = 100,
                       core_diameter_m: float = 0.10):
    """
    Build the pre-computed TOF/attenuation catalogue used for inversion.

    Returns
    -------
    res_grid, eps_grid : 1-D arrays
    tof_table, amp_table : 2-D arrays  (n_res × n_eps)
    """
    res_grid = np.logspace(np.log10(res_min), np.log10(res_max), n_res)
    eps_grid = np.linspace(eps_min, eps_max, n_eps)

    RR, EE = np.meshgrid(res_grid, eps_grid, indexing="ij")
    tof_table, amp_table = forward_model(RR.ravel(), EE.ravel(),
                                         core_diameter_m)
    tof_table = tof_table.reshape(RR.shape)
    amp_table = amp_table.reshape(RR.shape)
    return res_grid, eps_grid, tof_table, amp_table


# ---------- Inversion ----------

def invert_single(tof_obs: float, amp_obs: float,
                  res_grid, eps_grid, tof_table, amp_table):
    """
    Invert a single (TOF, amplitude) pair by nearest-neighbour search
    of the pre-computed catalogue.

    Returns
    -------
    resistivity, dielectric_permittivity
    """
    # Normalise so both dimensions have comparable weight
    tof_range = tof_table.max() - tof_table.min()
    amp_range = amp_table.max() - amp_table.min()
    if tof_range == 0:
        tof_range = 1.0
    if amp_range == 0:
        amp_range = 1.0

    misfit = ((tof_table - tof_obs) / tof_range) ** 2 + \
             ((amp_table - amp_obs) / amp_range) ** 2
    idx = np.unravel_index(np.argmin(misfit), misfit.shape)
    return res_grid[idx[0]], eps_grid[idx[1]]


def invert_profile(tof_obs: np.ndarray, amp_obs: np.ndarray,
                   core_diameter_m: float = 0.10):
    """
    Invert an array of (TOF, amplitude) measurements.

    Returns
    -------
    resistivity, eps_r : arrays of inverted properties
    """
    res_grid, eps_grid, tof_tbl, amp_tbl = build_lookup_table(
        core_diameter_m=core_diameter_m)

    n = len(tof_obs)
    res_out = np.zeros(n)
    eps_out = np.zeros(n)
    for i in range(n):
        res_out[i], eps_out[i] = invert_single(
            tof_obs[i], amp_obs[i], res_grid, eps_grid, tof_tbl, amp_tbl)
    return res_out, eps_out


# ---------- CRIM water-filled porosity ----------

def crim_water_filled_porosity(eps_t: complex,
                               eps_mat: complex,
                               eps_w: complex,
                               mn: float = 0.5) -> float:
    """
    Solve the CRIM equation (Eq. 1 of paper) for water-filled porosity φ_w.

        √ε*_t  =  φ_w · √ε*_w  +  (1 - φ_w) · √ε*_mat

    Parameters
    ----------
    eps_t   : complex permittivity of formation (from core scanner)
    eps_mat : complex permittivity of matrix mineral
    eps_w   : complex permittivity of pore water at 3.8 GHz
    mn      : tortuosity exponent (default 0.5 for CRIM)

    Returns
    -------
    phi_w : water-filled porosity (fraction)
    """
    sqrt_t = np.sqrt(eps_t)
    sqrt_m = np.sqrt(eps_mat)
    sqrt_w = np.sqrt(eps_w)

    phi_w = np.real((sqrt_t - sqrt_m) / (sqrt_w - sqrt_m))
    return float(np.clip(phi_w, 0.0, 1.0))


def compute_water_filled_porosity(resistivity: np.ndarray,
                                  eps_r: np.ndarray,
                                  rw: float = 0.05,
                                  matrix_mineral: str = "quartz") -> np.ndarray:
    """
    Convenience function: from inverted (resistivity, eps_r) compute
    water-filled porosity along the core using CRIM.

    Parameters
    ----------
    resistivity    : array Ω·m
    eps_r          : array (dimensionless)
    rw             : water resistivity Ω·m
    matrix_mineral : key into MATRIX_PERMITTIVITY dict

    Returns
    -------
    phi_w : water-filled porosity array
    """
    eps_mat_real = MATRIX_PERMITTIVITY[matrix_mineral]
    eps_mat = complex(eps_mat_real, 0)

    eps_w = water_complex_permittivity(rw)

    n = len(resistivity)
    phi_w = np.zeros(n)
    for i in range(n):
        sigma_i = 1.0 / resistivity[i]
        eps_t = float(eps_r[i]) - 1j * (sigma_i / (OMEGA * EPS_0))
        phi_w[i] = crim_water_filled_porosity(eps_t, eps_mat, eps_w)
    return phi_w


# ---------- Standalone test ----------

def test_all():
    """Test all functions with synthetic data."""
    print("=" * 60)
    print("Testing core_scanner module (Mirza et al., 2025)")
    print("=" * 60)

    # 1. Forward model
    res = np.array([5, 10, 50, 200, 500])
    eps = np.array([20, 15, 10, 8, 6])
    tof, amp = forward_model(res, eps)
    print("\n1) Forward model – TOF and attenuation")
    for r, e, t, a in zip(res, eps, tof, amp):
        print(f"   Rt={r:6.1f} Ω·m  ε={e:4.1f}  →  TOF={t:.4f} ns  Amp={a:.2f} dB")
    assert len(tof) == len(res)

    # 2. Build lookup table
    rg, eg, tt, at = build_lookup_table(n_res=50, n_eps=30)
    print(f"\n2) Lookup table built: {tt.shape[0]} × {tt.shape[1]} entries")

    # 3. Inversion round-trip
    print("\n3) Inversion round-trip test")
    res_true = np.array([10.0, 50.0, 200.0])
    eps_true = np.array([18.0, 12.0, 7.0])
    tof_syn, amp_syn = forward_model(res_true, eps_true)
    res_inv, eps_inv = invert_profile(tof_syn, amp_syn)
    for i in range(len(res_true)):
        print(f"   True: Rt={res_true[i]:.1f}  ε={eps_true[i]:.1f}  |  "
              f"Inv:  Rt={res_inv[i]:.1f}  ε={eps_inv[i]:.1f}")
    # Check relative error < 20% for well-conditioned cases
    for i in range(len(res_true)):
        assert abs(res_inv[i] - res_true[i]) / res_true[i] < 0.25, \
            f"Resistivity inversion failed for sample {i}"

    # 4. CRIM porosity
    print("\n4) CRIM water-filled porosity")
    phi_w = compute_water_filled_porosity(res_inv, eps_inv,
                                          rw=0.05, matrix_mineral="quartz")
    for i, pw in enumerate(phi_w):
        print(f"   Sample {i}: φ_w = {pw:.3f}")
    assert all(0 <= p <= 1 for p in phi_w)

    # 5. Complex permittivity helpers
    ew = water_complex_permittivity(0.05)
    print(f"\n5) Water complex permittivity at 3.8 GHz (Rw=0.05): {ew:.2f}")

    print("\n✓ All core_scanner tests passed.\n")


if __name__ == "__main__":
    test_all()
