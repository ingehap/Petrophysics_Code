#!/usr/bin/env python3
"""
Lateral Permeability Variations from LWD NMR & Micro-Resistivity Imaging
==========================================================================
Based on: Fouda et al. (2024), "Understanding Lateral Permeability
Variations in Heterogeneous Carbonate Reservoirs Using Logging-While-
Drilling NMR, Microresistivity Imaging, and Azimuthally Oriented
Formation Testing", Petrophysics, Vol. 65, No. 5, pp. 772-788.

Implements:
- Timur-Coates NMR permeability model
- SDR (Schlumberger-Doll Research) NMR permeability model
- Azimuthal permeability estimation from oriented formation tests
- Heterogeneity index from micro-resistivity imaging
- Lateral permeability profile construction

Reference: DOI:10.30632/PJV65N5-2024a7
"""

import numpy as np
from typing import Tuple, Dict, Optional


def timur_coates_perm(phi: np.ndarray, BVI: np.ndarray, FFI: np.ndarray,
                      C: float = 10.0, a: float = 4.0,
                      b: float = 2.0) -> np.ndarray:
    """
    Timur-Coates NMR permeability model:

        k = C * phi^a * (FFI / BVI)^b

    Parameters
    ----------
    phi : array-like
        NMR total porosity (fraction).
    BVI : array-like
        Bulk Volume Irreducible (fraction of bulk volume).
    FFI : array-like
        Free Fluid Index (fraction of bulk volume).
    C : float
        Empirical constant (default 10 for carbonates).
    a, b : float
        Exponents.

    Returns
    -------
    np.ndarray
        Permeability (md).
    """
    phi = np.asarray(phi, dtype=float)
    BVI = np.asarray(BVI, dtype=float)
    FFI = np.asarray(FFI, dtype=float)
    ratio = np.divide(FFI, BVI, out=np.zeros_like(FFI),
                      where=BVI > 1e-6)
    k = C * phi ** a * ratio ** b
    return np.maximum(k, 1e-4)


def sdr_perm(phi: np.ndarray, T2lm: np.ndarray,
             C: float = 4.0, a: float = 4.0, b: float = 2.0
             ) -> np.ndarray:
    """
    SDR (Schlumberger-Doll Research) NMR permeability model:

        k = C * phi^a * T2lm^b

    Parameters
    ----------
    phi : array-like
        NMR total porosity (fraction).
    T2lm : array-like
        Logarithmic mean T2 relaxation time (ms).
    C, a, b : float
        Empirical parameters.

    Returns
    -------
    np.ndarray
        Permeability (md).
    """
    phi = np.asarray(phi, dtype=float)
    T2lm = np.asarray(T2lm, dtype=float)
    k = C * phi ** a * T2lm ** b
    return np.maximum(k, 1e-4)


def azimuthal_perm_from_formation_test(
        pressure_data: Dict[str, float],
        mu: float = 1.0,
        rp: float = 0.005,
        C_probe: float = 5660.0
) -> Dict[str, float]:
    """
    Estimate directional permeability from oriented formation test data.

    k = (q * mu) / (C_probe * delta_P)

    Parameters
    ----------
    pressure_data : dict
        Keys are azimuth labels (e.g., 'top', 'bottom', 'left', 'right'),
        values are tuples (flow_rate_cc_s, pressure_drop_psi).
    mu : float
        Fluid viscosity (cp).
    rp : float
        Probe radius (m).
    C_probe : float
        Probe geometric factor.

    Returns
    -------
    dict
        {azimuth: permeability_md}
    """
    results = {}
    for azimuth, (q, dP) in pressure_data.items():
        if dP > 0:
            k = (q * mu) / (C_probe * dP) * 1e6  # convert to md
            results[azimuth] = max(k, 0.0)
        else:
            results[azimuth] = 0.0
    return results


def heterogeneity_index_from_imaging(resistivity_image: np.ndarray
                                     ) -> float:
    """
    Compute a heterogeneity index from micro-resistivity image data.

    HI = coefficient of variation of log(resistivity) within a window.

    Parameters
    ----------
    resistivity_image : np.ndarray
        2-D array of resistivity values (depth × azimuth).

    Returns
    -------
    float
        Heterogeneity index (dimensionless).
    """
    log_res = np.log10(np.maximum(resistivity_image, 1e-3))
    return float(np.std(log_res) / (np.abs(np.mean(log_res)) + 1e-6))


def lateral_perm_profile(phi: np.ndarray, T2lm: np.ndarray,
                         azimuths: np.ndarray,
                         method: str = "SDR"
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a lateral (azimuthal) permeability profile from NMR data.

    Parameters
    ----------
    phi : np.ndarray  shape (n_depths, n_azimuths)
        Azimuthal porosity.
    T2lm : np.ndarray  shape (n_depths, n_azimuths)
        Azimuthal T2 log-mean.
    azimuths : np.ndarray  shape (n_azimuths,)
        Azimuth angles (degrees).
    method : str
        'SDR' or 'Timur-Coates'.

    Returns
    -------
    azimuths : np.ndarray
    k_mean : np.ndarray   shape (n_azimuths,)
        Mean permeability at each azimuth.
    """
    if method == "SDR":
        k = sdr_perm(phi, T2lm)
    else:
        # Approximate BVI/FFI split: BVI ≈ phi * (1 - T2lm/500)
        BVI = phi * np.clip(1.0 - T2lm / 500.0, 0.1, 0.9)
        FFI = phi - BVI
        k = timur_coates_perm(phi, BVI, FFI)

    k_mean = np.mean(k, axis=0)
    return azimuths, k_mean


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Lateral Permeability Variations Module Demo ===\n")
    np.random.seed(42)

    # Synthetic NMR data
    n = 50
    phi = np.random.uniform(0.10, 0.30, n)
    T2lm = np.random.uniform(20, 300, n)
    BVI = phi * np.random.uniform(0.2, 0.5, n)
    FFI = phi - BVI

    k_sdr = sdr_perm(phi, T2lm)
    k_tc = timur_coates_perm(phi, BVI, FFI)
    print(f"SDR perm range: [{k_sdr.min():.2f}, {k_sdr.max():.2f}] md")
    print(f"Timur-Coates perm range: [{k_tc.min():.2f}, {k_tc.max():.2f}] md")

    # Azimuthal formation test
    pressure_data = {
        'top': (0.5, 50),
        'bottom': (0.5, 30),
        'left': (0.5, 80),
        'right': (0.5, 45),
    }
    az_perm = azimuthal_perm_from_formation_test(pressure_data)
    print("\nAzimuthal permeabilities from formation test:")
    for az, k in az_perm.items():
        print(f"  {az:>8s}: {k:.2f} md")

    # Heterogeneity index
    res_image = 10 ** np.random.normal(1.5, 0.5, (100, 36))
    hi = heterogeneity_index_from_imaging(res_image)
    print(f"\nHeterogeneity index: {hi:.4f}")

    # Lateral profile
    n_depths, n_az = 30, 8
    phi_2d = np.random.uniform(0.10, 0.25, (n_depths, n_az))
    T2_2d = np.random.uniform(50, 200, (n_depths, n_az))
    az_angles = np.linspace(0, 315, n_az)
    az, k_mean = lateral_perm_profile(phi_2d, T2_2d, az_angles)
    print("\nLateral permeability profile:")
    for a, k in zip(az, k_mean):
        print(f"  {a:6.1f}°: {k:.2f} md")
