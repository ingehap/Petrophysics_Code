"""
Digital Rock Physics Wettability and Pore-Scale Flow Simulation
================================================================

Implements the ideas of:

    Faisal, T.F., Nono, F., Regaieg, M., Brugidou, R., and Caubit, C., 2025,
    "Variation of Wettability Across Different Lithotypes in a Reservoir and
    Its Impact on Digital Rock Physics Pore-Scale Simulations",
    Petrophysics, 66(6), 996–1012.
    DOI: 10.30632/PJV66N6-2025a6

Key ideas
---------
* Mixed-wet contact-angle assignment on segmented pore networks based
  on pore-size-dependent wettability models.
* Water-wet vs. oil-wet fraction parameterisation.
* Computation of relative permeability from pore-network models under
  different wettability scenarios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. Contact-Angle Distribution Models
# ──────────────────────────────────────────────────────────────────────
def uniform_contact_angle(
    n_pores: int,
    theta_min: float = 0.0,
    theta_max: float = 180.0,
    seed: int | None = None,
) -> np.ndarray:
    """Assign contact angles from a uniform distribution.

    Parameters
    ----------
    n_pores : int
        Number of pores in the network.
    theta_min, theta_max : float
        Minimum and maximum contact angle (degrees).
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Contact angles (degrees) for each pore.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(theta_min, theta_max, n_pores)


def mixed_wet_model(
    pore_radii: np.ndarray,
    r_threshold: float,
    theta_ww: float = 30.0,
    theta_ow: float = 140.0,
) -> np.ndarray:
    """Size-dependent mixed-wet contact-angle assignment.

    Pores with radius > r_threshold are assigned oil-wet contact angles
    (larger pores invaded during primary drainage).
    Pores with radius ≤ r_threshold remain water-wet.

    Parameters
    ----------
    pore_radii : array_like
        Pore inscribed radii.
    r_threshold : float
        Critical radius separating water-wet from oil-wet pores.
    theta_ww : float
        Water-wet contact angle (degrees).
    theta_ow : float
        Oil-wet contact angle (degrees).

    Returns
    -------
    np.ndarray
        Contact angle for each pore.
    """
    radii = np.asarray(pore_radii, float)
    theta = np.where(radii > r_threshold, theta_ow, theta_ww)
    return theta


def oil_wet_fraction(
    pore_radii: np.ndarray,
    r_threshold: float,
) -> float:
    """Fraction of pore volume that is oil-wet.

    Assumes pore volume ∝ r³.
    """
    radii = np.asarray(pore_radii, float)
    vol = radii ** 3
    return float(np.sum(vol[radii > r_threshold]) / np.sum(vol))


# ──────────────────────────────────────────────────────────────────────
# 2. Simplified Pore-Network Relative Permeability
# ──────────────────────────────────────────────────────────────────────
def _young_laplace(r: float, theta_deg: float, sigma: float) -> float:
    """Young–Laplace capillary entry pressure for a cylindrical pore.

    Pc = 2 σ cos(θ) / r
    """
    theta_rad = math.radians(theta_deg)
    return 2.0 * sigma * math.cos(theta_rad) / r


@dataclass
class PoreNetworkResult:
    """Results of a pore-network drainage simulation."""
    Sw: np.ndarray
    kro: np.ndarray
    krw: np.ndarray
    Pc: np.ndarray


def simple_bundle_of_tubes_kr(
    pore_radii: np.ndarray,
    theta: np.ndarray,
    sigma: float = 0.03,
    n_steps: int = 50,
) -> PoreNetworkResult:
    """Compute kr from a bundle-of-tubes model with mixed wettability.

    Each tube has its own contact angle (from the wettability model).
    Drainage: oil invades tubes in order of decreasing Pc (largest
    water-wet tubes first, then oil-wet tubes by re-imbibition pressure).

    This is a *simplified* demonstration — real DRP workflows use
    full network extraction and multi-phase LBM or finite-volume solvers.

    Parameters
    ----------
    pore_radii : array_like
        Tube radii (m).
    theta : array_like
        Contact angle per tube (degrees).
    sigma : float
        Interfacial tension (N/m).
    n_steps : int
        Number of capillary-pressure steps.

    Returns
    -------
    PoreNetworkResult
    """
    radii = np.asarray(pore_radii, float)
    theta = np.asarray(theta, float)
    n = len(radii)

    # Entry pressures
    Pc_entry = np.array([_young_laplace(r, th, sigma)
                         for r, th in zip(radii, theta)])

    # Volume proportional to r^2 * L (L constant, omitted)
    vol = radii ** 2
    total_vol = vol.sum()

    # Conductance proportional to r^4 (Hagen-Poiseuille)
    cond = radii ** 4
    total_cond = cond.sum()

    # Sort by Pc_entry descending (drainage order)
    order = np.argsort(-Pc_entry)

    Pc_range = np.linspace(Pc_entry.max(), Pc_entry.min(), n_steps)

    Sw_list, kro_list, krw_list, Pc_list = [], [], [], []

    for Pc_val in Pc_range:
        invaded = Pc_entry[order] >= Pc_val
        cum_invaded = np.cumsum(invaded)

        # Indices of invaded tubes
        invaded_idx = order[invaded]
        water_idx = np.setdiff1d(np.arange(n), invaded_idx)

        Sw_val = vol[water_idx].sum() / total_vol if len(water_idx) > 0 else 0.0
        kro_val = cond[invaded_idx].sum() / total_cond if len(invaded_idx) > 0 else 0.0
        krw_val = cond[water_idx].sum() / total_cond if len(water_idx) > 0 else 0.0

        Sw_list.append(Sw_val)
        kro_list.append(kro_val)
        krw_list.append(krw_val)
        Pc_list.append(Pc_val)

    return PoreNetworkResult(
        Sw=np.array(Sw_list), kro=np.array(kro_list),
        krw=np.array(krw_list), Pc=np.array(Pc_list),
    )


# ──────────────────────────────────────────────────────────────────────
# 3. Wettability Sensitivity: Compare Two Lithotypes
# ──────────────────────────────────────────────────────────────────────
def compare_lithotypes(
    radii_A: np.ndarray, r_thresh_A: float,
    radii_B: np.ndarray, r_thresh_B: float,
    sigma: float = 0.03,
    theta_ww: float = 30.0,
    theta_ow: float = 140.0,
) -> Tuple[PoreNetworkResult, PoreNetworkResult]:
    """Run pore-network kr for two lithotypes with different wettability
    thresholds (as in Faisal et al., comparing Lithotype 1 and 2).

    Returns
    -------
    result_A, result_B : PoreNetworkResult
    """
    theta_A = mixed_wet_model(radii_A, r_thresh_A, theta_ww, theta_ow)
    theta_B = mixed_wet_model(radii_B, r_thresh_B, theta_ww, theta_ow)
    res_A = simple_bundle_of_tubes_kr(radii_A, theta_A, sigma)
    res_B = simple_bundle_of_tubes_kr(radii_B, theta_B, sigma)
    return res_A, res_B


# ──────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Lithotype A: coarser, more oil-wet large pores
    radii_A = rng.lognormal(mean=-11, sigma=0.8, size=500)
    r_thresh_A = np.median(radii_A)

    # Lithotype B: finer, fewer oil-wet pores
    radii_B = rng.lognormal(mean=-12, sigma=0.5, size=500)
    r_thresh_B = np.percentile(radii_B, 75)

    res_A, res_B = compare_lithotypes(radii_A, r_thresh_A,
                                      radii_B, r_thresh_B)

    print(f"Lithotype A  OW fraction: {oil_wet_fraction(radii_A, r_thresh_A):.2f}")
    print(f"Lithotype B  OW fraction: {oil_wet_fraction(radii_B, r_thresh_B):.2f}")
    print(f"Lithotype A  Sorw (approx): {res_A.Sw.min():.3f}")
    print(f"Lithotype B  Sorw (approx): {res_B.Sw.min():.3f}")
