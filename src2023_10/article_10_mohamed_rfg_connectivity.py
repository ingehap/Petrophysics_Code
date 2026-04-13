"""
article_10_mohamed_rfg_connectivity.py
======================================
Implementation of ideas from:

    Mohamed, T. S., Torres-Verdin, C., and Mullins, O. C. (2023).
    "Enhanced Reservoir Description via Areal Data Integration and
    Reservoir Fluid Geodynamics: A Case Study From Deepwater Gulf of
    Mexico."  Petrophysics, 64(5), 773-795.
    DOI: 10.30632/PJV64N5-2023a10

The interpretation workflow has three "pillars":

    1. Areal Downhole Fluid Analysis (DFA) - asphaltene gradient
       modelling (Flory-Huggins-Zuo Equation of State).
    2. Pressure-gradient analysis - identification of compartments
       and fault-block migration from formation-pressure surveys.
    3. Reservoir Fluid Geodynamics (RFG) - using GC fingerprints,
       biomarkers, and the asphaltene profile to deduce ongoing or
       past processes (gas charge, biodegradation, fault leakage).

This module implements:

    * The Flory-Huggins-Zuo (FHZ) compositional gradient equation for
      asphaltene mass fraction with depth (after Freed et al. 2010,
      Mullins 2019)
    * A piecewise pressure-gradient analysis that detects gradient
      changes (fluid contacts) and pressure offsets (compartments)
    * A simple oil-viscosity correlation from asphaltene fraction
      (Mullins et al. 2019)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Flory-Huggins-Zuo equation of state for asphaltene gradient
# ---------------------------------------------------------------------------
@dataclass
class FHZParams:
    M_a: float = 1700.0     # asphaltene molar mass (g/mol; nano-aggregate ~ 2000)
    rho_a: float = 1.20     # asphaltene density (g/cc)
    rho_o: float = 0.80     # oil density (g/cc)
    delta_a: float = 21.85  # asphaltene solubility parameter (MPa^0.5)
    delta_o: float = 18.50  # oil solubility parameter (MPa^0.5)
    T_K: float = 365.0      # temperature
    g: float = 9.81


def fhz_gradient(z_m: np.ndarray, phi_a_ref: float,
                 z_ref_m: float, p: FHZParams = FHZParams()) -> np.ndarray:
    """Compute the asphaltene volume fraction profile using the FHZ EoS:

        ln(phi_a(h2)/phi_a(h1)) = (V_a*g*(rho_o - rho_a) (h2-h1) / (R*T))
                                + (V_a/RT) * (delta_a - delta_o)^2 *
                                  ((1 - phi_a)^2 |_h2 - (1 - phi_a)^2 |_h1)

    For modest asphaltene fractions (<= 5 %) the second (solubility) term
    contributes weakly, but we keep it.  z_m is depth (m, positive
    downwards).
    """
    R = 8.314
    V_a = p.M_a / (p.rho_a * 1000.0)        # m^3/mol
    # Asphaltenes are denser than oil so (rho_a - rho_o) > 0; with z
    # positive downwards the gravitational term yields accumulation
    # at depth.
    grav_term = (V_a * (p.rho_a - p.rho_o) * 1000.0 * p.g
                 / (R * p.T_K))             # 1/m
    # Solve iteratively because solubility term depends on phi_a
    phi = np.full_like(z_m, phi_a_ref, dtype=float)
    for _ in range(20):
        sol_term = (V_a / (R * p.T_K)) * ((p.delta_a - p.delta_o) * 1e3) ** 2 \
            * ((1.0 - phi) ** 2 - (1.0 - phi_a_ref) ** 2)
        # Clip exponent argument to avoid overflow at large depth ranges
        arg = np.clip(grav_term * (z_m - z_ref_m) + sol_term, -10.0, 10.0)
        phi = phi_a_ref * np.exp(arg)
    return np.clip(phi, 1e-6, 0.5)


def viscosity_from_asphaltene(phi_a: np.ndarray,
                              mu0_cp: float = 1.0,
                              k: float = 25.0) -> np.ndarray:
    """Empirical exponential viscosity correlation
    mu = mu0 * exp(k * phi_a)   (Mullins et al. 2007/2019)."""
    return mu0_cp * np.exp(k * phi_a)


# ---------------------------------------------------------------------------
# Pressure gradient analysis
# ---------------------------------------------------------------------------
def fit_pressure_gradients(z_m: np.ndarray, p_psi: np.ndarray,
                           min_pts: int = 3,
                           min_seg_len_m: float = 5.0
                           ) -> list[tuple[float, float, float, float]]:
    """Greedy piecewise-linear segmentation: walk through the data and
    start a new gradient when the local slope deviates from the running
    fit by more than a few standard deviations.

    Returns a list of (z_top, z_bot, gradient_psi_per_m, intercept_psi).
    """
    segs: list[tuple[float, float, float, float]] = []
    i = 0
    n = z_m.size
    while i < n - min_pts:
        j = i + min_pts
        a, b = np.polyfit(z_m[i:j], p_psi[i:j], 1)
        while j < n:
            a2, b2 = np.polyfit(z_m[i:j + 1], p_psi[i:j + 1], 1)
            # Compute residual for the trial fit
            resid = np.std(p_psi[i:j + 1] - (a2 * z_m[i:j + 1] + b2))
            if resid > 2.0 and (z_m[j] - z_m[i]) > min_seg_len_m:
                break
            a, b = a2, b2
            j += 1
        segs.append((z_m[i], z_m[j - 1], float(a), float(b)))
        i = j
    return segs


def detect_compartments(segments: list[tuple[float, float, float, float]],
                        depth_test_m: float,
                        tol_psi: float = 5.0) -> list[int]:
    """Compare extrapolated pressures of adjacent segments at a common
    depth - a step > tol_psi indicates a compartment / fault seal."""
    flags = []
    for k in range(len(segments) - 1):
        a1, b1 = segments[k][2], segments[k][3]
        a2, b2 = segments[k + 1][2], segments[k + 1][3]
        p1 = a1 * depth_test_m + b1
        p2 = a2 * depth_test_m + b2
        if abs(p1 - p2) > tol_psi:
            flags.append(k)
    return flags


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    rng = np.random.default_rng(10)

    # FHZ asphaltene gradient
    z = np.linspace(2500.0, 2700.0, 21)
    phi_a = fhz_gradient(z, phi_a_ref=0.005, z_ref_m=2500.0)
    # Asphaltenes accumulate at depth (rho_a > rho_o)
    assert phi_a[-1] > phi_a[0], (phi_a[0], phi_a[-1])
    assert phi_a[0] == 0.005

    # Viscosity grows with depth
    mu = viscosity_from_asphaltene(phi_a, mu0_cp=2.0, k=30.0)
    assert mu[-1] > mu[0]

    # Two-compartment synthetic pressure survey
    z_top = np.linspace(2400, 2500, 6)
    p_top = 0.45 * z_top + 100.0 + 0.5 * rng.standard_normal(z_top.size)
    z_bot = np.linspace(2510, 2620, 8)
    p_bot = 0.34 * z_bot + 280.0 + 0.5 * rng.standard_normal(z_bot.size)
    z_all = np.concatenate([z_top, z_bot])
    p_all = np.concatenate([p_top, p_bot])
    segs = fit_pressure_gradients(z_all, p_all,
                                  min_pts=3, min_seg_len_m=8.0)
    assert len(segs) >= 2, segs
    flags = detect_compartments(segs, depth_test_m=2505.0, tol_psi=5.0)
    assert len(flags) >= 1, "Should detect at least one compartment break"

    # Single-compartment survey - should NOT flag
    z1 = np.linspace(2400, 2620, 16)
    p1 = 0.42 * z1 + 150.0 + 0.4 * rng.standard_normal(z1.size)
    segs1 = fit_pressure_gradients(z1, p1, min_pts=3, min_seg_len_m=8.0)
    flags1 = detect_compartments(segs1, depth_test_m=2510.0, tol_psi=5.0)
    assert len(flags1) == 0

    print(f"article_10_mohamed_rfg_connectivity: OK  "
          f"(asph {phi_a[0]*100:.2f}->{phi_a[-1]*100:.2f}%, "
          f"mu {mu[0]:.2f}->{mu[-1]:.2f} cp, segs={len(segs)})")


if __name__ == "__main__":
    test_all()
