"""
article1_hdt.py
================
Implementation of ideas from:

    Fernandes, V., Nicot, B., Pairoys, F., Bertin, H., Lachaud, J., Caubit, C.
    "Hybrid Technique for Setting Initial Water Saturation on Core Samples"
    Petrophysics, Vol. 64, No. 3 (June 2023), pp. 325-339
    DOI: 10.30632/PJV64N3-2023a1

The Hybrid Drainage Technique (HDT) couples a viscous-flooding step
(Phase 1) with a porous-plate step (Phase 2) inside the same overburden
cell.  Phase 1 quickly brings the average water saturation close to the
target Swi but leaves a capillary end effect (CEE).  Phase 2 then
homogenises the saturation profile by imposing a constant capillary
pressure through a semi-permeable plate at the outlet.

This module provides:
  * `centrifuge_pc(omega, R, r, delta_rho)`  - Hassler-Brunner Pc(r) eq.
  * `viscous_flood_profile(...)`             - Phase 1 saturation profile
  * `porous_plate_homogenize(...)`           - Phase 2 relaxation toward
                                                a uniform Sw
  * `simulate_hdt(...)`                      - couples both phases
  * `profile_stats(profile)`                 - std-dev / max-min metrics
  * `test_all()`                             - synthetic test
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Centrifuge capillary pressure (Eq. 1 of the paper, Hassler & Brunner 1945)
# ---------------------------------------------------------------------------
def centrifuge_pc(omega: float, R: float, r: np.ndarray,
                  delta_rho: float) -> np.ndarray:
    """
    Capillary pressure inside a centrifuged core sample.

        Pc(r) = 0.5 * delta_rho * omega**2 * (R**2 - r**2)

    Parameters
    ----------
    omega      : angular speed [rad/s]
    R          : distance from axis to outlet face [m]
    r          : distance from axis to point of interest [m]
    delta_rho  : density contrast (rho_water - rho_oil) [kg/m**3]

    Returns
    -------
    Pc [Pa]
    """
    return 0.5 * delta_rho * omega ** 2 * (R ** 2 - r ** 2)


# ---------------------------------------------------------------------------
# Phase 1 - viscous oil flood with capillary end effect
# ---------------------------------------------------------------------------
def viscous_flood_profile(L: float, n: int, swi_target: float,
                          cee_length: float = 0.01,
                          sw_outlet: float = 0.95,
                          noise: float = 0.0,
                          rng: np.random.Generator | None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic saturation profile after Phase 1 (viscous flooding).

    The bulk of the sample is at the target Swi while the outlet
    centimetre of the core retains a 'capillary foot' that rises
    smoothly (linear ramp) to ``sw_outlet``.

    Returns
    -------
    x   : positions along the core axis [m]
    sw  : water saturation profile (fractional)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    x = np.linspace(0.0, L, n)
    sw = np.full(n, swi_target, dtype=float)

    # Capillary foot near outlet
    foot = x >= (L - cee_length)
    if foot.any():
        ramp = (x[foot] - (L - cee_length)) / cee_length
        sw[foot] = swi_target + (sw_outlet - swi_target) * ramp

    sw += noise * rng.standard_normal(n)
    return x, np.clip(sw, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Phase 2 - porous plate equilibration
# ---------------------------------------------------------------------------
def porous_plate_homogenize(sw_profile: np.ndarray,
                            target_sw: float,
                            n_iter: int = 50,
                            relax: float = 0.15) -> np.ndarray:
    """
    Iterative relaxation of the saturation profile toward a uniform
    value imposed by the porous-plate (Phase 2 of HDT).

    A simple convex update is used (as if the porous plate locally
    equalises Pc and water is redistributed by Darcy flow).  This is
    not a full numerical simulator but reproduces the behaviour
    illustrated in Figs. 10, 12 and 16 of the paper.
    """
    sw = sw_profile.astype(float).copy()
    for _ in range(n_iter):
        # local diffusion (smoothing) toward neighbours
        smoothed = np.empty_like(sw)
        smoothed[1:-1] = 0.5 * (sw[:-2] + sw[2:])
        smoothed[0] = sw[0]
        smoothed[-1] = sw[-1]
        # pull globally toward the target imposed by the plate
        sw = (1 - relax) * sw + relax * (0.5 * smoothed + 0.5 * target_sw)
    return sw


# ---------------------------------------------------------------------------
# Profile statistics (cf. paper - 'Std-dev (Sw)' and 'Max-min (Sw)')
# ---------------------------------------------------------------------------
def profile_stats(sw_profile: np.ndarray) -> dict:
    """Return dictionary with std-dev and max-min for a Sw profile."""
    return {
        "mean": float(np.mean(sw_profile)),
        "std": float(np.std(sw_profile)),
        "max_min": float(np.max(sw_profile) - np.min(sw_profile)),
    }


# ---------------------------------------------------------------------------
# Full HDT simulation
# ---------------------------------------------------------------------------
def simulate_hdt(L: float = 0.05, n: int = 101,
                 swi_target: float = 0.22,
                 cee_length: float = 0.01,
                 sw_outlet_phase1: float = 0.9,
                 noise: float = 0.005,
                 phase2_iters: int = 80) -> dict:
    """
    Run a synthetic HDT experiment and return profiles before / after
    Phase 2 plus the corresponding statistics.
    """
    x, sw1 = viscous_flood_profile(L, n, swi_target, cee_length,
                                   sw_outlet_phase1, noise=noise)
    sw2 = porous_plate_homogenize(sw1, target_sw=swi_target,
                                  n_iter=phase2_iters)
    return {
        "x": x,
        "sw_phase1": sw1,
        "sw_phase2": sw2,
        "stats_phase1": profile_stats(sw1),
        "stats_phase2": profile_stats(sw2),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    """Synthetic-data test for module 1 (HDT)."""
    print("[article1] testing centrifuge_pc ...")
    R = 0.10
    r = np.linspace(0.05, R, 11)
    pc = centrifuge_pc(omega=2 * np.pi * 50, R=R, r=r, delta_rho=200.0)
    assert pc[-1] == 0.0, "Pc must vanish at r = R"
    assert np.all(np.diff(pc) <= 0), "Pc must decrease as r -> R"

    print("[article1] simulating Bentheimer HDT (Swi target 22%) ...")
    out = simulate_hdt(swi_target=0.22, sw_outlet_phase1=0.85, noise=0.005)
    s1, s2 = out["stats_phase1"], out["stats_phase2"]

    # Phase 2 should reduce both metrics significantly
    assert s2["std"] < s1["std"], (
        f"Phase 2 std ({s2['std']:.3f}) must be smaller than "
        f"Phase 1 std ({s1['std']:.3f})"
    )
    assert s2["max_min"] < s1["max_min"], "Phase 2 must shrink max-min"
    assert abs(s2["mean"] - 0.22) < 0.05, (
        f"Mean Sw after phase 2 should be near target, got {s2['mean']:.3f}"
    )

    print(f"           Phase 1: mean={s1['mean']:.3f}  std={s1['std']:.3f}  "
          f"max-min={s1['max_min']:.3f}")
    print(f"           Phase 2: mean={s2['mean']:.3f}  std={s2['std']:.3f}  "
          f"max-min={s2['max_min']:.3f}")
    print("[article1] OK")


if __name__ == "__main__":
    test_all()
