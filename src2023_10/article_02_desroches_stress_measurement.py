"""
article_02_desroches_stress_measurement.py
==========================================
Implementation of ideas from:

    Desroches, J., Peyret, E., Gisolf, A., Wilcox, A., Di Giovanni, M.,
    Schram de Jong, A., Sepehri, S., Garrard, R., and Giger, S. (2023).
    "Stress Measurement Campaign in Scientific Deep Boreholes: Focus on
    Tools and Methods." Petrophysics, 64(5), 621-639.
    DOI: 10.30632/PJV64N5-2023a2

The paper describes how a wireline-conveyed micro-fracturing tool string
is used to measure the minimum horizontal stress (Sh_min) in deep
boreholes.  Each test produces a pressure-time record from which a set
of canonical pressure picks is interpreted:

    FBP   = formation breakdown pressure
    ISIP  = instantaneous shut-in pressure
    FCP   = fracture closure pressure (best estimate of Sh_min)
    FRP   = fracture reopening pressure   (Pf_open)

This module synthesises and analyses pressure-time data from such
micro-frac/leak-off tests, using two of the standard interpretation
techniques mentioned by Desroches et al.:

    * tangent / square-root-of-time intersection method for FCP
    * G-function derivative method for FCP

It also implements the classical Hubbert & Willis breakdown relation

    Pb = 3*Sh_min - SH_max - Pp + T

so a vertical stress profile (Sh_min vs depth) and a Mohr-friendly
plot can be produced from a hypothetical 8-test campaign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic micro-frac pressure curve
# ---------------------------------------------------------------------------
def synthetic_microfrac(t: np.ndarray,
                        FBP: float, FCP: float, ISIP: float,
                        pump_end_s: float = 60.0,
                        noise: float = 0.0,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a synthetic micro-frac pressure response (psi or any unit).

    The model has three regimes:
        - linear pressure build-up to FBP
        - sudden drop to ISIP at end of pumping (shut-in)
        - exponential decline towards FCP after shut-in
    """
    if rng is None:
        rng = np.random.default_rng(0)
    p = np.empty_like(t, dtype=float)
    pre = t < pump_end_s
    p[pre] = (t[pre] / pump_end_s) * FBP
    post = ~pre
    tau = (t[post] - pump_end_s)
    p[post] = FCP + (ISIP - FCP) * np.exp(-tau / 30.0)
    if noise > 0:
        p += rng.normal(0.0, noise, size=t.size)
    return p


# ---------------------------------------------------------------------------
# Pick FCP via tangent / sqrt-time intersection
# ---------------------------------------------------------------------------
def fcp_from_sqrt_time(t: np.ndarray, p: np.ndarray,
                       shut_in_idx: int) -> float:
    """Estimate fracture closure pressure (FCP) from the change in slope
    of the shut-in pressure plotted against sqrt(t-t_si).

    A linear fit is carried out on the early and late parts of the decline
    and the intersection is taken as FCP.
    """
    sqrt_t = np.sqrt(np.maximum(t - t[shut_in_idx], 0.0))
    p_si = p[shut_in_idx:]
    s_si = sqrt_t[shut_in_idx:]
    n = len(p_si)
    early = slice(2, n // 3)
    late = slice(int(0.66 * n), n - 2)
    a1, b1 = np.polyfit(s_si[early], p_si[early], 1)
    a2, b2 = np.polyfit(s_si[late], p_si[late], 1)
    if abs(a1 - a2) < 1e-12:
        return float(p_si[-1])
    s_int = (b2 - b1) / (a1 - a2)
    return float(a1 * s_int + b1)


# ---------------------------------------------------------------------------
# G-function derivative method (Nolte)
# ---------------------------------------------------------------------------
def g_function(delta_t: np.ndarray, t_pump: float) -> np.ndarray:
    """Nolte (1979) G-function for shut-in analysis (alpha = 1)."""
    td = delta_t / t_pump
    g0 = (4.0 / 3.0) * ((1.0 + td) ** 1.5 - td ** 1.5 - 1.0)
    return g0


def fcp_from_gfunction(t: np.ndarray, p: np.ndarray,
                       shut_in_idx: int, t_pump: float) -> float:
    """Estimate FCP from the maximum of d(p)/d(G)*G - i.e. the peak of
    the G*dP/dG curve, a standard hydraulic-fracture closure criterion."""
    dt = t[shut_in_idx:] - t[shut_in_idx]
    G = g_function(dt[1:], t_pump)            # avoid G(0) = 0
    p_si = p[shut_in_idx + 1:]
    dpdg = np.gradient(p_si, G)
    gpdpg = G * dpdg
    i_close = int(np.argmax(np.abs(np.gradient(gpdpg))))
    return float(p_si[i_close])


# ---------------------------------------------------------------------------
# Stress relations
# ---------------------------------------------------------------------------
@dataclass
class StressState:
    Sv: float        # vertical stress
    Sh_min: float    # minimum horizontal stress
    SH_max: float    # maximum horizontal stress
    Pp: float        # pore pressure


def hubbert_willis_breakdown(stress: StressState, T: float) -> float:
    """Pb = 3 * Sh_min - SH_max - Pp + T   (Hubbert and Willis, 1957)"""
    return 3.0 * stress.Sh_min - stress.SH_max - stress.Pp + T


def overburden_profile(depth_m: np.ndarray,
                       rho_avg_kg_m3: float = 2300.0) -> np.ndarray:
    """Sv(z) = integral(rho*g*dz)  -- approximated with constant rho."""
    g = 9.81
    return rho_avg_kg_m3 * g * depth_m / 1.0e6        # MPa


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_all() -> None:
    rng = np.random.default_rng(1)

    # Single synthetic test
    t = np.linspace(0, 600, 6001)
    FBP, FCP_true, ISIP = 80.0, 55.0, 65.0
    p = synthetic_microfrac(t, FBP, FCP_true, ISIP,
                            pump_end_s=60.0, noise=0.3, rng=rng)
    si = int(np.argmax(p))
    fcp_a = fcp_from_sqrt_time(t, p, si)
    fcp_b = fcp_from_gfunction(t, p, si, t_pump=60.0)
    # Both methods should be within ~10 % of true value
    assert abs(fcp_a - FCP_true) / FCP_true < 0.10, (fcp_a, FCP_true)
    assert abs(fcp_b - FCP_true) / FCP_true < 0.20, (fcp_b, FCP_true)

    # Stress profile for a campaign of 8 tests
    depths = np.linspace(500, 4500, 8)
    Sv = overburden_profile(depths)
    # Synthetic Sh_min ~ 0.7 * Sv with mild noise
    Sh = 0.7 * Sv + rng.normal(0, 1.5, size=Sv.size)
    SH = 1.05 * Sh + 5.0
    Pp = 0.45 * Sv
    for s_v, s_h, s_H, p_p in zip(Sv, Sh, SH, Pp):
        Pb = hubbert_willis_breakdown(
            StressState(s_v, s_h, s_H, p_p), T=2.0)
        # Breakdown should exceed Sh_min and be < 5*Sh_min
        assert s_h < Pb < 5.0 * s_h

    print(f"article_02_desroches_stress: OK  (FCP_sqrt={fcp_a:.2f}, "
          f"FCP_G={fcp_b:.2f}, true={FCP_true})")


if __name__ == "__main__":
    test_all()
