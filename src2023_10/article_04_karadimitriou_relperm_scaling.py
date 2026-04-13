"""
article_04_karadimitriou_relperm_scaling.py
===========================================
Implementation of ideas from:

    Karadimitriou, N., Valavanides, M. S., Mouravas, K., and Steeb, H.
    (2023).  "Flow-Dependent Relative Permeability Scaling for
    Steady-State Two-Phase Flow in Porous Media: Laboratory Validation
    on a Microfluidic Network."  Petrophysics, 64(5), 656-679.
    DOI: 10.30632/PJV64N5-2023a4

The paper investigates whether a *universal* relative-permeability
scaling exists once flow rate is accounted for, by performing
steady-state co-injection experiments on a microfluidic network and
plotting krw, krn vs the *capillary number* Ca and the flow-rate ratio
r = qn / qw.

Key ideas implemented here:

    * Brooks-Corey relative permeabilities as the "no-rate" baseline:
            krw  = krw_max  * Sw_e ** nw
            krnw = krnw_max * (1 - Sw_e) ** nnw
      with Sw_e = (Sw - Swir) / (1 - Swir - Snwr)

    * Capillary number scaling:
            Ca   = mu_w * v_w / sigma
      and a Valavanides-type rate dependence
            krw  = krw_BC  * (1 + alpha * log10(Ca / Ca_ref))
            krnw = krnw_BC * (1 + beta  * log10(Ca / Ca_ref))

    * Steady-state two-phase flow simulator on a 2-D network of pores
      (a simple bond-percolation-like model) used to compute krw, krn
      and reproduce the rate dependence empirically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Brooks-Corey baseline
# ---------------------------------------------------------------------------
@dataclass
class CoreyParams:
    Swir: float = 0.15
    Snwr: float = 0.10
    krw_max: float = 0.4
    krnw_max: float = 0.7
    nw: float = 2.5
    nnw: float = 2.0


def brooks_corey_kr(Sw: np.ndarray, p: CoreyParams) -> tuple[np.ndarray,
                                                             np.ndarray]:
    Swe = np.clip((Sw - p.Swir) / (1.0 - p.Swir - p.Snwr), 0.0, 1.0)
    krw = p.krw_max * Swe ** p.nw
    krnw = p.krnw_max * (1.0 - Swe) ** p.nnw
    return krw, krnw


# ---------------------------------------------------------------------------
# Capillary number and rate-dependent scaling
# ---------------------------------------------------------------------------
def capillary_number(mu_w_Pas: float, v_w_m_s: float,
                     ift_Nm: float) -> float:
    return mu_w_Pas * v_w_m_s / ift_Nm


def rate_scaled_kr(Sw: np.ndarray, Ca: float, p: CoreyParams,
                   Ca_ref: float = 1e-6,
                   alpha: float = 0.10, beta: float = -0.05
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Apply Valavanides-style log-Ca scaling on top of Brooks-Corey."""
    krw_bc, krnw_bc = brooks_corey_kr(Sw, p)
    log_ratio = np.log10(Ca / Ca_ref)
    fw = max(0.05, 1.0 + alpha * log_ratio)
    fn = max(0.05, 1.0 + beta * log_ratio)
    return krw_bc * fw, krnw_bc * fn


# ---------------------------------------------------------------------------
# Steady-state simulator on a synthetic micro-fluidic network
# ---------------------------------------------------------------------------
@dataclass
class Network:
    n_throats: int
    radii_m: np.ndarray         # throat radii distribution
    length_m: float = 1e-2

    @classmethod
    def random(cls, n: int = 5000, mean_r_um: float = 30.0,
               sigma_r_um: float = 8.0,
               rng: np.random.Generator | None = None) -> "Network":
        if rng is None:
            rng = np.random.default_rng(0)
        r = np.maximum(rng.lognormal(np.log(mean_r_um), sigma_r_um / mean_r_um,
                                     n) * 1e-6, 1e-6)
        return cls(n, r)


def steady_state_kr(net: Network, Sw: float, Ca: float, p: CoreyParams,
                    rng: np.random.Generator | None = None
                    ) -> tuple[float, float]:
    """Very small Pore-Network-Model 'simulator'.  Throats are randomly
    occupied by water/oil with frequency Sw / (1-Sw).  At higher Ca the
    invading phase preferentially occupies smaller throats (capillary
    fingering), at lower Ca the larger throats (capillary equilibrium).
    Effective phase conductivity is the sum of Hagen-Poiseuille
    conductances (~ r^4) for that phase, scaled by the single-phase
    total."""
    if rng is None:
        rng = np.random.default_rng(0)
    # Probability a throat is wet = Sw modulated by capillary preference
    pref = 1.0 / (1.0 + (net.radii_m / np.median(net.radii_m)) ** (-0.5
                          * np.log10(max(Ca, 1e-12) / 1e-6)))
    pref = np.clip(pref, 0.05, 0.95)
    p_wet = Sw * pref / np.mean(pref)
    p_wet = np.clip(p_wet, 0.0, 1.0)
    is_wet = rng.uniform(size=net.n_throats) < p_wet
    g = net.radii_m ** 4
    g_total = g.sum()
    g_w = g[is_wet].sum()
    g_n = g[~is_wet].sum()
    krw = g_w / g_total
    krn = g_n / g_total
    # Scale to user-supplied Brooks-Corey end-points
    krw_max_pnm = max(g.sum(), 1e-30)
    return float(p.krw_max * krw), float(p.krnw_max * krn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    rng = np.random.default_rng(0)
    p = CoreyParams()
    Sw = np.linspace(0.16, 0.89, 25)
    krw, krn = brooks_corey_kr(Sw, p)
    # Monotonicity
    assert np.all(np.diff(krw) >= -1e-12)
    assert np.all(np.diff(krn) <= 1e-12)
    # Endpoints near residual saturations
    assert krw[0] < 1e-3
    assert krn[-1] < 1e-2

    # Capillary number scaling: increasing Ca should raise krw, lower krn
    krw_lo, krn_lo = rate_scaled_kr(Sw, Ca=1e-8, p=p)
    krw_hi, krn_hi = rate_scaled_kr(Sw, Ca=1e-4, p=p)
    mid = len(Sw) // 2
    assert krw_hi[mid] > krw_lo[mid]
    assert krn_hi[mid] < krn_lo[mid]

    # PNM steady state
    net = Network.random(n=4000, rng=rng)
    krw_pnm, krn_pnm = steady_state_kr(net, Sw=0.5, Ca=1e-6, p=p, rng=rng)
    assert 0.0 < krw_pnm < p.krw_max
    assert 0.0 < krn_pnm < p.krnw_max
    # increasing Sw -> higher krw
    krw_low, _ = steady_state_kr(net, 0.2, 1e-6, p, rng)
    krw_high, _ = steady_state_kr(net, 0.8, 1e-6, p, rng)
    assert krw_high > krw_low

    Ca = capillary_number(1e-3, 1e-4, 30e-3)
    assert 1e-9 < Ca < 1e-3

    print(f"article_04_karadimitriou_relperm_scaling: OK "
          f"(krw_pnm={krw_pnm:.3f}, krn_pnm={krn_pnm:.3f})")


if __name__ == "__main__":
    test_all()
