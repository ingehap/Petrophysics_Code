"""
article_08_zhao_sp_resistivity_inversion.py
===========================================
Implementation of ideas from:

    Zhao, P., Wang, Y., Li, G., Hu, C., Xie, J., Duan, W., and Mao, Z.
    (2023).  "Joint Inversion of Saturation and Qv in Low-Permeability
    Sandstones Using Spontaneous Potential and Resistivity Logs."
    Petrophysics, 64(5), 741-752.  DOI: 10.30632/PJV64N5-2023a8

The paper derives an analytical expression for the SP membrane-potential
anomaly DSP in oil-bearing shaly sandstones based on Smits (1968) and
Waxman & Smits (1968), then jointly inverts SP and resistivity logs for
oil saturation Sw and the cation-exchange capacity per unit pore volume
Qv using particle-swarm optimisation (PSO).

Implemented:

    * Waxman-Smits resistivity for shaly sand:
        1/Rt = phi^m* / (a * Rw) * Sw^n* * (1 + B*Qv*Rw/Sw)
    * Smits-style SP membrane potential -> DSP analytical model
      (a tractable simplification with the same monotonic dependence
      on Cw, Qv and Sw as Eqs. 1-2 and the empirical curves in
      Figs. 1-3)
    * PSO optimiser with no derivative information
    * Joint objective: minimise normalised squared residuals of DSP
      and Rt with a weighting coefficient `a`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Forward models
# ---------------------------------------------------------------------------
@dataclass
class WSParams:
    a: float = 1.0
    m_star: float = 2.226       # from Fig. 6 fit
    n_star: float = 1.95
    B_factor: float = 4.0       # Waxman-Smits B coefficient (S/m)/(meq/cm3)


def waxman_smits_resistivity(phi: float, Sw: float, Qv: float,
                             Rw: float, p: WSParams) -> float:
    """Waxman-Smits resistivity for shaly sand (oil-bearing)."""
    F_star = p.a * phi ** (-p.m_star)
    cond_w = 1.0 / Rw + p.B_factor * Qv / max(Sw, 1e-3)
    Ct = (Sw ** p.n_star) / F_star * cond_w
    return 1.0 / max(Ct, 1e-12)


def sp_anomaly(phi: float, Sw: float, Qv: float,
               Cw_S_m: float, Cmf_S_m: float,
               T_K: float = 350.0,
               K_sp: float = -70.2) -> float:
    """Simplified analytic SP anomaly (mV) following Smits (1968) and
    the form used in Zhao et al. (Eqs. 1-3).

    The static SP between formation water and mud filtrate is

        Esh = -K * log10(a_w / a_mf)

    with a clay-related reduction factor that depends on Qv and a
    saturation reduction factor that depends on Sw (Ortiz et al. 1973)::

        DSP = Esh * f_clay(Qv, Cw) * f_sat(Sw)

    f_clay and f_sat are bounded between 0 (full clay/oil shielding)
    and 1 (clean, water-saturated).
    """
    Esh = K_sp * np.log10(max(Cw_S_m / Cmf_S_m, 1e-6))
    f_clay = 1.0 / (1.0 + 0.4 * Qv / max(Cw_S_m, 1e-6))
    f_sat = Sw ** 1.5
    return float(Esh * f_clay * f_sat)


# ---------------------------------------------------------------------------
# PSO
# ---------------------------------------------------------------------------
def pso(objective, bounds: list[tuple[float, float]],
        n_particles: int = 30, n_iter: int = 200,
        omega: float = 0.7, c1: float = 1.5, c2: float = 1.5,
        rng: np.random.Generator | None = None
        ) -> tuple[np.ndarray, float, list[float]]:
    """Plain PSO; returns (best position, best fitness, history)."""
    if rng is None:
        rng = np.random.default_rng(0)
    d = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    X = rng.uniform(lb, ub, size=(n_particles, d))
    V = rng.uniform(-(ub - lb), ub - lb, size=(n_particles, d)) * 0.1
    fit = np.array([objective(x) for x in X])
    pbest = X.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(fit))
    gbest = X[g_idx].copy()
    gbest_fit = fit[g_idx]
    history = [gbest_fit]
    for _ in range(n_iter):
        r1 = rng.uniform(size=(n_particles, d))
        r2 = rng.uniform(size=(n_particles, d))
        V = omega * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, lb, ub)
        fit = np.array([objective(x) for x in X])
        better = fit < pbest_fit
        pbest[better] = X[better]
        pbest_fit[better] = fit[better]
        g_idx = int(np.argmin(pbest_fit))
        if pbest_fit[g_idx] < gbest_fit:
            gbest = pbest[g_idx].copy()
            gbest_fit = pbest_fit[g_idx]
        history.append(gbest_fit)
    return gbest, gbest_fit, history


# ---------------------------------------------------------------------------
# Joint inversion
# ---------------------------------------------------------------------------
def joint_invert(phi: float, Rt_obs: float, DSP_obs: float,
                 Rw: float, Cmf_S_m: float,
                 ws: WSParams = WSParams(),
                 weight_sp: float = 1.0,
                 rng: np.random.Generator | None = None
                 ) -> tuple[float, float, float]:
    """Recover (Sw, Qv) from a single (Rt, DSP) observation."""
    Cw = 1.0 / Rw

    def obj(x: np.ndarray) -> float:
        Sw, Qv = x
        Rt_pred = waxman_smits_resistivity(phi, Sw, Qv, Rw, ws)
        DSP_pred = sp_anomaly(phi, Sw, Qv, Cw, Cmf_S_m)
        e1 = (Rt_pred - Rt_obs) / max(abs(Rt_obs), 1e-3)
        e2 = (DSP_pred - DSP_obs) / max(abs(DSP_obs), 1e-3)
        return e1 ** 2 + (weight_sp * e2) ** 2

    bounds = [(0.05, 0.95), (0.01, 5.0)]
    x_best, f_best, _ = pso(obj, bounds, n_particles=40, n_iter=150, rng=rng)
    return float(x_best[0]), float(x_best[1]), float(f_best)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    rng = np.random.default_rng(8)
    ws = WSParams()

    # Forward model sanity
    phi = 0.12
    Rw = 0.13                 # 50 kppm
    Cmf = 1.0 / 0.9
    # Water-bearing
    Rt_w = waxman_smits_resistivity(phi, 1.0, 1.0, Rw, ws)
    Rt_o = waxman_smits_resistivity(phi, 0.4, 1.0, Rw, ws)
    assert Rt_o > Rt_w        # oil-bearing -> higher Rt

    DSP_clean = sp_anomaly(phi, 1.0, 0.0, 1 / Rw, Cmf)
    DSP_oil = sp_anomaly(phi, 0.4, 1.5, 1 / Rw, Cmf)
    assert abs(DSP_clean) > abs(DSP_oil)   # oil + clay reduce SP magnitude

    # Single-point inversion: synthetic Sw, Qv
    Sw_true, Qv_true = 0.55, 1.4
    Rt_obs = waxman_smits_resistivity(phi, Sw_true, Qv_true, Rw, ws) \
        * (1 + 0.02 * rng.standard_normal())
    DSP_obs = sp_anomaly(phi, Sw_true, Qv_true, 1 / Rw, Cmf) \
        + 0.5 * rng.standard_normal()

    Sw_inv, Qv_inv, fit = joint_invert(phi, Rt_obs, DSP_obs, Rw, Cmf,
                                       ws=ws, rng=rng)
    assert abs(Sw_inv - Sw_true) < 0.1, (Sw_inv, Sw_true)
    assert abs(Qv_inv - Qv_true) < 0.5, (Qv_inv, Qv_true)

    # Multi-depth test
    n = 12
    Sw_arr = rng.uniform(0.3, 0.85, n)
    Qv_arr = rng.uniform(0.5, 2.5, n)
    errs_sw, errs_qv = [], []
    for sw, qv in zip(Sw_arr, Qv_arr):
        rt = waxman_smits_resistivity(phi, sw, qv, Rw, ws)
        ds = sp_anomaly(phi, sw, qv, 1 / Rw, Cmf)
        s_e, q_e, _ = joint_invert(phi, rt, ds, Rw, Cmf, ws=ws, rng=rng)
        errs_sw.append(abs(s_e - sw))
        errs_qv.append(abs(q_e - qv))
    mae_sw = float(np.mean(errs_sw))
    mae_qv = float(np.mean(errs_qv))
    assert mae_sw < 0.15, mae_sw
    assert mae_qv < 0.6, mae_qv

    print(f"article_08_zhao_sp_resistivity_inversion: OK  "
          f"(MAE Sw={mae_sw:.3f}, MAE Qv={mae_qv:.3f})")


if __name__ == "__main__":
    test_all()
