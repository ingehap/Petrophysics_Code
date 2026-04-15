"""
article2_wiri.py
=================
Implementation of ideas from:

    Danielczick, Q., Nepesov, A., Rochereau, L., Lescoulie, S.,
    De Oliveira Fernandes, V., Nicot, B.
    "Wireless Acquisition for Resistivity Index in Centrifuge - WiRI:
    A Comparative Study of Three Pc-RI Methods"
    Petrophysics, Vol. 64, No. 3 (June 2023), pp. 340-352
    DOI: 10.30632/PJV64N3-2023a2

Three methods to determine Archie's saturation exponent ``n`` are
compared:

    PP        - Porous-Plate (reference, slow, log-log linear fit)
    UFPCRI    - Ultra-Fast Pc + RI from centrifuge with NMR profiling
                (set of independent RI/Sw points)
    WiRI      - Wireless Resistivity Index inside centrifuge
                (global least-squares inversion on all RI data at once)

The module reproduces the Monte-Carlo style sensitivity study of
Figs. 3 and 5 of the paper: it propagates random errors on production
volumes (saturation) and on resistivity, then estimates the resulting
distribution of n.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Forward Archie model
# ---------------------------------------------------------------------------
def archie_ri(sw: np.ndarray, n: float) -> np.ndarray:
    """Resistivity Index   RI = Sw**(-n)."""
    return np.power(sw, -n)


# ---------------------------------------------------------------------------
# PP  - Linear regression of log(RI) vs log(Sw)
# ---------------------------------------------------------------------------
def estimate_n_pp(sw: np.ndarray, ri: np.ndarray) -> float:
    """Estimate n by least-squares fit of log(RI) = -n * log(Sw)."""
    log_sw = np.log(sw)
    log_ri = np.log(ri)
    # n is -slope of log(RI) vs log(Sw)
    slope, _ = np.polyfit(log_sw, log_ri, 1)
    return float(-slope)


# ---------------------------------------------------------------------------
# UFPCRI - same fit but on independent (Sw, RI) points obtained at
# multiple centrifuge speeds.  Mathematically identical to PP for our
# Monte Carlo, the difference comes from the noise model applied to Sw.
# ---------------------------------------------------------------------------
def estimate_n_ufpcri(sw: np.ndarray, ri: np.ndarray) -> float:
    return estimate_n_pp(sw, ri)


# ---------------------------------------------------------------------------
# WiRI - global least-squares inversion of all RI data simultaneously
# Minimises sum_i ( log(RI_i) + n * log(Sw_i) )^2 with a single n
# ---------------------------------------------------------------------------
def estimate_n_wiri(sw: np.ndarray, ri: np.ndarray) -> float:
    log_sw = np.log(sw)
    log_ri = np.log(ri)
    # closed-form least squares without intercept (forced through origin)
    num = np.sum(log_sw * log_ri)
    den = np.sum(log_sw * log_sw)
    return float(-num / den)


# ---------------------------------------------------------------------------
# Monte-Carlo sensitivity study
# ---------------------------------------------------------------------------
def monte_carlo_n(method: str,
                  sw_true: np.ndarray,
                  n_true: float,
                  vol_err_su: float = 0.0,
                  rho_rel_err: float = 0.0,
                  n_runs: int = 2000,
                  seed: int | None = 0) -> np.ndarray:
    """
    Run a Monte-Carlo simulation for one method.

    Parameters
    ----------
    method      : 'PP' | 'UFPCRI' | 'WiRI'
    sw_true     : array of true Sw points used for the experiment
    n_true      : true Archie n exponent
    vol_err_su  : standard deviation of the absolute random error on
                  saturation [saturation units, e.g. 0.05 = 5 s.u.]
    rho_rel_err : standard deviation of the relative random error on
                  resistivity (fraction, e.g. 0.05 = 5 %)
    n_runs      : number of Monte-Carlo realisations
    """
    rng = np.random.default_rng(seed)
    estimator = {
        "PP": estimate_n_pp,
        "UFPCRI": estimate_n_ufpcri,
        "WiRI": estimate_n_wiri,
    }[method]

    ri_true = archie_ri(sw_true, n_true)
    out = np.empty(n_runs)
    for k in range(n_runs):
        sw = sw_true + vol_err_su * rng.standard_normal(sw_true.size)
        sw = np.clip(sw, 1e-3, 1.0)
        ri = ri_true * (1.0 + rho_rel_err * rng.standard_normal(ri_true.size))
        ri = np.clip(ri, 1e-3, None)
        out[k] = estimator(sw, ri)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    """Synthetic-data test for module 2 (WiRI vs UFPCRI vs PP)."""
    print("[article2] testing forward Archie law ...")
    sw = np.array([0.95, 0.7, 0.5, 0.35, 0.25, 0.18])
    n_true = 2.0
    ri = archie_ri(sw, n_true)

    # Without noise all three methods must recover n exactly
    for m in ("PP", "UFPCRI", "WiRI"):
        n_hat = {"PP": estimate_n_pp, "UFPCRI": estimate_n_ufpcri,
                 "WiRI": estimate_n_wiri}[m](sw, ri)
        assert abs(n_hat - n_true) < 1e-9, f"{m} failed: {n_hat}"
    print("           noise-free recovery of n=2  OK")

    print("[article2] running Monte-Carlo with 5 s.u. saturation error ...")
    results = {m: monte_carlo_n(m, sw, n_true, vol_err_su=0.05,
                                n_runs=2000) for m in ("PP", "UFPCRI", "WiRI")}
    for m, r in results.items():
        print(f"           {m:7s}  mean(n) = {r.mean():.3f}  "
              f"std = {r.std():.3f}  "
              f"5-95%% = [{np.percentile(r, 5):.2f}, "
              f"{np.percentile(r, 95):.2f}]")

    # Cf. paper: PP exhibits a downward bias when production-volume error
    # increases, while WiRI keeps a near-unbiased mean.
    assert results["PP"].mean() < n_true + 0.05, "PP mean should not over-shoot"
    assert abs(results["WiRI"].mean() - n_true) < 0.5, "WiRI mean off"

    print("[article2] running Monte-Carlo with 5%% resistivity error ...")
    res_r = monte_carlo_n("WiRI", sw, n_true, rho_rel_err=0.05, n_runs=2000)
    assert abs(res_r.mean() - n_true) < 0.3
    print(f"           WiRI under resistivity noise -> mean(n)={res_r.mean():.3f}")
    print("[article2] OK")


if __name__ == "__main__":
    test_all()
