"""
article5_iterative_resistivity.py
==================================
Implements the iterative resistivity-modelling workflow described in:

    Merletti, G., Rabinovich, M., Al Hajri, S., Dawson, W., Farmer, R.,
    Ambia, J., Torres-Verdín, C. (2023).  "New Iterative Resistivity Modeling
    Workflow Reduces Uncertainty in the Assessment of Water Saturation in
    Deeply Invaded Reservoirs", Petrophysics, Vol. 64, No. 4, pp. 555-567.
    DOI: 10.30632/PJV64N4-2023a5

Workflow components
-------------------
1. Bed-boundary detection from a high-resolution resistivity curve using a
   sliding-window first-derivative + variance test (González et al., 2019).
2. Construction of an OBM-equivalent Sw=f(phi) regression with the P5/P50/P95
   prediction interval (Eq. 1 in the paper):

        Sw_P5  = 0.0136 * phi^(-1.313)
        Sw_P50 = 0.0123 * phi^(-1.210)
        Sw_P95 = 0.0114 * phi^(-1.007)

   These yield an Rt(P5/P50/P95) envelope through Archie's law that is then
   used to constrain a Bayesian / Markov-Chain Monte Carlo (MCMC) inversion
   of (Rt, Rxo) at every layer.
3. A simplified analytic forward model that maps an axisymmetric
   {Rxo, Rt, Lxo} earth model to four laterolog-array apparent resistivities
   with depth-of-investigation weights.
4. An MCMC sampler that explores Rt and Rxo in log-space with a Gaussian
   proposal, accepts a log-likelihood derived from the data misfit, and
   penalises Rt values that fall outside the OBM-equivalent envelope.

Run as a script for the synthetic test suite:

    python article5_iterative_resistivity.py
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Bed-boundary detection (variance + first-derivative sliding window)
# ---------------------------------------------------------------------------
def detect_bed_boundaries(log_curve, depth, win=11, threshold=None):
    """Return depth indices of detected bed boundaries.

    Implements the technique referenced as "González et al., 2019": a
    sliding window computes (a) the first derivative magnitude and
    (b) the local variance of the log; their product is thresholded.
    """
    log_curve = np.asarray(log_curve, dtype=float)
    n = len(log_curve)
    half = win // 2
    deriv = np.gradient(log_curve)
    variance = np.zeros(n)
    for i in range(half, n - half):
        variance[i] = np.var(log_curve[i - half:i + half + 1])
    metric = np.abs(deriv) * np.sqrt(variance)
    if threshold is None:
        threshold = np.mean(metric) + 2.0 * np.std(metric)
    # Local maxima of metric above threshold
    boundaries = []
    for i in range(1, n - 1):
        if metric[i] > threshold and metric[i] >= metric[i - 1] \
                and metric[i] >= metric[i + 1]:
            boundaries.append(i)
    return np.array(boundaries, dtype=int)


# ---------------------------------------------------------------------------
# 2. OBM-equivalent Sw envelope and Archie-derived Rt envelope (Eq. 1)
# ---------------------------------------------------------------------------
def sw_envelope(phi):
    """Return P5, P50, P95 Sw given porosity phi."""
    phi = np.asarray(phi, dtype=float)
    sw_p5  = np.clip(0.0136 * np.power(phi, -1.313), 0.0, 1.0)
    sw_p50 = np.clip(0.0123 * np.power(phi, -1.210), 0.0, 1.0)
    sw_p95 = np.clip(0.0114 * np.power(phi, -1.007), 0.0, 1.0)
    return sw_p5, sw_p50, sw_p95


def rt_envelope(phi, rw, a=1.0, m=2.0, n=2.0):
    """Convert the Sw envelope to an Rt envelope through Archie.

    Names are kept consistent with monotonic ordering:
    rt_low = lowest Rt (corresponds to high-Sw P5 case),
    rt_high = highest Rt (corresponds to low-Sw P95 case).
    """
    sw_p5, sw_p50, sw_p95 = sw_envelope(phi)
    # Higher Sw -> lower Rt
    rt_low  = a * rw / (np.power(phi, m) * np.power(sw_p5,  n))   # high Sw
    rt_med  = a * rw / (np.power(phi, m) * np.power(sw_p50, n))
    rt_high = a * rw / (np.power(phi, m) * np.power(sw_p95, n))   # low Sw
    return rt_low, rt_med, rt_high


# ---------------------------------------------------------------------------
# 3. Simplified array-laterolog forward model
# ---------------------------------------------------------------------------
def laterolog_apparent(rxo, rt, lxo,
                       doi=(0.15, 0.40, 0.80, 1.50)):
    """Simplified array-laterolog forward model.

    Each array at depth-of-investigation `doi_i` (m) returns

            Ra_i = ( w_i / Rxo + (1 - w_i) / Rt )^(-1)

    where the weight w_i is a smooth function of how much of the DOI
    falls inside the invaded shell of length lxo:

            w_i = 0.5*(1 - tanh( (doi_i - lxo) / 0.10 ))

    (purely conductive series-mixing approximation that captures the
    same trends as the vendor inversion shown in Figs. 4-5 of the paper).
    """
    rxo = np.atleast_1d(np.asarray(rxo, dtype=float))
    rt  = np.atleast_1d(np.asarray(rt,  dtype=float))
    lxo = np.atleast_1d(np.asarray(lxo, dtype=float))
    out = np.zeros((len(rxo), len(doi)))
    for i, d in enumerate(doi):
        w = 0.5 * (1.0 - np.tanh((d - lxo) / 0.10))
        out[:, i] = 1.0 / (w / rxo + (1.0 - w) / rt)
    return out


# ---------------------------------------------------------------------------
# 4. Constrained Bayesian / MCMC inversion for (Rt, Rxo) per layer
# ---------------------------------------------------------------------------
def mcmc_invert_layer(measured, lxo, rt_env, rxo_init=2.0,
                      n_iter=4000, burn_in=1000,
                      sigma_log=0.10, sigma_data=0.05,
                      doi=(0.15, 0.40, 0.80, 1.50),
                      array_weights=(0.5, 0.7, 1.0, 1.5),
                      seed=None):
    """Markov-Chain Monte Carlo inversion of Rt and Rxo for a single layer.

    Parameters
    ----------
    measured     : (n_array,) measured apparent resistivities
    lxo          : invasion radius for this layer (m, fixed in this routine)
    rt_env       : (rt_low, rt_med, rt_high) prior envelope on Rt
    rxo_init     : initial guess for Rxo
    sigma_log    : Gaussian step in log-space
    sigma_data   : multiplicative data-noise sigma
    doi          : depths of investigation of the four arrays
    array_weights: per-array log-likelihood weights (deeper = larger)

    Returns dict with keys 'rt', 'rxo' (medians), and the chains.
    """
    rng = np.random.default_rng(seed)
    rt_low, rt_med, rt_high = rt_env
    log_rxo = np.log(rxo_init)
    log_rt  = np.log(rt_med)
    aw = np.asarray(array_weights, dtype=float)

    def log_post(lrxo, lrt):
        rxo, rt = np.exp(lrxo), np.exp(lrt)
        if rxo <= 0 or rt <= 0:
            return -np.inf
        ra = laterolog_apparent(rxo, rt, lxo, doi=doi)[0]
        ll = -0.5 * np.sum(aw * ((np.log(ra) - np.log(measured)) / sigma_data) ** 2)
        # Soft prior: penalise log Rt outside [log rt_low, log rt_high]
        if rt < rt_low or rt > rt_high:
            mu = 0.5 * (np.log(rt_low) + np.log(rt_high))
            sd = 0.5 * (np.log(rt_high) - np.log(rt_low))
            ll += -0.5 * ((lrt - mu) / max(sd, 1e-3)) ** 2
        return ll

    cur = log_post(log_rxo, log_rt)
    rxo_chain, rt_chain = np.empty(n_iter), np.empty(n_iter)
    accepted = 0
    for it in range(n_iter):
        log_rxo_p = log_rxo + sigma_log * rng.standard_normal()
        log_rt_p  = log_rt  + sigma_log * rng.standard_normal()
        new = log_post(log_rxo_p, log_rt_p)
        if np.log(rng.uniform()) < new - cur:
            log_rxo, log_rt, cur = log_rxo_p, log_rt_p, new
            accepted += 1
        rxo_chain[it] = np.exp(log_rxo)
        rt_chain[it]  = np.exp(log_rt)

    return dict(
        rxo=float(np.median(rxo_chain[burn_in:])),
        rt=float(np.median(rt_chain[burn_in:])),
        rxo_chain=rxo_chain,
        rt_chain=rt_chain,
        acceptance=accepted / n_iter,
    )


def iterative_workflow(measured_arrays, phi, rw,
                       lxo_init=0.4, n_outer=3, **mcmc_kw):
    """Outer iterative loop with Lxo refinement (Sec. "2D INVERSION WORKFLOW").

    For each layer we (a) compute the Rt envelope from porosity, (b) MCMC for
    Rt/Rxo with Lxo fixed, then (c) refine Lxo by a coarse grid search to
    minimise data misfit.  A few outer iterations are run.
    """
    n_layers, n_arrays = measured_arrays.shape
    lxo = np.full(n_layers, lxo_init)
    rxo = np.full(n_layers, np.median(measured_arrays[:, 0]))
    rt  = np.full(n_layers, np.median(measured_arrays[:, -1]))
    for outer in range(n_outer):
        for j in range(n_layers):
            env = rt_envelope(phi[j], rw)
            res = mcmc_invert_layer(measured_arrays[j], lxo[j], env,
                                    rxo_init=rxo[j], **mcmc_kw)
            rxo[j], rt[j] = res["rxo"], res["rt"]
        # Lxo refinement -- coarse grid search per layer
        for j in range(n_layers):
            best_lxo, best_err = lxo[j], np.inf
            for cand in np.linspace(0.1, 2.5, 25):
                ra = laterolog_apparent(rxo[j], rt[j], cand)[0]
                err = np.sum((np.log(ra) - np.log(measured_arrays[j])) ** 2)
                if err < best_err:
                    best_err, best_lxo = err, cand
            lxo[j] = best_lxo
    return dict(rxo=rxo, rt=rt, lxo=lxo)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
def test_all(verbose=True):
    rng = np.random.default_rng(0)

    # --- 1. Bed-boundary detection on a synthetic blocky log ---------------
    depth = np.linspace(0, 30, 600)
    log = np.where(depth < 10, 5.0,
          np.where(depth < 20, 50.0, 5.0))
    log_noisy = log + rng.normal(scale=0.5, size=log.shape)
    bnds = detect_bed_boundaries(np.log10(log_noisy), depth)
    # Two true boundaries near indices 200 and 400 (depths 10 and 20)
    assert any(abs(b - 200) < 10 for b in bnds), "Top boundary not found"
    assert any(abs(b - 400) < 10 for b in bnds), "Bottom boundary not found"
    if verbose:
        print(f"[1] Bed-boundary detection OK    ({len(bnds)} boundaries)")

    # --- 2. OBM-equivalent envelope ----------------------------------------
    phi_grid = np.linspace(0.06, 0.18, 13)
    p5, p50, p95 = sw_envelope(phi_grid)
    assert np.all(p5 >= p50 - 1e-9) and np.all(p50 >= p95 - 1e-9)
    rt_lo, rt_md, rt_hi = rt_envelope(phi_grid, rw=0.05)
    assert np.all(rt_lo <= rt_md) and np.all(rt_md <= rt_hi)
    if verbose:
        print(f"[2] Sw envelope OK              "
              f"(@phi=0.10: Sw P5/P50/P95 = "
              f"{sw_envelope(0.10)[0]:.2f}/{sw_envelope(0.10)[1]:.2f}/"
              f"{sw_envelope(0.10)[2]:.2f})")

    # --- 3. Forward model trends ------------------------------------------
    ra = laterolog_apparent(rxo=1.0, rt=50.0, lxo=0.3)[0]
    assert ra[0] < ra[-1], "Shallowest array should read lower than deep"
    if verbose:
        print(f"[3] Forward model OK             "
              f"(arrays = {np.round(ra, 2)})")

    # --- 4. Single-layer MCMC recovery ------------------------------------
    rt_true, rxo_true, lxo_true, phi_true = 50.0, 1.5, 0.7, 0.12
    meas = laterolog_apparent(rxo_true, rt_true, lxo_true)[0]
    meas *= 1.0 + 0.02 * rng.standard_normal(meas.shape)
    env = rt_envelope(phi_true, rw=0.05)
    res = mcmc_invert_layer(meas, lxo_true, env,
                            rxo_init=1.0, n_iter=3000, burn_in=800,
                            seed=11)
    rel_err = abs(res["rt"] - rt_true) / rt_true
    assert rel_err < 0.30, f"MCMC Rt error {rel_err:.2%} too high"
    if verbose:
        print(f"[4] MCMC inversion OK            "
              f"(Rt = {res['rt']:.1f} vs {rt_true} ohm.m, "
              f"acc = {res['acceptance']:.2%})")

    # --- 5. Multi-layer iterative workflow --------------------------------
    n_layers = 4
    rt_truth  = np.array([20.0, 80.0, 60.0, 30.0])
    rxo_truth = np.array([1.0, 1.5, 2.0, 1.2])
    lxo_truth = np.array([0.5, 1.5, 1.0, 0.4])
    phi       = np.array([0.08, 0.13, 0.10, 0.09])
    measured = np.stack([laterolog_apparent(rxo_truth[k], rt_truth[k], lxo_truth[k])[0]
                         for k in range(n_layers)])
    measured *= 1.0 + 0.02 * rng.standard_normal(measured.shape)
    out = iterative_workflow(measured, phi, rw=0.05,
                             lxo_init=0.5, n_outer=2,
                             n_iter=1500, burn_in=400, seed=2)
    rel = np.abs(out["rt"] - rt_truth) / rt_truth
    assert np.median(rel) < 0.40
    if verbose:
        print(f"[5] Iterative workflow OK        "
              f"(median |Rt rel err| = {np.median(rel):.2%})")

    if verbose:
        print("\nAll article-5 tests passed.")
    return True


if __name__ == "__main__":
    test_all()
