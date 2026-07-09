"""Stochastic inversion: likelihoods, priors, MCMC, MALA, ensemble methods.

Bayesian sampling and ensemble solvers: Gaussian log-likelihood and simple
priors, the Metropolis and MALA samplers, an ensemble-Kalman update, and the
Levenberg-Marquardt ensemble randomized maximum likelihood (LM-EnRML).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .nonlinear import fd_gradient

_Float = NDArray[np.float64]
_LogPost = Callable[[_Float], float]


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


class Chain(NamedTuple):
    """MCMC output: posterior ``samples`` and the ``acceptance`` fraction."""

    samples: _Float
    acceptance: float


def gaussian_loglik(
    obs: ArrayLike,
    pred: ArrayLike,
    sigma: ArrayLike,
    weights: ArrayLike | None = None,
    log_space: bool = False,
) -> float:
    """Gaussian log-likelihood ``-0.5*sum(w*((obs-pred)/sigma)^2)``."""
    o = _arr(obs)
    p = _arr(pred)
    if log_space:
        o = np.log(o)
        p = np.log(p)
    r = (o - p) / _arr(sigma)
    w = np.ones_like(r) if weights is None else _arr(weights)
    return float(-0.5 * np.sum(w * r**2))


def uniform_logprior(x: ArrayLike, bounds: list[tuple[float, float]]) -> float:
    """Uniform (box) log-prior: ``0`` inside ``bounds``, ``-inf`` outside."""
    x_arr = _arr(x)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    if np.all(x_arr >= lo) and np.all(x_arr <= hi):
        return 0.0
    return -np.inf


def soft_envelope_logprior(x: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> float:
    """Soft box log-prior: ``0`` inside ``[lo, hi]`` and a quadratic penalty outside."""
    x_arr = _arr(x)
    below = np.minimum(x_arr - _arr(lo), 0.0)
    above = np.maximum(x_arr - _arr(hi), 0.0)
    return float(-0.5 * np.sum(below**2 + above**2))


def metropolis(
    log_post: _LogPost,
    x0: ArrayLike,
    step: ArrayLike,
    n_samples: int = 4000,
    *,
    log_space: bool = False,
    burn_in: int = 0,
    seed: int = 0,
) -> Chain:
    """Random-walk Metropolis sampler of ``log_post``.

    ``log_space=True`` proposes multiplicative moves ``x*exp(step*N(0,1))``;
    otherwise additive ``x + step*N(0,1)``.  Returns a :class:`Chain`.
    """
    rng = np.random.default_rng(seed)
    x = _arr(x0).copy()
    s = _arr(step)
    lp = log_post(x)
    samples = []
    n_acc = 0
    for i in range(n_samples):
        if log_space:
            cand = x * np.exp(s * rng.standard_normal(x.size))
        else:
            cand = x + s * rng.standard_normal(x.size)
        lp_c = log_post(cand)
        if np.log(rng.uniform()) < lp_c - lp:
            x, lp = cand, lp_c
            n_acc += 1
        if i >= burn_in:
            samples.append(x.copy())
    return Chain(np.array(samples), n_acc / n_samples)


def mala(log_post: _LogPost, x0: ArrayLike, step: float, n_samples: int, seed: int = 0) -> Chain:
    """Metropolis-adjusted Langevin (MALA) sampler with a finite-difference drift."""
    rng = np.random.default_rng(seed)
    x = _arr(x0).copy()
    s2 = step**2

    def grad(z: _Float) -> _Float:
        return fd_gradient(log_post, z, eps=1e-5)

    lp = log_post(x)
    g = grad(x)
    samples = []
    n_acc = 0
    for _ in range(n_samples):
        cand = x + 0.5 * s2 * g + step * rng.standard_normal(x.size)
        lp_c = log_post(cand)
        g_c = grad(cand)
        q_fwd = -np.sum((cand - x - 0.5 * s2 * g) ** 2) / (2.0 * s2)
        q_bwd = -np.sum((x - cand - 0.5 * s2 * g_c) ** 2) / (2.0 * s2)
        if np.log(rng.uniform()) < lp_c - lp + q_bwd - q_fwd:
            x, lp, g = cand, lp_c, g_c
            n_acc += 1
        samples.append(x.copy())
    return Chain(np.array(samples), n_acc / n_samples)


def enkf_update(
    ens: ArrayLike,
    obs: ArrayLike,
    obs_cov: ArrayLike,
    obs_op: Callable[[_Float], Any] | ArrayLike,
    seed: int = 0,
) -> _Float:
    """Stochastic (perturbed-observation) ensemble-Kalman update of ``ens``.

    ``ens`` is ``(n_ens, n_state)``; ``obs_op`` is the observation operator (a
    callable state->obs or a matrix ``H``).  Returns the updated ensemble.
    """
    rng = np.random.default_rng(seed)
    x = _arr(ens)
    n_ens = x.shape[0]
    o = _arr(obs)
    r_cov = _arr(obs_cov)
    if callable(obs_op):
        hx = np.array([_arr(obs_op(row)) for row in x])
    else:
        hx = x @ _arr(obs_op).T
    xa = x - x.mean(0)
    hxa = hx - hx.mean(0)
    p_hh = hxa.T @ hxa / (n_ens - 1) + r_cov
    p_xh = xa.T @ hxa / (n_ens - 1)
    k = p_xh @ np.linalg.solve(p_hh, np.eye(o.size))
    x_new = np.empty_like(x)
    for j in range(n_ens):
        pert = o + rng.multivariate_normal(np.zeros(o.size), r_cov)
        x_new[j] = x[j] + k @ (pert - hx[j])
    return np.asarray(x_new)


def lm_enrml(
    prior_mean: ArrayLike,
    prior_cov: ArrayLike,
    obs: ArrayLike,
    obs_cov: ArrayLike,
    forward: Callable[[_Float], Any],
    n_ens: int = 80,
    n_iter: int = 12,
    lam0: float = 1.0,
    lam_up: float = 4.0,
    lam_dn: float = 0.4,
    seed: int = 0,
) -> _Float:
    """Levenberg-Marquardt ensemble randomized maximum likelihood (LM-EnRML).

    Samples a Gaussian prior ensemble, forms the empirical sensitivity by SVD,
    and applies damped Kalman-type updates with an adaptive ``lam`` schedule.
    Returns the final ensemble ``(n_ens, n_param)``.
    """
    rng = np.random.default_rng(seed)
    mp = _arr(prior_mean)
    cp = _arr(prior_cov)
    o = _arr(obs)
    r_cov = _arr(obs_cov)
    r_inv = np.linalg.inv(r_cov)
    ens = rng.multivariate_normal(mp, cp, size=n_ens)

    def predict(e: _Float) -> _Float:
        return np.array([_arr(forward(m)) for m in e])

    def data_misfit(pred: _Float) -> float:
        return float(np.mean([(o - d) @ r_inv @ (o - d) for d in pred]))

    d_pred = predict(ens)
    cur = data_misfit(d_pred)
    lam = lam0
    for _ in range(n_iter):
        ea = ens - ens.mean(0)
        da = d_pred - d_pred.mean(0)
        u, s, vt = np.linalg.svd(ea, full_matrices=False)
        s_inv = s / (s**2 + 1e-8)
        g = (da.T @ u) * s_inv @ vt
        s_mat = g @ cp @ g.T + (1.0 + lam) * r_cov
        k = cp @ g.T @ np.linalg.inv(s_mat)
        ens_new = np.empty_like(ens)
        for j in range(n_ens):
            pert = o + rng.multivariate_normal(np.zeros(o.size), r_cov)
            ens_new[j] = ens[j] + k @ (pert - d_pred[j])
        d_new = predict(ens_new)
        new = data_misfit(d_new)
        if new < cur:
            ens, d_pred, cur = ens_new, d_new, new
            lam *= lam_dn
        else:
            lam *= lam_up
    return np.asarray(ens)
