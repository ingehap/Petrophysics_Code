"""Nonlinear inversion: finite-difference derivatives, LM, Occam, search.

Deterministic nonlinear solvers: finite-difference Jacobian / gradient,
Levenberg-Marquardt with box bounds and optional log-parameters, an Occam
smoothest-model iteration, brute-force grid search, multistart, and
feasible-set (equivalence) sampling for uncertainty bounds.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linear import difference_operator

_Float = NDArray[np.float64]
_Forward = Callable[[_Float], Any]


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


class InvResult(NamedTuple):
    """Result of a deterministic nonlinear inversion."""

    model: _Float
    misfit: float
    n_iter: int
    converged: bool
    lam: float


def fd_jacobian(
    forward: _Forward,
    m: ArrayLike,
    eps: float = 1e-6,
    scheme: str = "central",
    relative: bool = True,
) -> _Float:
    """Finite-difference Jacobian ``d forward / d m`` (``n_data x n_param``)."""
    m_arr = _arr(m)
    f0 = _arr(forward(m_arr))
    jac = np.zeros((f0.size, m_arr.size))
    for i in range(m_arr.size):
        dm = eps * max(abs(m_arr[i]), 1.0) if relative else eps
        mp = m_arr.copy()
        mp[i] += dm
        if scheme == "central":
            mm = m_arr.copy()
            mm[i] -= dm
            jac[:, i] = (_arr(forward(mp)) - _arr(forward(mm))) / (2.0 * dm)
        elif scheme == "forward":
            jac[:, i] = (_arr(forward(mp)) - f0) / dm
        else:
            raise ValueError(f"unknown scheme {scheme!r}; use 'central' or 'forward'")
    return jac


def fd_gradient(f: Callable[[_Float], float], x: ArrayLike, **kw: Any) -> _Float:
    """Finite-difference gradient of a scalar objective ``f``."""
    return fd_jacobian(lambda z: np.array([f(z)]), x, **kw).ravel()


def levenberg_marquardt(
    forward: _Forward,
    data: ArrayLike,
    m0: ArrayLike,
    *,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    log_params: bool = False,
    lam0: float = 1e-3,
    lam_up: float = 5.0,
    lam_dn: float = 0.5,
    damping: str = "diag",
    max_iter: int = 50,
    tol: float = 1e-8,
) -> InvResult:
    """Levenberg-Marquardt least-squares inversion of ``forward(m) ~ data``.

    ``bounds=(lo, hi)`` box-constrains the (physical) model; ``log_params=True``
    inverts for ``log(m)``; ``damping`` is ``'diag'`` (Marquardt) or ``'identity'``.
    Returns an :class:`InvResult`.
    """
    d = _arr(data)
    lo, hi = (None, None) if bounds is None else (_arr(bounds[0]), _arr(bounds[1]))

    def to_phys(p: _Float) -> _Float:
        m = np.exp(p) if log_params else p
        if lo is not None and hi is not None:
            m = np.clip(m, lo, hi)
        return np.asarray(m)

    def to_inv(m: _Float) -> _Float:
        return np.asarray(np.log(m) if log_params else m)

    def resid(p: _Float) -> _Float:
        return np.asarray(_arr(forward(to_phys(p))) - d)

    x = to_inv(_arr(m0))
    r = resid(x)
    cost = float(r @ r)
    lam = lam0
    converged = False
    n_iter = 0
    for step_i in range(1, max_iter + 1):
        n_iter = step_i
        jac = fd_jacobian(lambda p: _arr(forward(to_phys(p))), x)
        jtj = jac.T @ jac
        d_mat = np.diag(np.diag(jtj)) if damping == "diag" else np.eye(x.size)
        step = np.linalg.solve(jtj + lam * d_mat, -(jac.T @ r))
        x_new = to_inv(to_phys(x + step))
        r_new = resid(x_new)
        cost_new = float(r_new @ r_new)
        if cost_new < cost:
            x, r = x_new, r_new
            lam *= lam_dn
            if cost - cost_new < tol:
                cost = cost_new
                converged = True
                break
            cost = cost_new
        else:
            lam *= lam_up
    return InvResult(to_phys(x), cost, n_iter, converged, lam)


def occam(
    forward: _Forward,
    data: ArrayLike,
    m0: ArrayLike,
    noise_level: float,
    *,
    reg_order: int = 2,
    lam0: float = 100.0,
    max_iter: int = 50,
) -> _Float:
    """Occam smoothest-model iteration toward a target data misfit.

    Minimizes the ``reg_order`` roughness ``||L m||^2`` subject to fitting ``data``
    to within the ``noise_level`` (chi-square target = number of data).
    """
    d = _arr(data)
    m = _arr(m0).copy()
    ell = difference_operator(m.size, reg_order)
    ltl = ell.T @ ell
    w = 1.0 / noise_level
    target = float(d.size)
    lam = lam0
    for _ in range(max_iter):
        jac = fd_jacobian(forward, m) * w
        rw = w * (d - _arr(forward(m)))
        jtj = jac.T @ jac
        dm = np.linalg.solve(jtj + lam * ltl + 1e-10 * np.eye(m.size), jac.T @ rw - lam * (ltl @ m))
        m = m + 0.5 * dm
        chi2 = float(np.sum((w * (d - _arr(forward(m)))) ** 2))
        if chi2 <= target:
            break
        lam *= 0.8
    return np.asarray(m)


def grid_search(
    forward: _Forward, data: ArrayLike, grids: list[ArrayLike], misfit: str = "log_l2"
) -> tuple[_Float, float]:
    """Brute-force grid search minimizing the misfit over the Cartesian ``grids``."""
    d = _arr(data)
    best = _arr([np.asarray(g, np.float64).reshape(-1)[0] for g in grids])
    best_err = np.inf
    for combo in itertools.product(*[np.asarray(g, np.float64) for g in grids]):
        m = np.array(combo, np.float64)
        pred = _arr(forward(m))
        if misfit == "log_l2":
            err = float(np.sum((np.log(pred) - np.log(d)) ** 2))
        else:
            err = float(np.sum((pred - d) ** 2))
        if err < best_err:
            best_err = err
            best = m
    return best, best_err


def multistart(
    solver: Callable[[_Float], InvResult | tuple[ArrayLike, float]],
    bounds: list[tuple[float, float]],
    n_starts: int = 50,
    seed: int = 0,
    aggregate: str = "misfit_weighted",
) -> _Float:
    """Run ``solver`` from many random starts and aggregate the models.

    ``aggregate='best'`` returns the lowest-misfit model; ``'misfit_weighted'``
    returns a softmax(-misfit) weighted average.
    """
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    models = []
    misfits = []
    for _ in range(n_starts):
        m0 = lo + rng.random(len(bounds)) * (hi - lo)
        res = solver(m0)
        if isinstance(res, InvResult):
            models.append(_arr(res.model))
            misfits.append(res.misfit)
        else:
            models.append(_arr(res[0]))
            misfits.append(float(res[1]))
    models_a = np.array(models)
    misfits_a = np.array(misfits)
    if aggregate == "best":
        return np.asarray(models_a[int(np.argmin(misfits_a))])
    w = np.exp(-(misfits_a - misfits_a.min()))
    w = w / w.sum()
    return np.asarray(w @ models_a)


def feasible_set_sampling(
    forward: _Forward,
    data: ArrayLike,
    m_center: ArrayLike,
    bounds: list[tuple[float, float]],
    noise_level: float,
    n_samples: int = 2000,
    step_frac: float = 0.15,
    seed: int = 1,
) -> dict[str, _Float]:
    """Equivalence / feasible-set sampling around a solution for uncertainty bounds.

    Random-walks around ``m_center`` within ``bounds``, keeping models whose RMS
    data residual is within ``2*noise_level``, and returns per-parameter
    ``P5/P50/P95`` and the accepted ``models``.
    """
    rng = np.random.default_rng(seed)
    d = _arr(data)
    center = _arr(m_center)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    span = (hi - lo) * step_frac
    accepted = []
    for _ in range(n_samples):
        cand = np.clip(center + rng.normal(0.0, span), lo, hi)
        rms = float(np.sqrt(np.mean((_arr(forward(cand)) - d) ** 2)))
        if rms <= 2.0 * noise_level:
            accepted.append(cand)
    models = np.array(accepted) if accepted else center[None, :]
    return {
        "P5": np.percentile(models, 5, axis=0),
        "P50": np.percentile(models, 50, axis=0),
        "P95": np.percentile(models, 95, axis=0),
        "models": models,
    }
