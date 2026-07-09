"""Global / gradient optimization: particle swarm and gradient descent.

Derivative-free particle-swarm optimization and a finite-difference gradient
descent (with optional backtracking line search) for the objectives the corpus
minimizes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .nonlinear import fd_gradient

_Float = NDArray[np.float64]


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


def pso(
    objective: Callable[[_Float], float],
    bounds: list[tuple[float, float]],
    n_particles: int = 30,
    n_iter: int = 200,
    omega: float = 0.7,
    inertia_decay: float | None = None,
    c1: float = 1.5,
    c2: float = 1.5,
    seed: int = 0,
) -> tuple[_Float, float, _Float]:
    """Particle-swarm optimization over box ``bounds``.

    Returns ``(x_best, f_best, history)`` where ``history`` is the best objective
    per iteration.  ``inertia_decay`` optionally multiplies ``omega`` each step.
    """
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    dim = len(bounds)
    x = lo + rng.random((n_particles, dim)) * (hi - lo)
    v = np.zeros((n_particles, dim))
    pbest = x.copy()
    pbest_f = np.array([objective(row) for row in x])
    g = int(np.argmin(pbest_f))
    gbest = pbest[g].copy()
    gbest_f = float(pbest_f[g])
    history = [gbest_f]
    w = omega
    for _ in range(n_iter):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = np.clip(x + v, lo, hi)
        f = np.array([objective(row) for row in x])
        improved = f < pbest_f
        pbest[improved] = x[improved]
        pbest_f[improved] = f[improved]
        gi = int(np.argmin(pbest_f))
        if pbest_f[gi] < gbest_f:
            gbest = pbest[gi].copy()
            gbest_f = float(pbest_f[gi])
        if inertia_decay is not None:
            w *= inertia_decay
        history.append(gbest_f)
    return gbest, gbest_f, np.array(history)


def gradient_descent(
    f: Callable[[_Float], float],
    x0: ArrayLike,
    lr: float | None = None,
    backtracking: bool = False,
    max_iter: int = 4000,
    tol: float = 1e-10,
) -> _Float:
    """Finite-difference gradient descent of a scalar objective ``f``.

    ``backtracking=True`` shrinks the step by an Armijo condition each iteration;
    otherwise a fixed ``lr`` (default ``1e-2``) is used.
    """
    x = _arr(x0).copy()
    fx = f(x)
    rate = 1e-2 if lr is None else lr
    for _ in range(max_iter):
        g = fd_gradient(f, x, eps=1e-6)
        if backtracking:
            step = rate
            while step > 1e-16:
                if f(x - step * g) < fx - 1e-4 * step * (g @ g):
                    break
                step *= 0.5
        else:
            step = rate
        x_new = x - step * g
        fn = f(x_new)
        if abs(fx - fn) < tol:
            x, fx = x_new, fn
            break
        x, fx = x_new, fn
    return np.asarray(x)
