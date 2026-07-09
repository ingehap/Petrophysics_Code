"""Grid PDE solvers: 2D effective conductivity and 1D diffusion.

The recurring numerical grid solvers: a 2D effective-conductivity Laplace solve
by harmonic-mean-face Jacobi relaxation, and an explicit 1D diffusion time-step
with its CFL number.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


def cfl_number(alpha: float, dt: float, dx: float) -> float:
    """Diffusion CFL number ``alpha*dt/dx^2`` (explicit stability needs ``<= 0.5``)."""
    return float(alpha * dt / dx**2)


def effective_conductivity_2d(sigma_map: ArrayLike, n_iter: int = 5000, tol: float = 1e-7) -> float:
    """Effective vertical conductivity of a 2D conductivity map.

    Solves the steady Laplace problem ``div(sigma grad V)=0`` with harmonic-mean
    face conductivities by Jacobi relaxation, top/bottom Dirichlet electrodes
    (``V=1`` / ``V=0``) and no-flux sides, then returns the effective conductivity
    from the top-layer current.
    """
    s = _arr(sigma_map)
    ny, nx = s.shape
    g_s = 2.0 * s[:-1] * s[1:] / (s[:-1] + s[1:])  # south-face conductivity, (ny-1, nx)
    g_e = 2.0 * s[:, :-1] * s[:, 1:] / (s[:, :-1] + s[:, 1:])  # east-face, (ny, nx-1)
    v = np.tile(np.linspace(1.0, 0.0, ny)[:, None], (1, nx))
    for _ in range(n_iter):
        v_old = v.copy()
        num = np.zeros_like(v)
        den = np.zeros_like(v)
        num[1:] += g_s * v[:-1]
        den[1:] += g_s
        num[:-1] += g_s * v[1:]
        den[:-1] += g_s
        num[:, 1:] += g_e * v[:, :-1]
        den[:, 1:] += g_e
        num[:, :-1] += g_e * v[:, 1:]
        den[:, :-1] += g_e
        v = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), v)
        v[0] = 1.0
        v[-1] = 0.0
        if np.max(np.abs(v - v_old)) < tol:
            break
    current = float(np.sum(g_s[0] * (v[0] - v[1])))
    return float(current * (ny - 1) / nx)


def diffusion_step_1d(
    u: ArrayLike,
    alpha: float,
    dt: float,
    dx: float,
    source: ArrayLike | None = None,
    bc: str = "neumann",
) -> _Float:
    """One explicit finite-difference step of the 1D diffusion equation.

    ``u_new = u + alpha*dt/dx^2 * laplacian(u) + dt*source``.  ``bc='neumann'``
    uses no-flux ends; ``bc='dirichlet'`` holds the endpoint values fixed.
    """
    u_arr = _arr(u)
    lap = np.zeros_like(u_arr)
    lap[1:-1] = u_arr[2:] - 2.0 * u_arr[1:-1] + u_arr[:-2]
    if bc == "neumann":
        lap[0] = u_arr[1] - u_arr[0]
        lap[-1] = u_arr[-2] - u_arr[-1]
    elif bc == "dirichlet":
        lap[0] = 0.0
        lap[-1] = 0.0
    else:
        raise ValueError(f"unknown bc {bc!r}; use 'neumann' or 'dirichlet'")
    u_new = u_arr + alpha * dt / dx**2 * lap
    if source is not None:
        u_new = u_new + dt * _arr(source)
    if bc == "dirichlet":
        u_new[0] = u_arr[0]
        u_new[-1] = u_arr[-1]
    return np.asarray(u_new)
