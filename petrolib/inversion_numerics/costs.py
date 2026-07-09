"""Cost / misfit functions and regularization-parameter schedules.

Data-misfit measures (L2, RMS, relative, chi-square) and the two regularization
schedules the corpus uses: Habashy-Abubakar multiplicative cooling and the
discrepancy-principle (BRD) bisection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linear import tikhonov_solve

_Float = NDArray[np.float64]


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


def misfit(
    sim: ArrayLike,
    obs: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    kind: str = "l2",
    log_space: bool = False,
) -> float:
    """Data misfit between a simulation and observations.

    ``kind``:
      * ``'l2'``     -- weighted sum of squares ``sum(w*(sim-obs)^2)``
      * ``'rms'``    -- ``sqrt(mean(w*(sim-obs)^2))``
      * ``'rel_data'``-- ``sum(w*((sim-obs)/obs)^2)`` (per-datum relative)
      * ``'rel_norm'``-- ``||sim-obs|| / ||obs||`` (relative 2-norm)
      * ``'chi2'``   -- ``sum(w*(sim-obs)^2)`` with ``w = 1/sigma^2``

    ``log_space=True`` compares ``log10`` of the values first.
    """
    s = _arr(sim)
    o = _arr(obs)
    if log_space:
        s = np.log10(s)
        o = np.log10(o)
    r = s - o
    w = np.ones_like(r) if weights is None else _arr(weights)
    if kind == "l2":
        return float(np.sum(w * r**2))
    if kind == "rms":
        return float(np.sqrt(np.mean(w * r**2)))
    if kind == "rel_data":
        return float(np.sum(w * (r / o) ** 2))
    if kind == "rel_norm":
        return float(np.linalg.norm(r) / np.linalg.norm(o))
    if kind == "chi2":
        return float(np.sum(w * r**2))
    raise ValueError(f"unknown kind {kind!r}; use l2/rms/rel_data/rel_norm/chi2")


def reg_lambda_multiplicative(
    misfit_value: float, alpha: float, beta: float, lam_max: float = np.inf
) -> float:
    """Habashy-Abubakar multiplicative cooling ``lam = min(alpha*misfit^beta, lam_max)``."""
    return float(min(alpha * misfit_value**beta, lam_max))


def reg_lambda_brd(
    a: ArrayLike,
    b: ArrayLike,
    chi2_target: float,
    bracket: tuple[float, float] = (1e-6, 1e3),
    *,
    max_iter: int = 50,
) -> float:
    """Discrepancy-principle regularization weight by geometric bisection.

    Finds ``lam`` such that the Tikhonov data misfit ``||A x_lam - b||^2`` equals
    ``chi2_target`` (e.g. ``n*sigma^2``), bisecting on ``log(lam)`` within
    ``bracket``.
    """
    a_arr = _arr(a)
    b_arr = _arr(b)
    lo, hi = bracket

    def chi2(lam: float) -> float:
        x = tikhonov_solve(a_arr, b_arr, lam)
        return float(np.sum((a_arr @ x - b_arr) ** 2))

    for _ in range(max_iter):
        mid = float(np.sqrt(lo * hi))
        if chi2(mid) > chi2_target:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))
