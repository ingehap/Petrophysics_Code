"""Domain-agnostic numerical inversion, optimization, and fitting solvers.

The canonical home for the linear-algebra, cost, nonlinear, stochastic,
optimization, curve-fitting, and grid-PDE machinery that the article
implementations re-derive (see LIBRARY_MERGE_PLAN.md).  Submodules:

* :mod:`~petrolib.inversion_numerics.linear` -- Tikhonov / MAP / SVD / NNLS /
  unmixing / convolution / difference operators.
* :mod:`~petrolib.inversion_numerics.costs` -- misfit measures and
  regularization-parameter schedules.
* :mod:`~petrolib.inversion_numerics.nonlinear` -- FD Jacobian, Levenberg-
  Marquardt, Occam, grid search, multistart, feasible-set sampling.
* :mod:`~petrolib.inversion_numerics.stochastic` -- likelihoods / priors,
  Metropolis, MALA, EnKF, LM-EnRML.
* :mod:`~petrolib.inversion_numerics.optimize` -- particle swarm, gradient descent.
* :mod:`~petrolib.inversion_numerics.fitting` -- line / power-law / exponential /
  cosine fits.
* :mod:`~petrolib.inversion_numerics.pde` -- 2D effective conductivity, 1D diffusion.

Convention: ``lam`` is applied once (never squared).  scipy is imported lazily
inside the functions that need it (NNLS, ``curve_fit``).
"""

from __future__ import annotations

from . import costs, fitting, linear, nonlinear, optimize, pde, stochastic
from .fitting import LineFit
from .nonlinear import InvResult
from .stochastic import Chain

__all__ = [
    "Chain",
    "InvResult",
    "LineFit",
    "costs",
    "fitting",
    "linear",
    "nonlinear",
    "optimize",
    "pde",
    "stochastic",
]
