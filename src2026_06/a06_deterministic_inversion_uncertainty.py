"""
Becoming Certain About Deterministic Inversion Uncertainty

Reference:
    Bower, M., Xie, H., Cuevas, N., Hong, X., Harms, K., Gremillion, J.,
    and Viandante, M. (2026). Becoming Certain About Deterministic Inversion
    Uncertainty. Petrophysics, 67(3), 560-570.
    DOI: 10.30632/PJV67N3-2026a6

The paper estimates the uncertainty of a *deterministic* parametric inversion
of deep / ultradeep azimuthal resistivity data. Reported findings encoded
here:
    - In simple 1D environments, ~50 initial guesses are sufficient to
      describe the *average* estimate of the resistivity distribution.
    - A wider distribution (P5-P95) is obtained by an a-posteriori sampling
      of feasible models around that average model (models whose data misfit
      stays within the noise level).

This module implements:
    - A multistart deterministic inversion driver (Gauss-Newton / damped
      least squares on a user-supplied forward model).
    - Aggregation of the multistart solutions into an average model.
    - A-posteriori feasible-set sampling that keeps models whose misfit is
      below a noise-defined threshold, then reports P5/P50/P95 bands.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Deterministic (damped least squares) inversion
# ---------------------------------------------------------------------------

def _jacobian(forward: Callable[[np.ndarray], np.ndarray],
              m: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Finite-difference Jacobian of the forward operator at model m."""
    f0 = forward(m)
    J = np.zeros((f0.size, m.size))
    for j in range(m.size):
        dm = np.zeros_like(m)
        dm[j] = eps * max(abs(m[j]), 1.0)
        J[:, j] = (forward(m + dm) - f0) / dm[j]
    return J


def deterministic_inversion(forward: Callable[[np.ndarray], np.ndarray],
                            data: np.ndarray, m0: np.ndarray,
                            bounds: Sequence[Tuple[float, float]],
                            lam: float = 1e-2, n_iter: int = 30,
                            tol: float = 1e-8
                            ) -> Tuple[np.ndarray, float]:
    """
    Damped Gauss-Newton (Levenberg-Marquardt) deterministic inversion.

    Parameters
    ----------
    forward : model -> predicted data.
    data    : observed data vector.
    m0      : initial guess.
    bounds  : per-parameter (lo, hi) box constraints.
    lam     : damping factor.
    n_iter  : maximum iterations.
    tol     : stop when the chi (rms misfit) change falls below this.

    Returns
    -------
    (model, rms_misfit)
    """
    m = np.array(m0, dtype=float)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    prev = np.inf
    for _ in range(n_iter):
        r = data - forward(m)
        rms = float(np.sqrt(np.mean(r ** 2)))
        if abs(prev - rms) < tol:
            break
        prev = rms
        J = _jacobian(forward, m)
        JtJ = J.T @ J
        step = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ)) + 1e-12 * np.eye(m.size),
                               J.T @ r)
        m = np.clip(m + step, lo, hi)
    return m, float(np.sqrt(np.mean((data - forward(m)) ** 2)))


def multistart_inversion(forward, data, bounds, n_starts: int = 50,
                         seed: int = 0) -> Dict[str, np.ndarray]:
    """
    Run the deterministic inversion from `n_starts` random initial guesses
    (default 50, as reported sufficient for simple 1D cases) and aggregate.

    Returns
    -------
    dict with:
        'models'  : (n_starts, n_param) array of converged models,
        'misfits' : (n_starts,) rms misfits,
        'average' : misfit-weighted average model (the deterministic estimate).
    """
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    models, misfits = [], []
    for _ in range(n_starts):
        m0 = lo + rng.random(len(bounds)) * (hi - lo)
        m, rms = deterministic_inversion(forward, data, m0, bounds)
        models.append(m)
        misfits.append(rms)
    models = np.array(models)
    misfits = np.array(misfits)
    w = 1.0 / (misfits + 1e-12)
    average = np.average(models, axis=0, weights=w)
    return {"models": models, "misfits": misfits, "average": average}


# ---------------------------------------------------------------------------
# 2. A-posteriori feasible-set sampling for P5-P95 bands
# ---------------------------------------------------------------------------

def feasible_set_sampling(forward, data, m_avg: np.ndarray,
                          bounds: Sequence[Tuple[float, float]],
                          noise_level: float, n_samples: int = 2000,
                          step_frac: float = 0.15, seed: int = 1
                          ) -> Dict[str, np.ndarray]:
    """
    Sample feasible models around the average solution and keep those whose
    rms data misfit stays below the noise level (the "equivalent" models).

    Parameters
    ----------
    forward     : model -> data.
    data        : observed data.
    m_avg       : average (deterministic) model to sample around.
    bounds      : per-parameter box constraints.
    noise_level : rms misfit threshold defining feasibility.
    n_samples   : number of trial perturbations.
    step_frac   : perturbation size as a fraction of each parameter range.

    Returns
    -------
    dict with 'accepted' models and the P5/P50/P95 per parameter.
    """
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    scale = step_frac * (hi - lo)
    accepted = [m_avg.copy()]
    for _ in range(n_samples):
        trial = np.clip(m_avg + rng.normal(0.0, scale), lo, hi)
        rms = float(np.sqrt(np.mean((data - forward(trial)) ** 2)))
        if rms <= noise_level:
            accepted.append(trial)
    A = np.array(accepted)
    return {
        "accepted": A,
        "p5": np.percentile(A, 5, axis=0),
        "p50": np.percentile(A, 50, axis=0),
        "p95": np.percentile(A, 95, axis=0),
        "n_accepted": len(A),
    }


# ---------------------------------------------------------------------------
# 3. Convenience: full workflow example
# ---------------------------------------------------------------------------

def example_workflow():
    """Run a complete example and print key results."""
    print("=" * 64)
    print("Deterministic Inversion Uncertainty (multistart + a-posteriori)")
    print("Ref: Bower et al., Petrophysics 67(3) 2026")
    print("=" * 64)

    # Simple 1D two-parameter "resistivity distribution" forward model:
    # log-conductivity response at several offsets to [R_top, R_bottom].
    offsets = np.linspace(1.0, 10.0, 12)

    def forward(m):
        r_top, r_bottom = m
        return np.log(r_top) * np.exp(-offsets / 6.0) + \
            np.log(r_bottom) * (1.0 - np.exp(-offsets / 6.0))

    true_m = np.array([5.0, 50.0])
    rng = np.random.default_rng(42)
    noise = 0.01
    data = forward(true_m) + rng.normal(0.0, noise, offsets.size)

    bounds = [(1.0, 20.0), (10.0, 200.0)]
    ms = multistart_inversion(forward, data, bounds, n_starts=50)
    print(f"\n50 initial guesses -> average model:")
    print(f"  R_top    = {ms['average'][0]:.2f}  (true 5.0)")
    print(f"  R_bottom = {ms['average'][1]:.2f}  (true 50.0)")
    print(f"  mean rms misfit = {ms['misfits'].mean():.4f}")

    fs = feasible_set_sampling(forward, data, ms["average"], bounds,
                               noise_level=2.5 * noise)
    print(f"\nFeasible-set (P5-P95) uncertainty bands "
          f"({fs['n_accepted']} models):")
    for i, name in enumerate(["R_top", "R_bottom"]):
        print(f"  {name:<9s} P5={fs['p5'][i]:7.2f}  "
              f"P50={fs['p50'][i]:7.2f}  P95={fs['p95'][i]:7.2f}")

    return ms


if __name__ == "__main__":
    example_workflow()
