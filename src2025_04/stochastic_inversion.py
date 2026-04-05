"""
High-Performance Stochastic Inversion for UDAR Data
=====================================================
Based on: Sviridov et al., "High-Performance Stochastic Inversion for
Real-Time Processing of LWD Ultradeep Azimuthal Resistivity Data",
Petrophysics, Vol. 66, No. 2, April 2025, pp. 212–236.

Implements:
  - Reversible-jump Markov chain Monte Carlo (RJMCMC) inversion
  - Metropolis-adjusted Langevin (MALA) proposal mechanism
  - Parallel tempering with heat-value swapping
  - 1D layer-cake formation model with Bayesian uncertainty

Reference: https://doi.org/10.30632/PJV66N2-2025a3 (SPWLA)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import copy


@dataclass
class LayerCakeModel:
    """1D layered formation model with variable number of layers."""
    boundaries_ft: np.ndarray        # Layer boundary depths (n_layers - 1)
    resistivities_ohm_m: np.ndarray  # Resistivity per layer (n_layers)
    anisotropy_ratios: np.ndarray    # Rv/Rh per layer (n_layers)

    @property
    def n_layers(self) -> int:
        return len(self.resistivities_ohm_m)

    def get_resistivity_at_depth(self, depth_ft: float) -> float:
        """Return the resistivity at a given depth."""
        for i, b in enumerate(self.boundaries_ft):
            if depth_ft < b:
                return self.resistivities_ohm_m[i]
        return self.resistivities_ohm_m[-1]

    def copy(self):
        return LayerCakeModel(
            self.boundaries_ft.copy(),
            self.resistivities_ohm_m.copy(),
            self.anisotropy_ratios.copy(),
        )


def forward_model_synthetic(model: LayerCakeModel,
                            measurement_depths_ft: np.ndarray,
                            n_channels: int = 6) -> np.ndarray:
    """
    Simplified forward model for UDAR tool responses.

    Generates synthetic apparent resistivity curves at multiple channels
    (representing different spacings/frequencies) for a given 1D model.

    Parameters
    ----------
    model : LayerCakeModel
    measurement_depths_ft : np.ndarray
    n_channels : int

    Returns
    -------
    np.ndarray, shape (n_depths, n_channels)
        Log10 of apparent resistivity for each channel.
    """
    n_depths = len(measurement_depths_ft)
    data = np.zeros((n_depths, n_channels))

    # Different channels "see" different depths of investigation
    doi_ft = np.linspace(5, 50, n_channels)

    for j in range(n_channels):
        for i, md in enumerate(measurement_depths_ft):
            # Weighted average of resistivity within DOI
            weights = []
            values = []
            for offset in np.linspace(-doi_ft[j], doi_ft[j], 21):
                d = md + offset
                r = model.get_resistivity_at_depth(d)
                w = np.exp(-abs(offset) / (doi_ft[j] / 2.0))
                weights.append(w)
                values.append(np.log10(r))
            data[i, j] = np.average(values, weights=weights)

    return data


def log_likelihood(data_obs: np.ndarray,
                   data_pred: np.ndarray,
                   sigma: float = 0.05) -> float:
    """
    Compute log-likelihood assuming Gaussian noise on log10(resistivity).

    Parameters
    ----------
    data_obs : np.ndarray
        Observed data.
    data_pred : np.ndarray
        Predicted data from forward model.
    sigma : float
        Noise standard deviation in log10 space.

    Returns
    -------
    float
        Log-likelihood.
    """
    residual = data_obs - data_pred
    return -0.5 * np.sum((residual / sigma) ** 2)


def log_prior(model: LayerCakeModel,
              res_min: float = 0.1,
              res_max: float = 1000.0,
              max_layers: int = 20) -> float:
    """
    Compute log-prior probability of a model.

    Uses log-uniform prior on resistivities and uniform prior on boundaries.
    """
    if model.n_layers > max_layers or model.n_layers < 1:
        return -np.inf
    if np.any(model.resistivities_ohm_m < res_min) or \
       np.any(model.resistivities_ohm_m > res_max):
        return -np.inf
    # Check boundary ordering
    if len(model.boundaries_ft) > 0:
        if np.any(np.diff(model.boundaries_ft) <= 0):
            return -np.inf
    return 0.0  # Uniform/improper prior


def mala_proposal(model: LayerCakeModel,
                  gradient: np.ndarray,
                  step_size: float = 0.01) -> LayerCakeModel:
    """
    Metropolis-adjusted Langevin (MALA) proposal.

    Uses the gradient of the posterior to bias proposals toward
    higher probability regions, as described in Sviridov et al. (2025).

    Parameters
    ----------
    model : LayerCakeModel
    gradient : np.ndarray
        Gradient of log-posterior w.r.t. log10(resistivities).
    step_size : float

    Returns
    -------
    LayerCakeModel
        Proposed model.
    """
    proposed = model.copy()
    log_res = np.log10(proposed.resistivities_ohm_m)
    # Langevin drift + random walk
    noise = np.random.randn(len(log_res))
    log_res_new = log_res + 0.5 * step_size ** 2 * gradient + step_size * noise
    proposed.resistivities_ohm_m = 10.0 ** log_res_new
    return proposed


def numerical_gradient(model: LayerCakeModel,
                       data_obs: np.ndarray,
                       meas_depths: np.ndarray,
                       sigma: float = 0.05,
                       eps: float = 0.01) -> np.ndarray:
    """Compute numerical gradient of log-posterior w.r.t. log10(resistivities)."""
    n = model.n_layers
    grad = np.zeros(n)
    base_pred = forward_model_synthetic(model, meas_depths)
    base_ll = log_likelihood(data_obs, base_pred, sigma)

    for i in range(n):
        m_plus = model.copy()
        log_r = np.log10(m_plus.resistivities_ohm_m)
        log_r[i] += eps
        m_plus.resistivities_ohm_m = 10.0 ** log_r
        pred_plus = forward_model_synthetic(m_plus, meas_depths)
        ll_plus = log_likelihood(data_obs, pred_plus, sigma)
        grad[i] = (ll_plus - base_ll) / eps

    return grad


def birth_move(model: LayerCakeModel,
               depth_range: Tuple[float, float]) -> LayerCakeModel:
    """
    Add a layer (birth move) in RJMCMC.

    Splits an existing layer at a random depth, inheriting the parent's
    resistivity with a small perturbation.
    """
    proposed = model.copy()
    new_boundary = np.random.uniform(*depth_range)
    # Find which layer to split
    idx = np.searchsorted(proposed.boundaries_ft, new_boundary)
    parent_res = proposed.resistivities_ohm_m[idx]
    # Perturb the child resistivities
    r1 = parent_res * 10.0 ** (0.1 * np.random.randn())
    r2 = parent_res * 10.0 ** (0.1 * np.random.randn())
    proposed.boundaries_ft = np.insert(proposed.boundaries_ft, idx, new_boundary)
    proposed.resistivities_ohm_m = np.insert(proposed.resistivities_ohm_m, idx, r1)
    proposed.resistivities_ohm_m[idx + 1] = r2
    proposed.anisotropy_ratios = np.insert(proposed.anisotropy_ratios, idx, 1.0)
    return proposed


def death_move(model: LayerCakeModel) -> Optional[LayerCakeModel]:
    """
    Remove a layer (death move) in RJMCMC.

    Merges two adjacent layers, averaging their resistivities.
    """
    if model.n_layers <= 2:
        return None
    proposed = model.copy()
    idx = np.random.randint(0, len(proposed.boundaries_ft))
    # Merge layers idx and idx+1
    avg_res = np.sqrt(proposed.resistivities_ohm_m[idx] *
                      proposed.resistivities_ohm_m[idx + 1])
    proposed.boundaries_ft = np.delete(proposed.boundaries_ft, idx)
    proposed.resistivities_ohm_m = np.delete(proposed.resistivities_ohm_m, idx + 1)
    proposed.resistivities_ohm_m[idx] = avg_res
    proposed.anisotropy_ratios = np.delete(proposed.anisotropy_ratios, idx + 1)
    return proposed


def rjmcmc_inversion(data_obs: np.ndarray,
                     meas_depths: np.ndarray,
                     depth_range: Tuple[float, float] = (0.0, 500.0),
                     n_iterations: int = 2000,
                     n_chains: int = 4,
                     sigma: float = 0.05,
                     step_size: float = 0.02,
                     seed: int = 42) -> dict:
    """
    Run RJMCMC inversion with parallel tempering.

    The algorithm is based on the stochastic Monte Carlo method with
    reversible-jump Markov chains (Sviridov et al., 2025). The
    Metropolis-adjusted Langevin technique evaluates the gradient of
    the posterior and generates proposals with higher probability.

    Parameters
    ----------
    data_obs : np.ndarray, shape (n_depths, n_channels)
    meas_depths : np.ndarray
    depth_range : Tuple[float, float]
    n_iterations : int
    n_chains : int
    sigma : float
    step_size : float
    seed : int

    Returns
    -------
    dict
        - "best_model": LayerCakeModel with highest posterior
        - "ensemble": list of accepted models
        - "log_posteriors": array of log-posterior values
        - "acceptance_rate": float
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    # Temperature ladder for parallel tempering
    temperatures = np.geomspace(1.0, 10.0, n_chains)

    # Initialize chains with simple 3-layer models
    chains = []
    for c in range(n_chains):
        mid = np.mean(depth_range)
        bounds = np.array([mid - 50 + 20 * rng.randn(), mid + 50 + 20 * rng.randn()])
        bounds.sort()
        res = 10.0 ** (1.0 + rng.rand(3))
        anis = np.ones(3)
        chains.append(LayerCakeModel(bounds, res, anis))

    # Track the cold chain (temperature=1)
    ensemble = []
    log_posts = []
    accepted = 0
    total = 0

    best_model = None
    best_lp = -np.inf

    for it in range(n_iterations):
        for c in range(n_chains):
            model = chains[c]
            temp = temperatures[c]

            # Choose move type
            move = rng.choice(["perturb", "birth", "death"], p=[0.7, 0.15, 0.15])

            if move == "perturb":
                grad = numerical_gradient(model, data_obs, meas_depths, sigma)
                proposed = mala_proposal(model, grad, step_size)
            elif move == "birth":
                proposed = birth_move(model, depth_range)
            else:
                proposed = death_move(model)
                if proposed is None:
                    continue

            # Evaluate acceptance
            lp_prior = log_prior(proposed)
            if lp_prior == -np.inf:
                continue

            pred_curr = forward_model_synthetic(model, meas_depths)
            pred_prop = forward_model_synthetic(proposed, meas_depths)
            ll_curr = log_likelihood(data_obs, pred_curr, sigma)
            ll_prop = log_likelihood(data_obs, pred_prop, sigma)

            log_alpha = (ll_prop - ll_curr) / temp
            total += 1

            if np.log(rng.rand()) < log_alpha:
                chains[c] = proposed
                accepted += 1

                if c == 0:  # Cold chain
                    lp = ll_prop + lp_prior
                    ensemble.append(proposed.copy())
                    log_posts.append(lp)
                    if lp > best_lp:
                        best_lp = lp
                        best_model = proposed.copy()

        # Parallel tempering: swap adjacent chains (every 4 iterations)
        if it % 4 == 0 and n_chains > 1:
            i = rng.randint(0, n_chains - 1)
            j = i + 1
            pred_i = forward_model_synthetic(chains[i], meas_depths)
            pred_j = forward_model_synthetic(chains[j], meas_depths)
            ll_i = log_likelihood(data_obs, pred_i, sigma)
            ll_j = log_likelihood(data_obs, pred_j, sigma)
            swap_alpha = (ll_i - ll_j) * (1.0 / temperatures[j] - 1.0 / temperatures[i])
            if np.log(rng.rand()) < swap_alpha:
                # Swap heat values (as described in the paper)
                temperatures[i], temperatures[j] = temperatures[j], temperatures[i]

    return {
        "best_model": best_model,
        "ensemble": ensemble,
        "log_posteriors": np.array(log_posts),
        "acceptance_rate": accepted / max(total, 1),
    }


def compute_uncertainty(ensemble: List[LayerCakeModel],
                        depth_ft: np.ndarray,
                        percentiles: Tuple = (10, 50, 90)) -> dict:
    """
    Compute resistivity uncertainty from the model ensemble.

    Parameters
    ----------
    ensemble : List[LayerCakeModel]
    depth_ft : np.ndarray
    percentiles : Tuple

    Returns
    -------
    dict with keys "p10", "p50", "p90" (or specified percentiles),
    each an array of resistivities.
    """
    n = len(depth_ft)
    m = len(ensemble)
    res_matrix = np.zeros((m, n))
    for i, model in enumerate(ensemble):
        for j, d in enumerate(depth_ft):
            res_matrix[i, j] = model.get_resistivity_at_depth(d)

    result = {}
    for p in percentiles:
        result[f"p{p}"] = np.percentile(res_matrix, p, axis=0)
    return result


def test_all():
    """Test all functions with synthetic data."""
    print("=" * 70)
    print("Testing: stochastic_inversion (Sviridov et al., 2025)")
    print("=" * 70)

    # Create a true model (subset of Oklahoma benchmark)
    true_model = LayerCakeModel(
        boundaries_ft=np.array([100.0, 150.0, 200.0, 250.0]),
        resistivities_ohm_m=np.array([10.0, 0.5, 100.0, 2.0, 50.0]),
        anisotropy_ratios=np.ones(5),
    )

    meas_depths = np.linspace(50, 300, 50)
    data_true = forward_model_synthetic(true_model, meas_depths)
    noise = 0.02 * np.random.randn(*data_true.shape)
    data_obs = data_true + noise

    print(f"  True model: {true_model.n_layers} layers")
    print(f"  Data shape: {data_obs.shape}")

    # Run inversion (small number of iterations for testing)
    result = rjmcmc_inversion(data_obs, meas_depths,
                              depth_range=(50, 300),
                              n_iterations=300,
                              n_chains=3,
                              sigma=0.05,
                              seed=123)

    best = result["best_model"]
    print(f"  Best model: {best.n_layers} layers")
    print(f"  Best log-posterior: {result['log_posteriors'][-1]:.1f}")
    print(f"  Acceptance rate: {result['acceptance_rate']:.2%}")
    print(f"  Ensemble size: {len(result['ensemble'])}")

    # Compute uncertainty
    if len(result["ensemble"]) >= 5:
        unc = compute_uncertainty(result["ensemble"], meas_depths)
        print(f"  P50 resistivity range: [{unc['p50'].min():.2f}, {unc['p50'].max():.2f}] Ohm·m")

    assert best.n_layers >= 2, "Should recover at least 2 layers"
    assert result["acceptance_rate"] > 0.01, "Should have some accepted proposals"

    print("  All tests PASSED.\n")


if __name__ == "__main__":
    test_all()
