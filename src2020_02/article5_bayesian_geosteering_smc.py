"""
Article 5: Bayesian Geosteering Using Sequential Monte Carlo Methods
Akkam Veettil, Clark (2020)
DOI: 10.30632/PJV61N1-2020a4

Geosteering updates the estimate of the distance to a geological boundary as the
well advances, using noisy logging-while-drilling measurements.  A Sequential
Monte Carlo (particle filter) represents the posterior distribution of the
boundary distance with weighted particles, propagating them with a motion model,
reweighting them by the measurement likelihood, and resampling to combat
particle degeneracy.

Implements:

  - Particle propagation (random-walk state model)
  - Bayesian weight update from a Gaussian measurement likelihood
  - Effective sample size and systematic resampling
  - Posterior (weighted-mean) estimate of the distance to boundary

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard particle-filter / SMC framework the paper's
title describes.
"""

import numpy as np


# ---------------------------------------------- SMC steps ---------------

def propagate(particles, step, sigma_process, rng):
    """Motion model: advance each particle by `step` plus process noise."""
    return particles + step + rng.normal(0.0, sigma_process, particles.shape)


def update_weights(weights, particles, measurement, sigma_meas, measure_fn):
    """Bayesian reweight by the Gaussian measurement likelihood."""
    predicted = measure_fn(particles)
    like = np.exp(-0.5 * ((measurement - predicted) / sigma_meas) ** 2)
    w = np.asarray(weights, float) * like
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


def effective_sample_size(weights):
    """Effective sample size  1/sum(w^2)."""
    w = np.asarray(weights, float)
    return 1.0 / np.sum(w ** 2)


def systematic_resample(particles, weights, rng):
    """Systematic resampling; returns (new_particles, uniform_weights)."""
    n = len(particles)
    positions = (np.arange(n) + rng.uniform()) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    idx = np.searchsorted(cumsum, positions)
    return particles[idx].copy(), np.full(n, 1.0 / n)


def estimate(particles, weights):
    """Posterior mean estimate  sum(w*particle)."""
    return float(np.sum(np.asarray(weights, float) * np.asarray(particles, float)))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Bayesian Geosteering via Sequential Monte Carlo")
    print("=" * 60)

    rng = np.random.default_rng(0)
    n = 2000

    # True boundary distance decreases as the well approaches it (10 -> 0 m)
    measure_fn = lambda d: d                          # measurement ~ distance
    true_dist = 10.0
    particles = rng.uniform(0.0, 20.0, n)             # diffuse prior
    weights = np.full(n, 1.0 / n)

    est_history, truth = [], []
    for k in range(20):
        true_dist -= 0.5                              # drilling toward boundary
        particles = propagate(particles, step=-0.5, sigma_process=0.3, rng=rng)
        meas = true_dist + rng.normal(0.0, 0.5)       # noisy LWD reading
        weights = update_weights(weights, particles, meas, 0.5, measure_fn)
        if effective_sample_size(weights) < n / 2:
            particles, weights = systematic_resample(particles, weights, rng)
        est_history.append(estimate(particles, weights))
        truth.append(true_dist)

    est_history = np.array(est_history); truth = np.array(truth)
    rmse = float(np.sqrt(np.mean((est_history - truth) ** 2)))
    print(f"  tracking RMSE          = {rmse:.3f} m")
    print(f"  final est / true       = {est_history[-1]:.2f} / {truth[-1]:.2f} m")
    assert rmse < 0.5                                 # tracks within the noise level

    # Weight update concentrates on particles near the measurement
    w0 = np.full(5, 0.2)
    p = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    w = update_weights(w0, p, measurement=2.0, sigma_meas=0.5, measure_fn=lambda x: x)
    assert np.argmax(w) == 2 and abs(w.sum() - 1.0) < 1e-9

    # Resampling resets to uniform weights and preserves particle count
    pr, wr = systematic_resample(p, w, rng)
    assert len(pr) == 5 and np.allclose(wr, 0.2)
    print("  PASS")
    return {"rmse": rmse, "final_est": float(est_history[-1])}


if __name__ == "__main__":
    test_all()
