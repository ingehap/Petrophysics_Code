"""
Article 3: Accelerated Whole-Core Analysis Optimization With Wellsite Tomography
           Instrumentation and Bayesian Inversion
Mendoza, Roininen, Girolami, Heikkinen, Haario (2019)
DOI: 10.30632/PJV60N3-2019a2

Whole-core CT at the wellsite is accelerated by acquiring fewer projections and
reconstructing the core density with a Bayesian (maximum-a-posteriori) inversion
that uses a smoothness prior.  The prior regularizes the under-determined,
noisy problem so a usable image is obtained from far fewer measurements than
classical filtered back-projection needs.

Implements:

  - Linear forward projection  d = G @ m  (+ noise)
  - Bayesian MAP estimate with a Gaussian smoothness prior
  - Posterior covariance / uncertainty
  - Comparison vs unregularized least squares

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the journal metadata and these are faithful
standard-form reconstructions of the Bayesian linear-inversion method the paper
applies.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- forward -----------------

def project(G, m):
    """Linear CT forward projection  d = G @ m."""
    return np.asarray(G, float) @ np.asarray(m, float)


def difference_operator(n):
    """First-difference operator L (smoothness prior precision sqrt)."""
    L = np.zeros((n - 1, n))
    for i in range(n - 1):
        L[i, i] = -1.0
        L[i, i + 1] = 1.0
    return L


# ---------------------------------------------- Bayesian MAP ------------

def map_estimate(G, d, L, noise_var, prior_strength, m_prior=None):
    """Bayesian MAP estimate with a Gaussian smoothness prior.

        m_MAP = (G^T G/noise_var + lambda*L^T L)^-1 (G^T d/noise_var + lambda*L^T L*m_prior)
    """
    G = np.asarray(G, float); d = np.asarray(d, float)
    n = G.shape[1]
    if m_prior is None:
        m_prior = np.zeros(n)
    LtL = L.T @ L
    A = G.T @ G / noise_var + prior_strength * LtL
    b = G.T @ d / noise_var + prior_strength * LtL @ m_prior
    return np.linalg.solve(A, b)


def posterior_covariance(G, L, noise_var, prior_strength):
    """Posterior covariance  (G^T G/noise_var + lambda*L^T L)^-1."""
    A = G.T @ G / noise_var + prior_strength * L.T @ L
    return np.linalg.inv(A)


def least_squares(G, d):
    """Unregularized least-squares (minimum-norm) reconstruction."""
    return petrolib.inversion_numerics.linear.deconvolve(d, G)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Wellsite Tomography Bayesian Inversion")
    print("=" * 60)

    rng = np.random.default_rng(2)
    n = 40
    # True density profile across the core: a smooth ramp with a dense streak
    m_true = 2.3 + 0.2 * np.sin(np.linspace(0, 3, n))
    m_true[18:23] += 0.3

    # Few noisy projections (under-determined: 25 measurements, 40 unknowns)
    G = rng.normal(size=(25, n))
    noise_var = 0.02 ** 2
    d = project(G, m_true) + rng.normal(0, 0.02, 25)

    L = difference_operator(n)
    m_map = map_estimate(G, d, L, noise_var, prior_strength=5.0)
    m_ls = least_squares(G, d)

    err_map = np.linalg.norm(m_map - m_true)
    err_ls = np.linalg.norm(m_ls - m_true)
    print(f"  reconstruction error MAP / LS = {err_map:.3f} / {err_ls:.3f}")
    assert err_map < err_ls                       # the prior beats unregularized LS

    # The MAP reconstruction is smoother (smaller roughness) than LS
    rough_map = np.linalg.norm(L @ m_map)
    rough_ls = np.linalg.norm(L @ m_ls)
    print(f"  roughness MAP / LS     = {rough_map:.2f} / {rough_ls:.2f}")
    assert rough_map < rough_ls

    # Posterior covariance is symmetric positive-definite (valid uncertainty)
    cov = posterior_covariance(G, L, noise_var, 5.0)
    assert np.allclose(cov, cov.T) and np.all(np.linalg.eigvalsh(cov) > 0)

    # More projections -> better reconstruction
    G2 = rng.normal(size=(40, n))
    d2 = project(G2, m_true) + rng.normal(0, 0.02, 40)
    err_more = np.linalg.norm(map_estimate(G2, d2, L, noise_var, 5.0) - m_true)
    print(f"  error 25 vs 40 projections = {err_map:.3f} / {err_more:.3f}")
    assert err_more < err_map
    print("  PASS")
    return {"err_map": float(err_map), "err_ls": float(err_ls)}


if __name__ == "__main__":
    test_all()
