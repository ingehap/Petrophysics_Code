"""
Article 4: A Physics-Driven Deep-Learning Network for Solving Nonlinear Inverse
           Problems
Jin, Shen, Wu, Chen, Huang (2020)
DOI: 10.30632/PJV61N1-2020a3

Geophysical inverse problems (e.g. recovering a resistivity/conductivity model
from electromagnetic data) are ill-posed and nonlinear.  A physics-driven
approach embeds the forward operator in the inversion and adds regularization,
minimizing a data-misfit-plus-physics objective so the recovered model both
fits the measurements and stays physically smooth/stable.

Implements:

  - Forward operator  d = G @ m  (+ noise)
  - Data misfit  ||G m - d||^2 and Tikhonov-regularized objective
  - Regularized (ridge) inversion  m = (G^T G + lambda I)^-1 G^T d
  - Iterative gradient-descent inversion (the network-training analogue)

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the regularized nonlinear-inversion framework the paper's
title describes (a deep network is replaced by an equivalent
physics-constrained optimizer).
"""

import numpy as np


# ---------------------------------------------- forward / misfit --------

def forward(G, m):
    """Linear forward operator  d = G @ m."""
    return np.asarray(G, float) @ np.asarray(m, float)


def data_misfit(G, m, d):
    """Data misfit  ||G m - d||^2."""
    r = forward(G, m) - np.asarray(d, float)
    return float(r @ r)


def objective(G, m, d, lam):
    """Tikhonov objective  ||G m - d||^2 + lambda*||m||^2."""
    m = np.asarray(m, float)
    return data_misfit(G, m, d) + lam * float(m @ m)


# ---------------------------------------------- inversions --------------

def ridge_inverse(G, d, lam):
    """Closed-form Tikhonov inversion  m = (G^T G + lambda I)^-1 G^T d."""
    G = np.asarray(G, float)
    n = G.shape[1]
    return np.linalg.solve(G.T @ G + lam * np.eye(n), G.T @ np.asarray(d, float))


def gd_inverse(G, d, lam, iters=4000, lr=None, seed=0):
    """Gradient-descent inversion of the Tikhonov objective (training analogue)."""
    G = np.asarray(G, float); d = np.asarray(d, float)
    n = G.shape[1]
    m = np.zeros(n)
    GtG = G.T @ G; Gtd = G.T @ d
    if lr is None:
        # gradient has a factor of 2, so the max stable step is 1/(L+lam);
        # use half of it to stay safely in the convergent regime
        lr = 0.5 / (np.linalg.norm(GtG, 2) + lam)
    for _ in range(iters):
        grad = 2.0 * (GtG @ m - Gtd) + 2.0 * lam * m
        m = m - lr * grad
    return m


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Physics-Driven Deep-Learning Inversion")
    print("=" * 60)

    rng = np.random.default_rng(0)
    G = rng.normal(size=(40, 12))
    m_true = rng.normal(size=12)
    d_clean = forward(G, m_true)

    # Well-conditioned, noise-free: ridge with tiny lambda recovers the model
    m_hat = ridge_inverse(G, d_clean, lam=1e-8)
    err = np.linalg.norm(m_hat - m_true) / np.linalg.norm(m_true)
    print(f"  noise-free recovery err = {err:.2e}")
    assert err < 1e-4

    # Closed-form and gradient-descent solutions agree (the training analogue)
    m_gd = gd_inverse(G, d_clean, lam=1e-3, iters=20000)
    m_cf = ridge_inverse(G, d_clean, lam=1e-3)
    assert np.linalg.norm(m_gd - m_cf) / np.linalg.norm(m_cf) < 1e-2

    # Ill-posed + noisy: regularization stabilizes the solution
    d_noisy = d_clean + rng.normal(0, 0.5, d_clean.size)
    m_unreg = ridge_inverse(G, d_noisy, lam=1e-8)
    m_reg = ridge_inverse(G, d_noisy, lam=1.0)
    err_unreg = np.linalg.norm(m_unreg - m_true)
    err_reg = np.linalg.norm(m_reg - m_true)
    print(f"  noisy err unreg / reg  = {err_unreg:.3f} / {err_reg:.3f}")
    assert err_reg < err_unreg                 # Tikhonov reduces error under noise

    # Regularization shrinks the model norm and raises the data misfit
    assert (m_reg @ m_reg) < (m_unreg @ m_unreg)
    assert data_misfit(G, m_reg, d_noisy) > data_misfit(G, m_unreg, d_noisy)
    print("  PASS")
    return {"recovery_err": err, "err_unreg": float(err_unreg),
            "err_reg": float(err_reg)}


if __name__ == "__main__":
    test_all()
