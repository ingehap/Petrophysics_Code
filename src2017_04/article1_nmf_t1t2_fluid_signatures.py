"""
Article 1: Unlocking the Potential of Unconventional Reservoirs Through New
           Generation NMR T1/T2 Logging Measurements Integrated with Advanced
           Wireline Logs
Anand, Ali, Abubakar, Grover, Neto, Pirie, Gonzalez Iglesias (2017)
Reference: Petrophysics Vol. 58, No. 2 (April 2017), pp. 81-96
DOI: none assigned (this issue predates SPWLA DOI assignment)

A blind-source-separation / non-negative matrix factorization (NMF) decomposes a
stack of NMR T1-T2 maps into a small number of poro-fluid signatures and their
per-depth volumetric fractions, with no preset fluid model.  Each T1-T2 map is
vectorized into a column of V; NMF factors V ~ W*H (W = signatures, H = volumes),
both non-negative, and the optimal number of signatures is chosen by a PCA of the
volume matrix.  A spectroscopy-based transform converts carbon weight to a
hydrocarbon volume.

Implements:

  - Non-negative matrix factorization  V ~ W*H (multiplicative updates)
  - Reconstruction error  ||V - W*H||
  - Rank selection from the eigenvalue spectrum of the volume matrix
  - Hydrocarbon volume from carbon weight fraction

Note: this issue's PDF has a text layer but the typeset display equations were
dropped, so the relations are faithful standard-form reconstructions (Lee-Seung
NMF).  Amplitudes are non-negative porosity contributions.
"""

import numpy as np


# ---------------------------------------------- NMF --------------

def nmf(v, rank, iters=1000, seed=0):
    """Factor V ~ W*H with non-negative W, H by multiplicative updates (Eqs. 3-8)."""
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.1, 1.0, (v.shape[0], rank))
    h = rng.uniform(0.1, 1.0, (rank, v.shape[1]))
    eps = 1e-12
    for _ in range(iters):
        h *= (w.T @ v) / (w.T @ w @ h + eps)
        w *= (v @ h.T) / (w @ h @ h.T + eps)
    return w, h


def reconstruction_error(v, w, h):
    """Relative Euclidean reconstruction error  ||V - W*H|| / ||V||  (Eq. 5)."""
    v = np.asarray(v, float)
    return float(np.linalg.norm(v - w @ h) / np.linalg.norm(v))


def select_rank(h, threshold=1e-3):
    """Number of significant signatures from the eigenvalue spectrum of cov(H)."""
    cov = np.cov(np.asarray(h, float))
    eig = np.linalg.eigvalsh(np.atleast_2d(cov))
    return int(np.sum(eig > threshold * eig.max()))


def hc_volume_from_carbon(toc, carbon_in_kerogen, rho_matrix, rho_oil):
    """Hydrocarbon volume from the non-kerogen carbon weight  V_HC = (TOC - Cker)*rho_m/rho_oil."""
    return (toc - carbon_in_kerogen) * rho_matrix / rho_oil


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: NMF of NMR T1-T2 Maps")
    print("=" * 60)

    # Build a synthetic low-rank non-negative map stack V = W_true*H_true
    rng = np.random.default_rng(1)
    r_true = 3
    w_true = rng.uniform(0, 1, (40, r_true))
    h_true = rng.uniform(0, 1, (r_true, 25))
    v = w_true @ h_true

    # NMF recovers V to within a small reconstruction error, with W,H >= 0
    w, h = nmf(v, rank=r_true, iters=1500)
    err = reconstruction_error(v, w, h)
    print(f"  reconstruction error   = {err:.4f}")
    assert err < 0.05 and np.all(w >= 0) and np.all(h >= 0)

    # Rank selection recovers the planted number of signatures
    rank = select_rank(h_true)
    print(f"  selected rank          = {rank}  (true {r_true})")
    assert rank == r_true

    # Hydrocarbon volume from carbon weight fraction
    v_hc = hc_volume_from_carbon(0.06, 0.04, rho_matrix=2.5, rho_oil=0.85)
    assert v_hc > 0
    print("  PASS")
    return {"recon_error": err, "rank": rank}


if __name__ == "__main__":
    test_all()
