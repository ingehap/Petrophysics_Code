"""
zhang_nmr_core.py
Implementation of ideas from:
Zhang, Song, Luo, Lin & Liu, "Core Analysis Using Nuclear Magnetic Resonance",
Petrophysics, Vol. 65, No. 2 (April 2024), pp. 173-193.

Core procedures implemented:
  - synthetic CPMG echo train generation,
  - inverse-Laplace style multi-exponential T2 inversion (NNLS),
  - bound/free fluid partitioning by T2 cutoff,
  - surface relaxivity to pore-radius conversion,
  - simple D-T2 (diffusion-relaxation) point evaluation.
"""
import numpy as np
from scipy.optimize import nnls


def cpmg_echoes(t, amplitudes, T2s, noise=0.0, rng=None):
    """Forward model: build a CPMG echo train from a discrete T2 distribution."""
    a = np.asarray(amplitudes)[None, :]
    T = np.asarray(T2s)[None, :]
    sig = (a * np.exp(-t[:, None] / T)).sum(axis=1)
    if noise:
        rng = rng or np.random.default_rng(0)
        sig = sig + rng.normal(0, noise, sig.shape)
    return sig


def t2_inversion(t, echoes, T2_grid, alpha=0.1):
    """Tikhonov-regularized NNLS inversion for T2 distribution."""
    K = np.exp(-t[:, None] / T2_grid[None, :])
    A = np.vstack([K, alpha * np.eye(len(T2_grid))])
    b = np.concatenate([echoes, np.zeros(len(T2_grid))])
    x, _ = nnls(A, b)
    return x


def bound_free_fluid(T2_grid, dist, T2_cutoff=33e-3):
    """Partition T2 distribution into bound (< cutoff) and free (>= cutoff) fluid."""
    bound = dist[T2_grid < T2_cutoff].sum()
    free = dist[T2_grid >= T2_cutoff].sum()
    return bound, free


def pore_radius_from_T2(T2, rho2=10e-6, geom=2):
    """1/T2 = rho2 * (S/V)  =>  r = geom * rho2 * T2  for spherical pores (geom=3)."""
    return geom * rho2 * np.asarray(T2)


def diffusion_relaxation(D_grid, T2_grid, D_true=2.3e-9, T2_true=0.1):
    """Make a tiny D-T2 map and locate the peak — like an oil/water identification."""
    sigma = 0.3
    DD, TT = np.meshgrid(np.log10(D_grid), np.log10(T2_grid), indexing="ij")
    M = np.exp(-((DD - np.log10(D_true)) ** 2 + (TT - np.log10(T2_true)) ** 2) / (2 * sigma**2))
    return M


def test_all():
    rng = np.random.default_rng(1)
    T2_true = np.array([0.005, 0.05, 0.5])
    amps_true = np.array([0.3, 0.5, 0.2])
    t = np.linspace(0.001, 3.0, 200)
    s = cpmg_echoes(t, amps_true, T2_true, noise=0.005, rng=rng)
    grid = np.logspace(-3, 1, 60)
    dist = t2_inversion(t, s, grid, alpha=0.05)
    assert dist.sum() > 0
    bound, free = bound_free_fluid(grid, dist, 33e-3)
    assert bound + free > 0
    r = pore_radius_from_T2(grid)
    assert (r > 0).all()
    M = diffusion_relaxation(np.logspace(-11, -8, 20), grid)
    assert M.max() > 0
    print("zhang_nmr_core OK  bound=%.3f free=%.3f" % (bound, free))


if __name__ == "__main__":
    test_all()
