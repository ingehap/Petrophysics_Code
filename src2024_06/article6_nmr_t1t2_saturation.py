"""
article6_nmr_t1t2_saturation.py
===============================

Implementation of the 2D T1-T2 NMR saturation determination scheme from:

    Althaus, S., Chen, J., Sun, Q., and Broyles, J. D. (2024).
    "Determine Oil and Water Saturations in Preserved Source Rocks
    From 2D T1-T2 NMR."  Petrophysics 65(3), 388-396.
    DOI: 10.30632/PJV65N3-2024a6

The workflow consists of:

1. Forward model:  an inversion-recovery CPMG experiment produces the
   signal

       S(t1, t2) = sum_i  M_i * (1 - 2*exp(-t1/T1_i)) * exp(-t2/T2_i)

   for a set of fluid components i with amplitudes M_i, longitudinal
   relaxation times T1_i and transverse relaxation times T2_i.

2. 2D inversion:  recover the amplitude density f(T1, T2) on a
   logarithmically spaced grid by Tikhonov-regularised non-negative
   least squares.  The paper uses the MUPen2D inversion package; here
   we use a compact L2-regularised solver plus clipping that reproduces
   the essential behaviour on synthetic data.

3. Saturation computation:  integrate f(T1, T2) over user-defined oil
   and water regions in the T1-T2 plane, convert the component volumes
   to fluid-filled porosities (Eq. 2 of the paper) and then to
   saturations via Eq. 3:

       phi_o = V_o / V_bulk           S_o = phi_o / phi_total
       phi_w = V_w / V_bulk           S_w = phi_w / phi_total .
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Forward model --------------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class NMRComponent:
    amplitude: float       # volume (ml) or intensity
    T1_ms: float
    T2_ms: float


def forward_signal(components: list[NMRComponent],
                   t1_grid: np.ndarray, t2_grid: np.ndarray) -> np.ndarray:
    """Inversion-recovery CPMG forward model, Eq. 1 of the paper.

    Returns a 2D array of shape (len(t1_grid), len(t2_grid)).
    """
    t1 = np.asarray(t1_grid, dtype=float).reshape(-1, 1)
    t2 = np.asarray(t2_grid, dtype=float).reshape(1, -1)
    S = np.zeros((t1.size, t2.size))
    for c in components:
        S += (c.amplitude *
              (1.0 - 2.0 * np.exp(-t1 / c.T1_ms)) *
              np.exp(-t2 / c.T2_ms))
    return S


# ---------------------------------------------------------------------------
# 2D inversion (Tikhonov) ---------------------------------------------------
# ---------------------------------------------------------------------------

def _log_grid(vmin: float, vmax: float, n: int) -> np.ndarray:
    return np.logspace(np.log10(vmin), np.log10(vmax), n)


def invert_t1t2(signal: np.ndarray, t1_obs_ms: np.ndarray,
                t2_obs_ms: np.ndarray, n_grid: int = 24,
                alpha: float = 1e-2,
                t1_range: tuple[float, float] = (0.1, 1e4),
                t2_range: tuple[float, float] = (0.1, 1e4)
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimal Tikhonov 2D T1-T2 inversion.

    The forward operator maps a grid amplitude map f(T1, T2) back onto
    the observation matrix.  Unknowns are stacked into a 1D vector and
    the linear system is solved with normal equations plus L2 penalty
    `alpha`; the solution is clipped at zero to enforce non-negativity.

    Returns (T1_grid, T2_grid, f_grid)  where f_grid has shape
    (n_grid, n_grid).
    """
    t1_obs = np.asarray(t1_obs_ms, dtype=float)
    t2_obs = np.asarray(t2_obs_ms, dtype=float)

    t1_grid = _log_grid(*t1_range, n_grid)
    t2_grid = _log_grid(*t2_range, n_grid)

    # Kernels  K1[i, p] = 1 - 2 * exp(-t1_obs[i]/T1_grid[p])
    #          K2[j, q] = exp(-t2_obs[j] / T2_grid[q])
    K1 = 1.0 - 2.0 * np.exp(-t1_obs[:, None] / t1_grid[None, :])
    K2 = np.exp(-t2_obs[:, None] / t2_grid[None, :])

    # Vectorised problem:   vec(S) = (K2 kron K1) vec(F)
    # We build the full operator (feasible for n_grid ~ 20).
    A = np.kron(K2, K1)
    b = signal.reshape(-1)
    n = A.shape[1]

    # Normal equations with Tikhonov regularisation
    AtA = A.T @ A + alpha * np.eye(n)
    Atb = A.T @ b
    f_vec = np.linalg.solve(AtA, Atb)
    f_vec = np.clip(f_vec, 0.0, None)

    f_grid = f_vec.reshape(t1_grid.size, t2_grid.size)
    return t1_grid, t2_grid, f_grid


# ---------------------------------------------------------------------------
# Saturation computation ----------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class FluidRegion:
    """Box in log10 T1-T2 space defining where oil or water are expected."""
    t1_min_ms: float
    t1_max_ms: float
    t2_min_ms: float
    t2_max_ms: float

    def mask(self, t1_grid: np.ndarray, t2_grid: np.ndarray) -> np.ndarray:
        m1 = (t1_grid >= self.t1_min_ms) & (t1_grid <= self.t1_max_ms)
        m2 = (t2_grid >= self.t2_min_ms) & (t2_grid <= self.t2_max_ms)
        return np.outer(m1, m2)


def integrate_region(f_grid: np.ndarray, t1_grid: np.ndarray,
                     t2_grid: np.ndarray, region: FluidRegion) -> float:
    """Sum up the amplitude map inside a T1-T2 box."""
    return float(f_grid[region.mask(t1_grid, t2_grid)].sum())


def saturations(v_oil: float, v_water: float, v_bulk: float,
                phi_total: float) -> tuple[float, float, float, float]:
    """Eq. 2 and 3 of Althaus et al.

    Returns  (phi_o, phi_w, S_o, S_w).
    """
    phi_o = v_oil / v_bulk
    phi_w = v_water / v_bulk
    s_o = phi_o / phi_total if phi_total > 0 else 0.0
    s_w = phi_w / phi_total if phi_total > 0 else 0.0
    return phi_o, phi_w, s_o, s_w


# ---------------------------------------------------------------------------
# Test harness ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_all(verbose: bool = True) -> None:
    # Two fluid components: a "water-like" fast-relaxing peak and an
    # "oil-like" slow-relaxing peak.
    water = NMRComponent(amplitude=0.5, T1_ms=2.0, T2_ms=1.0)
    oil = NMRComponent(amplitude=1.0, T1_ms=200.0, T2_ms=50.0)
    comps = [water, oil]

    # Observation grids: logarithmically spaced, like the paper uses.
    t1_obs = np.logspace(np.log10(0.5), np.log10(3000), 25)
    t2_obs = np.logspace(np.log10(0.1), np.log10(1500), 40)

    signal = forward_signal(comps, t1_obs, t2_obs)
    assert signal.shape == (25, 40)
    assert np.isfinite(signal).all()

    # Run the 2D inversion.
    t1_g, t2_g, f_grid = invert_t1t2(signal, t1_obs, t2_obs,
                                     n_grid=20, alpha=1e-2)
    assert f_grid.shape == (20, 20)
    assert f_grid.min() >= 0

    # Define T1-T2 boxes and sum the recovered amplitude inside each.
    water_box = FluidRegion(t1_min_ms=0.5, t1_max_ms=10,
                            t2_min_ms=0.3, t2_max_ms=5)
    oil_box = FluidRegion(t1_min_ms=30, t1_max_ms=1000,
                          t2_min_ms=5, t2_max_ms=400)
    v_w = integrate_region(f_grid, t1_g, t2_g, water_box)
    v_o = integrate_region(f_grid, t1_g, t2_g, oil_box)

    # The oil/water ratio should be ~2 (1.0 / 0.5) -- this is a low-res
    # regularised inversion so we accept a broad tolerance.
    ratio = v_o / max(v_w, 1e-12)
    assert 1.0 < ratio < 4.0, f"Unexpected oil/water ratio: {ratio:.2f}"

    # Saturation calculation.  Note v_o and v_w are NMR amplitude units
    # and not calibrated against v_bulk in this synthetic test, so we
    # only check that the computation runs and produces non-negative
    # numbers (a real experiment calibrates the instrument first).
    v_bulk = 10.0   # cc
    phi_total = 0.1
    phi_o, phi_w, s_o, s_w = saturations(v_oil=v_o, v_water=v_w,
                                          v_bulk=v_bulk, phi_total=phi_total)
    assert phi_o >= 0 and phi_w >= 0
    assert s_o >= 0 and s_w >= 0

    # Sanity: the total recovered amplitude is positive.  A regularised
    # 2D Fredholm-I inversion on a coarse log grid does not exactly
    # preserve the integrated amplitude, so we only assert non-zero.
    total = f_grid.sum()
    assert total > 0, "Inversion returned a zero grid"

    if verbose:
        print("Article 6 (2D T1-T2 NMR saturation): all tests passed.")
        print(f"  recovered water amp  = {v_w:.3f}  (true 0.500)")
        print(f"  recovered oil   amp  = {v_o:.3f}  (true 1.000)")
        print(f"  oil/water ratio      = {ratio:.3f}")
        print(f"  phi_o, phi_w         = {phi_o:.4f}, {phi_w:.4f}")
        print(f"  S_o,  S_w            = {s_o:.3f}, {s_w:.3f}")


if __name__ == "__main__":
    test_all()
