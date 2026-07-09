"""Tests for petrolib.inversion_numerics: linear / costs / nonlinear / stochastic
/ optimize / fitting / pde solvers -- recovery on synthetic problems, golden
values, and error paths.  scipy-dependent paths are importorskip-guarded."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import inversion_numerics as inv  # noqa: E402

# --- linear -------------------------------------------------------------------


def _lin_problem(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(20, 4))
    x_true = np.array([1.0, -2.0, 3.0, 0.5])
    b = a @ x_true + 0.01 * rng.normal(size=20)
    return a, b, x_true


def test_tikhonov_svd_map_recover() -> None:
    a, b, x_true = _lin_problem()
    np.testing.assert_allclose(inv.linear.tikhonov_solve(a, b, lam=1e-6), x_true, atol=0.05)
    np.testing.assert_allclose(inv.linear.svd_solve(a, b), x_true, atol=0.05)
    x_map, cov = inv.linear.map_estimate(a, b, 1e-4, 1e-6)
    np.testing.assert_allclose(x_map, x_true, atol=0.05)
    assert cov.shape == (4, 4)
    # rank truncation drops the smallest singular direction
    assert inv.linear.condition_number(a) > 1.0


def test_difference_operator_and_simplex() -> None:
    assert inv.linear.difference_operator(5, 1).shape == (4, 5)
    assert inv.linear.difference_operator(5, 2).shape == (3, 5)
    np.testing.assert_array_equal(inv.linear.difference_operator(4, 0), np.eye(4))
    sp = inv.linear.project_simplex([0.5, 0.3, 0.4])
    np.testing.assert_allclose(sp.sum(), 1.0)
    assert np.all(sp >= 0.0)


def test_convolution_deconvolve_roundtrip_and_unmix() -> None:
    g = inv.linear.convolution_matrix([0.25, 0.5, 0.25], 12)
    np.testing.assert_allclose(g.sum(axis=1), 1.0)  # row-normalized
    sig = np.sin(np.linspace(0.0, 3.0, 12))
    np.testing.assert_allclose(inv.linear.deconvolve(g @ sig, g), sig, atol=1e-6)
    rng = np.random.default_rng(1)
    r = np.abs(rng.normal(size=(6, 3)))
    v_true = np.array([0.5, 0.3, 0.2])
    vm = inv.linear.unmix(r @ v_true, r, nonneg="clip", normalize=True)
    np.testing.assert_allclose(vm.sum(), 1.0)
    assert np.all(vm >= 0.0)


def test_ista_l1_sparse() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=(40, 12))
    x_true = np.zeros(12)
    x_true[[2, 7]] = [3.0, 1.5]
    x = inv.linear.ista_l1(a, a @ x_true, eta=0.05, max_iter=800)
    # the two active coefficients dominate
    assert x[2] > 1.0 and x[7] > 0.5
    assert np.all(x >= 0.0)


def test_tikhonov_matrix_rhs() -> None:
    # a multi-column right-hand side solves each column independently and
    # matches the explicit ridge normal equations
    rng = np.random.default_rng(9)
    a = rng.normal(size=(15, 4))
    x_true = rng.normal(size=(4, 3))
    b = a @ x_true
    lam = 0.5
    x = inv.linear.tikhonov_solve(a, b, lam)
    expected = np.linalg.solve(a.T @ a + lam * np.eye(4), a.T @ b)
    assert x.shape == (4, 3)
    np.testing.assert_array_equal(x, expected)
    # each column matches the single-vector solve of that column (LAPACK batches
    # multi-RHS solves differently, so compare within tolerance, not bit-for-bit)
    for j in range(3):
        np.testing.assert_allclose(x[:, j], inv.linear.tikhonov_solve(a, b[:, j], lam))


# --- costs --------------------------------------------------------------------


def test_misfit_kinds_and_schedules() -> None:
    np.testing.assert_allclose(inv.costs.misfit([1, 2, 3], [1, 2, 3]), 0.0)
    assert inv.costs.misfit([1.1, 2, 3], [1, 2, 3], kind="rms") > 0
    np.testing.assert_allclose(
        inv.costs.misfit([2, 4], [1, 2], kind="rel_data"), (1.0) ** 2 + (2.0 / 2.0) ** 2
    )
    with pytest.raises(ValueError, match="kind"):
        inv.costs.misfit([1], [1], kind="bogus")
    np.testing.assert_allclose(inv.costs.reg_lambda_multiplicative(4.0, 2.0, 1.0), 8.0)
    np.testing.assert_allclose(inv.costs.reg_lambda_multiplicative(4.0, 2.0, 1.0, lam_max=5.0), 5.0)


def test_reg_lambda_brd_hits_target() -> None:
    a, b, _ = _lin_problem(3)
    lam = inv.costs.reg_lambda_brd(a, b, chi2_target=1.0, bracket=(1e-8, 1e3))
    x = inv.linear.tikhonov_solve(a, b, lam)
    chi2 = float(np.sum((a @ x - b) ** 2))
    assert chi2 == pytest.approx(1.0, rel=0.2)


# --- fitting ------------------------------------------------------------------


def test_fit_line_and_cosine() -> None:
    lf = inv.fitting.fit_line([1, 2, 3, 4], [2.1, 4.0, 6.1, 8.0])
    assert 1.9 < lf.slope < 2.1 and lf.r2 > 0.99
    lf2 = inv.fitting.fit_line([1, 10, 100], [10, 100, 1000], xform="log10", yform="log10")
    np.testing.assert_allclose(lf2.slope, 1.0, atol=0.01)
    with pytest.raises(ValueError, match="transform"):
        inv.fitting.fit_line([1, 2], [1, 2], xform="bogus")
    az = np.linspace(0.0, 330.0, 12)
    y = 5.0 + 2.0 * np.cos(np.radians(az - 30.0))
    mean, amp, phase = inv.fitting.fit_cosine(az, y)
    np.testing.assert_allclose([mean, amp, phase], [5.0, 2.0, 30.0], atol=1e-6)


def test_fit_powerlaw_linearized_and_exp_three_point() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    a, b = inv.fitting.fit_powerlaw_decay(x, 5.0 * x**-0.5, exponent=0.5)
    np.testing.assert_allclose([a, b], [5.0, 0.5])
    t = np.linspace(0.0, 10.0, 11)
    y = 3.0 + (1.0 - 3.0) * np.exp(-t / 2.0)
    asy, tau = inv.fitting.fit_exponential_approach(t, y, three_point=True)
    np.testing.assert_allclose([asy, tau], [3.0, 2.0], atol=0.1)


# --- pde ----------------------------------------------------------------------


def test_pde_cfl_conductivity_diffusion() -> None:
    np.testing.assert_allclose(inv.pde.cfl_number(1.0, 0.1, 1.0), 0.1)
    # a uniform conductivity map has that same effective conductivity
    np.testing.assert_allclose(
        inv.pde.effective_conductivity_2d(np.full((8, 8), 2.0), n_iter=3000), 2.0, atol=1e-3
    )
    u = np.zeros(11)
    u[5] = 1.0
    u1 = inv.pde.diffusion_step_1d(u, 1.0, 0.1, 1.0)
    assert u1[5] < 1.0 and u1[4] > 0.0 and u1[6] > 0.0
    with pytest.raises(ValueError, match="bc"):
        inv.pde.diffusion_step_1d(u, 1.0, 0.1, 1.0, bc="bogus")


# --- nonlinear ----------------------------------------------------------------


def _exp_forward(m: np.ndarray) -> np.ndarray:
    return np.asarray(m[0] * np.exp(-m[1] * np.linspace(0.0, 2.0, 15)))


def test_lm_grid_jacobian_recover() -> None:
    rng = np.random.default_rng(4)
    data = _exp_forward(np.array([3.0, 1.5])) + 0.001 * rng.normal(size=15)
    res = inv.nonlinear.levenberg_marquardt(_exp_forward, data, [1.0, 1.0], max_iter=60)
    np.testing.assert_allclose(res.model, [3.0, 1.5], atol=0.1)
    assert res.n_iter >= 1
    assert inv.nonlinear.fd_jacobian(_exp_forward, [3.0, 1.5]).shape == (15, 2)
    gm, _ = inv.nonlinear.grid_search(
        _exp_forward, data, [np.linspace(2.0, 4.0, 9), np.linspace(1.0, 2.0, 9)], misfit="l2"
    )
    np.testing.assert_allclose(gm, [3.0, 1.5], atol=0.3)


def test_multistart_and_feasible_set() -> None:
    rng = np.random.default_rng(5)
    data = _exp_forward(np.array([3.0, 1.5])) + 0.001 * rng.normal(size=15)

    def solver(m0: np.ndarray) -> inv.InvResult:
        return inv.nonlinear.levenberg_marquardt(_exp_forward, data, m0, max_iter=40)

    m = inv.nonlinear.multistart(solver, [(1.0, 5.0), (0.5, 3.0)], n_starts=8, seed=0)
    np.testing.assert_allclose(m, [3.0, 1.5], atol=0.2)
    fs = inv.nonlinear.feasible_set_sampling(
        _exp_forward, data, [3.0, 1.5], [(1.0, 5.0), (0.5, 3.0)], 0.05, n_samples=500
    )
    assert fs["P5"].shape == (2,) and np.all(fs["P5"] <= fs["P95"])


# --- stochastic ---------------------------------------------------------------


def test_metropolis_recovers_posterior() -> None:
    rng = np.random.default_rng(6)
    truth = np.array([3.0, 1.5])
    data = _exp_forward(truth) + 0.01 * rng.normal(size=15)

    def log_post(m: np.ndarray) -> float:
        return inv.stochastic.gaussian_loglik(
            data, _exp_forward(m), 0.01
        ) + inv.stochastic.uniform_logprior(m, [(0.0, 10.0), (0.0, 5.0)])

    chain = inv.stochastic.metropolis(
        log_post, truth, [0.05, 0.05], n_samples=3000, burn_in=1000, seed=1
    )
    np.testing.assert_allclose(chain.samples.mean(0), truth, atol=0.3)
    assert 0.0 < chain.acceptance <= 1.0
    assert inv.stochastic.uniform_logprior([20.0, 1.0], [(0.0, 10.0), (0.0, 5.0)]) == -np.inf


def test_enkf_update_linear_gaussian() -> None:
    rng = np.random.default_rng(7)
    ens = rng.normal(0.0, 2.0, size=(200, 2))
    h = np.array([[1.0, 0.0]])
    obs = np.array([3.0])
    updated = inv.stochastic.enkf_update(ens, obs, np.array([[0.01]]), h, seed=0)
    # the observed component is pulled toward the observation
    assert abs(updated[:, 0].mean() - 3.0) < 0.5


# --- optimize -----------------------------------------------------------------


def test_pso_and_gradient_descent() -> None:
    def rosen(x: np.ndarray) -> float:
        return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)

    x_best, f_best, history = inv.optimize.pso(
        rosen, [(-2.0, 2.0), (-1.0, 3.0)], n_particles=40, n_iter=150, seed=2
    )
    assert f_best < 0.1 and history[-1] <= history[0]
    x = inv.optimize.gradient_descent(
        lambda z: float((z[0] - 1.0) ** 2 + (z[1] + 2.0) ** 2), [0.0, 0.0], lr=0.1, max_iter=500
    )
    np.testing.assert_allclose(x, [1.0, -2.0], atol=1e-2)


# --- scipy-dependent paths ----------------------------------------------------


def test_nnls_paths() -> None:
    pytest.importorskip("scipy")
    rng = np.random.default_rng(8)
    a = np.abs(rng.normal(size=(10, 3)))
    x_true = np.array([2.0, 0.0, 1.5])
    x = inv.linear.nnls_solve(a, a @ x_true)
    np.testing.assert_allclose(x, x_true, atol=1e-6)
    assert np.all(x >= 0.0)


def test_curve_fit_paths() -> None:
    pytest.importorskip("scipy")
    x = np.linspace(1.0, 10.0, 20)
    a, b = inv.fitting.fit_powerlaw_decay(x, 4.0 * x**-1.3)
    np.testing.assert_allclose([a, b], [4.0, 1.3], atol=1e-3)
    t = np.linspace(0.0, 8.0, 30)
    asy, tau = inv.fitting.fit_exponential_approach(t, 2.0 + (5.0 - 2.0) * np.exp(-t / 1.5))
    np.testing.assert_allclose([asy, tau], [2.0, 1.5], atol=1e-2)
