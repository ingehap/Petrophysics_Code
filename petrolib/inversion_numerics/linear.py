"""Linear inversion: least squares, Tikhonov, SVD, unmixing, operators.

Deterministic linear-algebra solvers that recur across the corpus -- regularized
(Tikhonov / ridge) least squares, Bayesian MAP, truncated-SVD and pseudo-inverse
solves, non-negative least squares, mineral / spectral unmixing with closure and
non-negativity, convolution / deconvolution, and finite-difference smoothness
operators.

``lam`` (regularization weight) is applied **once** everywhere -- never squared.
scipy is imported lazily inside the functions that need it (NNLS).

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2014_06/article4_nmr_carbonate_permeability -- Article 4: Method for Predicting Permeability of
  Complex Carbonate Reservoirs Using NMR Logging Measurements Willian Trevizan, Paulo Neto,
  Bernardo Coutinho, Vinicius F. Machado, Edmilson H. Rios, Songhua Chen, Wei Shao, Pedro Romero
  (2014). Petrophysics Vol. 55, No. 3 (June 2014), pp. 240-252. DOI: none assigned (this issue
  predates SPWLA DOI assignment).
src2015_04/article5_nmr_short_t2_porosity -- Article 5: New Method to Estimate Porosity More
  Accurately from NMR Data with Short Relaxation Times. Venkataramanan, Gruber, LaVigne, Habashy,
  Iglesias, Cohorn, Anand, Rampurawala, Jain, Heaton, Akkurt, Rylander, Lewis (2015). Petrophysics
  Vol. 56, No. 2 (April 2015), pp. 147-157. DOI: none assigned (this issue predates SPWLA DOI
  assignment).
src2015_08/article4_spectroscopy_inversion -- Article 4: Petrophysical Interpretation of LWD,
  Neutron-Induced Gamma-Ray Spectroscopy Measurements: An Inversion-Based Approach. Ajayi, Torres-
  Verdin, Preeg (2015). Petrophysics Vol. 56, No. 4 (August 2015), pp. 358-378. DOI: none assigned
  (this issue predates SPWLA DOI assignment).
src2017_06/article3_forward_mineral_svd -- Article 3: Forward Mineral Modeling Using Regularized
  Least-Squares Regression With Singular Value Decomposition: Case Study From Qusaiba Shale. Xu,
  McCormick, Herron, Cheshire, Al-Salim, Almarzouq (2017). Petrophysics Vol. 58, No. 3 (June 2017),
  pp. 242-269. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2019_06/article3_wellsite_tomography_bayesian -- Article 3: Accelerated Whole-Core Analysis
  Optimization With Wellsite Tomography Instrumentation and Bayesian Inversion. Mendoza, Roininen,
  Girolami, Heikkinen, Haario (2019). DOI: 10.30632/PJV60N3-2019a2. Petrophysics Vol. 60 No. 3 (Jun
  2019).
src2019_12/article1_sonic_inversion_deconvolution -- Article 1: Inversion of High-Resolution High-
  Quality Sonic Compressional and Shear Logs for Unconventional Reservoirs. Lei, Zeroug, Bose,
  Prioul, Donald (2019). DOI: 10.30632/PJV60N6-2019a1. Petrophysics Vol. 60 No. 6 (Dec 2019).
src2020_02/article4_physics_deeplearning_inversion -- Article 4: A Physics-Driven Deep-Learning
  Network for Solving Nonlinear Inverse Problems. Jin, Shen, Wu, Chen, Huang (2020). DOI:
  10.30632/PJV61N1-2020a3. Petrophysics Vol. 61 No. 1 (Feb 2020).
src2021_02/article8_injectite_em_3d_inversion -- Article 8: Mapping Complex Injectite Bodies With
  Multiwell Electromagnetic 3D Inversion Data. Clegg, Eriksen, Best, Tollefsen, Kowicki, Marchant
  (2021). DOI: 10.30632/PJV62N1-2021a7. Petrophysics Vol. 62 No. 1 (Feb 2021).
src2026_02/nmr_discrete_inversion -- Article 3: Discrete Inversion Method for Nuclear Magnetic
  Resonance Data Processing and Its Applications to Fluid Typing and Quantification. Gao et al.
  (2026), Petrophysics, 67(1), 38-53. DOI: 10.30632/PJV67N1-2026a3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


def difference_operator(n: int, order: int = 1) -> _Float:
    """Finite-difference smoothness operator ``L`` of shape ``(n-order, n)``.

    ``order=1`` is the first difference, ``order=2`` the second difference, and
    ``order=0`` the identity.
    """
    if order == 0:
        return np.eye(n)
    d = np.zeros((n - 1, n))
    idx = np.arange(n - 1)
    d[idx, idx] = -1.0
    d[idx, idx + 1] = 1.0
    if order == 1:
        return d
    return np.asarray(difference_operator(n - 1, order - 1) @ d)


def condition_number(a: ArrayLike) -> float:
    """2-norm condition number ``s_max/s_min`` from the singular values.

    Sources: src2017_06/article3_forward_mineral_svd.
    """
    s = np.linalg.svd(_arr(a), compute_uv=False)
    return float(s[0] / s[-1])


def svd_solve(a: ArrayLike, b: ArrayLike, rank: int | None = None) -> _Float:
    """Least-squares / truncated-SVD solve of ``A x = b`` (rank-limited pseudo-inverse)."""
    a_arr = _arr(a)
    b_arr = _arr(b)
    u, s, vt = np.linalg.svd(a_arr, full_matrices=False)
    s_inv = np.zeros_like(s)
    keep = s.size if rank is None else min(rank, s.size)
    s_inv[:keep] = 1.0 / s[:keep]
    return np.asarray(vt.T @ (s_inv * (u.T @ b_arr)))


def tikhonov_solve(
    a: ArrayLike,
    b: ArrayLike,
    lam: float = 1.0,
    *,
    reg_op: ArrayLike | None = None,
    x_ref: ArrayLike | None = None,
    sigma: ArrayLike | None = None,
    nonneg: bool = False,
) -> _Float:
    """Tikhonov-regularized least squares.

    ``argmin_x ||W(Ax - b)||^2 + lam*||L(x - x_ref)||^2`` with ``L = reg_op``
    (identity by default), data weights ``W = diag(1/sigma)``, and ``lam`` applied
    **once**.  ``nonneg=True`` solves the augmented system with :func:`nnls_solve`.

    Sources: src2014_06/article4_nmr_carbonate_permeability,
    src2015_04/article5_nmr_short_t2_porosity, src2015_08/article4_spectroscopy_inversion,
    src2017_06/article3_forward_mineral_svd,
    src2020_02/article4_physics_deeplearning_inversion,
    src2021_02/article8_injectite_em_3d_inversion, src2026_02/nmr_discrete_inversion.
    """
    a_arr = _arr(a)
    b_arr = _arr(b)
    n = a_arr.shape[1]
    ell = np.eye(n) if reg_op is None else _arr(reg_op)
    if sigma is not None:
        w = 1.0 / _arr(sigma)
        a_arr = a_arr * w[:, None]
        b_arr = b_arr * w
    xref = None if x_ref is None else _arr(x_ref)
    if nonneg:
        ref_rhs = np.zeros(n) if xref is None else ell @ xref
        aug_a = np.vstack([a_arr, np.sqrt(lam) * ell])
        aug_b = np.concatenate([b_arr, np.sqrt(lam) * ref_rhs])
        return nnls_solve(aug_a, aug_b)
    ltl = ell.T @ ell
    lhs = a_arr.T @ a_arr + lam * ltl
    rhs = a_arr.T @ b_arr
    if xref is not None:
        rhs = rhs + lam * (ltl @ xref)
    return np.asarray(np.linalg.solve(lhs, rhs))


def map_estimate(
    g: ArrayLike,
    d: ArrayLike,
    noise_var: float,
    prior_strength: float,
    ell: ArrayLike | None = None,
    m_prior: ArrayLike | None = None,
) -> tuple[_Float, _Float]:
    """Bayesian MAP estimate and posterior covariance for a linear-Gaussian model.

    ``A = G^T G/noise_var + prior_strength*L^T L``; the MAP model is
    ``A^{-1}(G^T d/noise_var + prior_strength*L^T L m_prior)`` with posterior
    covariance ``A^{-1}``.  Returns ``(x, posterior_cov)``.
    """
    g_arr = _arr(g)
    d_arr = _arr(d)
    n = g_arr.shape[1]
    ell_m = np.eye(n) if ell is None else _arr(ell)
    ltl = ell_m.T @ ell_m
    a = g_arr.T @ g_arr / noise_var + prior_strength * ltl
    mp = np.zeros(n) if m_prior is None else _arr(m_prior)
    b = g_arr.T @ d_arr / noise_var + prior_strength * (ltl @ mp)
    cov = np.linalg.inv(a)
    return np.asarray(cov @ b), np.asarray(cov)


def nnls_solve(a: ArrayLike, b: ArrayLike, reg: float = 0.0) -> _Float:
    """Non-negative least squares (scipy), optionally ridge-augmented by ``reg``."""
    a_arr = _arr(a)
    b_arr = _arr(b)
    if reg > 0.0:
        n = a_arr.shape[1]
        a_arr = np.vstack([a_arr, np.sqrt(reg) * np.eye(n)])
        b_arr = np.concatenate([b_arr, np.zeros(n)])
    try:
        from scipy.optimize import nnls
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError("nnls_solve requires scipy (scipy.optimize.nnls)") from exc
    x, _ = nnls(a_arr, b_arr)
    return np.asarray(x, np.float64)


def ista_l1(
    a: ArrayLike, b: ArrayLike, eta: float = 0.1, nonneg: bool = True, max_iter: int = 500
) -> _Float:
    """Sparse L1 inversion by ISTA (iterative soft-thresholding).

    Minimizes ``0.5*||Ax-b||^2 + eta*||x||_1`` with an optional non-negativity
    projection.
    """
    a_arr = _arr(a)
    b_arr = _arr(b)
    x = np.zeros(a_arr.shape[1])
    step = 1.0 / np.linalg.norm(a_arr, 2) ** 2
    thr = eta * step
    for _ in range(max_iter):
        x = x - step * (a_arr.T @ (a_arr @ x - b_arr))
        x = np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)
        if nonneg:
            x = np.maximum(x, 0.0)
    return np.asarray(x)


def project_simplex(v: ArrayLike, total: float = 1.0) -> _Float:
    """Euclidean projection of ``v`` onto the simplex ``{x>=0, sum(x)=total}``."""
    v_arr = _arr(v)
    u = np.sort(v_arr)[::-1]
    css = np.cumsum(u) - total
    ind = np.arange(1, u.size + 1)
    rho = np.nonzero(u - css / ind > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.asarray(np.maximum(v_arr - theta, 0.0))


def unmix(
    measured: ArrayLike,
    response_matrix: ArrayLike,
    *,
    sigma: ArrayLike | None = None,
    closure: str | None = None,
    nonneg: str | None = "nnls",
    normalize: bool = False,
) -> _Float:
    """Mineral / spectral unmixing ``measured ~ response_matrix @ x``.

    ``closure`` in ``{None, 'row', 'simplex'}`` enforces sum-to-one (``'row'``
    appends a heavily weighted unit-sum equation; ``'simplex'`` projects the
    solution).  ``nonneg`` in ``{None, 'clip', 'nnls', 'simplex'}`` handles
    non-negativity.  ``sigma`` applies ``1/sigma`` data weights; ``normalize``
    rescales the result to sum to one.
    """
    a = _arr(response_matrix)
    b = _arr(measured)
    if sigma is not None:
        w = 1.0 / _arr(sigma)
        a = a * w[:, None]
        b = b * w
    if closure == "row":
        big = 1e3 * (np.abs(a).max() + 1.0)
        a = np.vstack([a, big * np.ones(a.shape[1])])
        b = np.concatenate([b, [big]])
    if nonneg == "nnls":
        x = nnls_solve(a, b)
    elif nonneg == "clip":
        x = np.clip(np.linalg.lstsq(a, b, rcond=None)[0], 0.0, None)
    elif nonneg == "simplex":
        x = project_simplex(np.linalg.lstsq(a, b, rcond=None)[0], total=1.0)
    else:
        x = np.asarray(np.linalg.lstsq(a, b, rcond=None)[0])
    if closure == "simplex":
        x = project_simplex(x, total=1.0)
    if normalize:
        s = x.sum()
        if s != 0.0:
            x = x / s
    return np.asarray(x)


def convolution_matrix(kernel: ArrayLike, n: int) -> _Float:
    """Row-normalized ``(n, n)`` convolution (blurring) matrix for a centered kernel."""
    k = _arr(kernel)
    half = k.size // 2
    g = np.zeros((n, n))
    for i in range(n):
        for j in range(k.size):
            col = i + j - half
            if 0 <= col < n:
                g[i, col] = k[j]
        row_sum = g[i].sum()
        if row_sum != 0.0:
            g[i] /= row_sum
    return g


def deconvolve(d: ArrayLike, g: ArrayLike, rank: int | None = None) -> _Float:
    """Deconvolve ``d = G x`` for ``x`` via (rank-limited) pseudo-inverse.

    Sources: src2019_06/article3_wellsite_tomography_bayesian,
    src2019_12/article1_sonic_inversion_deconvolution.
    """
    g_arr = _arr(g)
    d_arr = _arr(d)
    if rank is None:
        return np.asarray(np.linalg.pinv(g_arr) @ d_arr)
    return svd_solve(g_arr, d_arr, rank=rank)
