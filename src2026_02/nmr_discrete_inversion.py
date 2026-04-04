"""
Article 3: Discrete Inversion Method for Nuclear Magnetic Resonance Data
Processing and Its Applications to Fluid Typing and Quantification.

Authors: Gao, Kwak, Hursan, and Althaus (2026)
DOI: 10.30632/PJV67N1-2026a3

Implements NMR data processing methods including continuous (ILT) and discrete
inversion of T2 relaxation data, L1 and L2 regularization, and fluid typing.

Key ideas implemented:
    - CPMG echo train simulation with multi-exponential decay
    - Fredholm integral equation of the first kind (Eq. 1)
    - Tikhonov (L2) regularized inversion (Eq. 2)
    - L1 regularized (sparse) inversion for improved spectral resolution
    - Discrete component optimization for fluid typing
    - Butler-Reeds-Dawson regularization factor selection

References
----------
Gao et al. (2026), Petrophysics, 67(1), 38-53.
Butler et al. (1981); Song et al. (2005); Prange and Song (2009, 2010).
Ukkelberg et al. (2010); Sørland et al. (2022, 2024).
"""

import numpy as np
from typing import Optional


def generate_t2_basis(t2_min: float = 0.1,
                      t2_max: float = 10000.0,
                      n_bins: int = 100) -> np.ndarray:
    """Generate logarithmically spaced T2 relaxation time basis.

    Parameters
    ----------
    t2_min : float
        Minimum T2 value (ms).
    t2_max : float
        Maximum T2 value (ms).
    n_bins : int
        Number of T2 bins.

    Returns
    -------
    np.ndarray
        T2 basis values (ms), shape (n_bins,).
    """
    return np.logspace(np.log10(t2_min), np.log10(t2_max), n_bins)


def build_kernel_matrix(echo_times: np.ndarray,
                        t2_basis: np.ndarray) -> np.ndarray:
    """Build the kernel matrix A for the Fredholm integral equation.

    M(t) = integral f(T2) * exp(-t/T2) dT2 ≈ A · f

    where A_ij = exp(-t_i / T2_j)

    Parameters
    ----------
    echo_times : np.ndarray
        Echo times from CPMG sequence (ms), shape (n_echoes,).
    t2_basis : np.ndarray
        T2 relaxation time basis (ms), shape (n_bins,).

    Returns
    -------
    np.ndarray
        Kernel matrix, shape (n_echoes, n_bins).
    """
    return np.exp(-echo_times[:, None] / t2_basis[None, :])


def simulate_cpmg(t2_components: list,
                  amplitudes: list,
                  echo_spacing: float = 0.6,
                  n_echoes: int = 500,
                  noise_std: float = 0.0) -> tuple:
    """Simulate a CPMG echo train from multi-exponential decay.

    Parameters
    ----------
    t2_components : list[float]
        T2 relaxation times (ms) of the components.
    amplitudes : list[float]
        Corresponding amplitudes (proportional to fluid volume).
    echo_spacing : float
        Inter-echo time TE (ms).
    n_echoes : int
        Number of echoes.
    noise_std : float
        Standard deviation of Gaussian noise to add.

    Returns
    -------
    echo_times : np.ndarray
        Echo times (ms).
    signal : np.ndarray
        CPMG echo amplitudes.
    """
    echo_times = np.arange(1, n_echoes + 1) * echo_spacing
    signal = np.zeros(n_echoes)
    for t2, amp in zip(t2_components, amplitudes):
        signal += amp * np.exp(-echo_times / t2)
    if noise_std > 0:
        signal += np.random.randn(n_echoes) * noise_std
    return echo_times, signal


def inversion_tikhonov(signal: np.ndarray,
                       kernel: np.ndarray,
                       alpha: float = 1.0,
                       non_negative: bool = True) -> np.ndarray:
    """Tikhonov (L2) regularized inversion of NMR relaxation data (Eq. 2).

    Minimize: ||A·f - M||² + α·||f||²

    subject to f >= 0 (non-negativity constraint).

    Parameters
    ----------
    signal : np.ndarray
        Measured echo amplitudes M(t), shape (n_echoes,).
    kernel : np.ndarray
        Kernel matrix A, shape (n_echoes, n_bins).
    alpha : float
        Regularization factor. Higher values produce smoother distributions.
    non_negative : bool
        If True, enforce non-negativity via iterative projection.

    Returns
    -------
    np.ndarray
        T2 distribution f(T2), shape (n_bins,).
    """
    n_bins = kernel.shape[1]
    AtA = kernel.T @ kernel
    Atb = kernel.T @ signal
    I = np.eye(n_bins)

    # Closed-form Tikhonov solution
    f = np.linalg.solve(AtA + alpha * I, Atb)

    if non_negative:
        # Iterative projected gradient for non-negativity
        for _ in range(50):
            f = np.maximum(f, 0)
            residual = kernel @ f - signal
            grad = kernel.T @ residual + alpha * f
            step_size = 1.0 / (np.linalg.norm(AtA, ord=2) + alpha + 1e-10)
            f = f - step_size * grad
            f = np.maximum(f, 0)

    return f


def inversion_l1(signal: np.ndarray,
                 kernel: np.ndarray,
                 eta: float = 0.1,
                 max_iter: int = 200,
                 tol: float = 1e-6) -> np.ndarray:
    """L1 regularized (sparse) inversion for enhanced spectral resolution.

    Minimize: ||A·f - M||² + η·||f||₁

    L1 regularization produces sparse (spiky) solutions with higher spectral
    resolution than L2 regularization. Components differing by a factor of
    ~1.1 can be distinguished (vs. ~3 for L2).

    Parameters
    ----------
    signal : np.ndarray
        Measured echo amplitudes, shape (n_echoes,).
    kernel : np.ndarray
        Kernel matrix, shape (n_echoes, n_bins).
    eta : float
        L1 regularization factor.
    max_iter : int
        Maximum iterations for ISTA algorithm.
    tol : float
        Convergence tolerance.

    Returns
    -------
    np.ndarray
        Sparse T2 distribution, shape (n_bins,).
    """
    n_bins = kernel.shape[1]
    f = np.zeros(n_bins)
    L = np.linalg.norm(kernel.T @ kernel, ord=2)
    step = 1.0 / (L + 1e-10)

    for _ in range(max_iter):
        residual = kernel @ f - signal
        grad = kernel.T @ residual
        f_new = f - step * grad
        # Soft thresholding (proximal operator for L1)
        f_new = np.sign(f_new) * np.maximum(np.abs(f_new) - eta * step, 0)
        # Non-negativity
        f_new = np.maximum(f_new, 0)

        if np.linalg.norm(f_new - f) < tol * (np.linalg.norm(f) + 1e-10):
            break
        f = f_new

    return f


def discrete_inversion(signal: np.ndarray,
                       echo_times: np.ndarray,
                       n_components: int = 3,
                       max_iter: int = 500,
                       tol: float = 1e-8) -> tuple:
    """Discrete (delta-function) inversion for NMR data.

    Fits a sum of discrete exponential components to the echo train.
    Unlike ILT, no regularization is needed; spectral resolution is maximized.

    Minimize: ||Σ a_k exp(-t/T2_k) - M(t)||²

    over amplitudes a_k > 0 and relaxation times T2_k > 0.

    Parameters
    ----------
    signal : np.ndarray
        CPMG echo amplitudes.
    echo_times : np.ndarray
        Echo times (ms).
    n_components : int
        Number of discrete components to fit.
    max_iter : int
        Maximum optimization iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    t2_values : np.ndarray
        Fitted T2 relaxation times (ms), shape (n_components,).
    amplitudes : np.ndarray
        Fitted amplitudes, shape (n_components,).
    residual_norm : float
        Final residual norm.
    """
    # Initialize with log-spaced T2 guesses
    t2_min_est = echo_times[1]
    t2_max_est = echo_times[-1] * 2
    t2 = np.logspace(np.log10(t2_min_est), np.log10(t2_max_est), n_components)
    amps = np.ones(n_components) * np.max(signal) / n_components

    best_residual = np.inf

    for iteration in range(max_iter):
        # Fix T2 values, solve for amplitudes (linear least squares)
        A_local = np.exp(-echo_times[:, None] / t2[None, :])
        amps_new, _, _, _ = np.linalg.lstsq(A_local, signal, rcond=None)
        amps = np.maximum(amps_new, 0)

        # Fix amplitudes, update T2 via gradient descent
        predicted = A_local @ amps
        residual = predicted - signal
        res_norm = np.linalg.norm(residual)

        if abs(best_residual - res_norm) < tol:
            break
        best_residual = res_norm

        for k in range(n_components):
            if amps[k] < 1e-10:
                continue
            dA_dT2 = (echo_times / t2[k]**2) * np.exp(-echo_times / t2[k])
            grad_T2 = 2.0 * amps[k] * np.dot(residual, dA_dT2)
            # Adaptive step size
            step = t2[k] * 0.01 / (iteration + 1)
            t2[k] = max(t2[k] - step * np.sign(grad_T2), echo_times[0] * 0.1)

    return t2, amps, best_residual


def compute_regularization_factor_brd(signal: np.ndarray,
                                      kernel: np.ndarray,
                                      noise_std: float) -> float:
    """Estimate optimal regularization factor using Butler-Reeds-Dawson method.

    The optimal α is found where the residual norm equals the expected noise
    level: ||A·f_α - M||² ≈ n_echoes · σ²

    Parameters
    ----------
    signal : np.ndarray
        Measured signal.
    kernel : np.ndarray
        Kernel matrix.
    noise_std : float
        Estimated noise standard deviation.

    Returns
    -------
    float
        Estimated optimal regularization factor.
    """
    n_echoes = len(signal)
    target_misfit = n_echoes * noise_std**2

    # Binary search for optimal alpha
    alpha_lo, alpha_hi = 1e-6, 1e6

    for _ in range(50):
        alpha_mid = np.sqrt(alpha_lo * alpha_hi)
        f = inversion_tikhonov(signal, kernel, alpha_mid, non_negative=True)
        misfit = np.sum((kernel @ f - signal)**2)

        if misfit < target_misfit:
            alpha_lo = alpha_mid
        else:
            alpha_hi = alpha_mid

        if abs(alpha_hi / alpha_lo - 1.0) < 0.01:
            break

    return np.sqrt(alpha_lo * alpha_hi)


def partition_fluids(t2_distribution: np.ndarray,
                     t2_basis: np.ndarray,
                     t2_cutoff_cbw: float = 3.0,
                     t2_cutoff_bvi: float = 33.0) -> dict:
    """Partition T2 distribution into fluid types.

    Standard NMR fluid typing cutoffs:
        - Clay-bound water (CBW): T2 < 3 ms
        - Capillary-bound water (BVI): 3 ms < T2 < 33 ms
        - Free fluid (FFI): T2 > 33 ms

    Parameters
    ----------
    t2_distribution : np.ndarray
        T2 distribution f(T2).
    t2_basis : np.ndarray
        T2 basis values (ms).
    t2_cutoff_cbw : float
        T2 cutoff for clay-bound water (ms).
    t2_cutoff_bvi : float
        T2 cutoff for bound vs. free water (ms).

    Returns
    -------
    dict
        Fluid volumes: 'CBW', 'BVI', 'FFI', 'total_porosity', 'T2_log_mean'.
    """
    total = np.sum(t2_distribution)
    if total < 1e-30:
        return {"CBW": 0, "BVI": 0, "FFI": 0, "total_porosity": 0, "T2_log_mean": 0}

    cbw_mask = t2_basis < t2_cutoff_cbw
    bvi_mask = (t2_basis >= t2_cutoff_cbw) & (t2_basis < t2_cutoff_bvi)
    ffi_mask = t2_basis >= t2_cutoff_bvi

    cbw = np.sum(t2_distribution[cbw_mask])
    bvi = np.sum(t2_distribution[bvi_mask])
    ffi = np.sum(t2_distribution[ffi_mask])

    # T2 log-mean
    positive = t2_distribution > 0
    if np.any(positive):
        t2_lm = np.exp(
            np.sum(t2_distribution[positive] * np.log(t2_basis[positive])) /
            np.sum(t2_distribution[positive])
        )
    else:
        t2_lm = 0

    return {
        "CBW": cbw,
        "BVI": bvi,
        "FFI": ffi,
        "total_porosity": total,
        "T2_log_mean": t2_lm,
    }
