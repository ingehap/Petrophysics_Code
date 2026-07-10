"""Signal-to-noise, stacking, and controlled noise injection.

SNR is the power ratio ``10*log10(P_signal/P_noise)``; stacking of ``n``
repeats improves amplitude SNR by ``sqrt(n)``.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2021_04/article5 -- Article 5: An Integrated Petrophysical Workflow for Fluid Characterization
  and Contacts Identification Using NMR Continuous and Stationary Measurements in a High-Porosity
  Sandstone Formation, Offshore Norway. Kozlowski, Chakraborty, Jambunathan, Lowrey, Balliet,
  Engelman, Anensen, Kotwicki, Johansen (2021). DOI: 10.30632/PJV62N2-2021a5. Petrophysics Vol. 62
  No. 2 (Apr 2021).
src2021_12/article01 -- Article 1: Data Quality Considerations for Petrophysical Machine-Learning
  Models. McDonald (2021). DOI: 10.30632/PJV62N6-2021a1. Petrophysics Vol. 62 No. 6 (Dec 2021).
src2025_08/anomaly_detection_vmd -- Module 8: Automated Anomaly Detection via Signal Mode
  Decomposition Implements ideas from: Wang et al., "Automated Anomaly Detection of Multimetallic
  Tubulars in Well Integrity Logs Using Signal Mode Decomposition and Physics-Informed Decision
  Making," Petrophysics, vol. 66, no. 4, pp. 647–661, August 2025.
src2025_08/fiber_optics_sensing -- Module 2: Fiber-Optic Sensing for Well Integrity Diagnosis
  Implements ideas from: Bazaid et al., "Pioneering Well Logging: The Role of Fiber Optics in
  Modern Monitoring for Well Integrity Diagnosis," Petrophysics, vol. 66, no. 4, pp. 555–565,
  August 2025.
src2025_12/carbon13_mr -- ¹³C Magnetic Resonance Relaxation-Time Wettability Characterisation.
  Petrophysics, 66(6), 1073–1089. DOI: 10.30632/PJV66N6-2025a11.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


def snr_db(
    signal: ArrayLike,
    *,
    noise: ArrayLike | None = None,
    noise_std: float | None = None,
    guard: float = 0.0,
) -> float:
    """Signal-to-noise ratio in dB: ``10*log10(mean(s^2)/P_noise)``.

    Give either a noise record/residual (``noise``, power ``mean(n^2)+guard``)
    or a known standard deviation (``noise_std``, power ``noise_std**2``).
    Zero noise power returns ``inf``; ``guard`` reproduces the additive
    ``+1e-30`` convention of the VMD paper.  Sources: src2025_08/
    fiber_optics_sensing (compute_snr), src2025_08/anomaly_detection_vmd.
    """
    sig_power = np.mean(_arr(signal) ** 2)
    if noise_std is not None and noise is None:
        noise_power = float(noise_std) ** 2
    elif noise is not None and noise_std is None:
        noise_power = float(np.mean(_arr(noise) ** 2)) + guard
    else:
        raise ValueError("snr_db: give exactly one of noise or noise_std")
    if noise_power == 0:
        return float("inf")
    return float(10.0 * np.log10(sig_power / noise_power))


def stack_gain(n: ArrayLike) -> _Float:
    """Amplitude SNR gain ``sqrt(n)`` from stacking ``n`` repeat measurements.

    Sources: src2021_04/article5 (stacking_snr), src2025_12/carbon13_mr.
    """
    return np.asarray(np.sqrt(_arr(n)))


def block_stack(x: ArrayLike, n: int, axis: int = -1) -> _Float:
    """Block-mean stacking: average consecutive groups of ``n`` along ``axis``.

    The output length is ``size // n``; a trailing partial block is dropped.
    Source: src2025_08/fiber_optics_sensing (temporal_stack).
    """
    if n < 1:
        raise ValueError("block_stack: n must be >= 1")
    arr = _arr(x)
    moved = np.moveaxis(arr, axis, -1)
    n_out = moved.shape[-1] // n
    trimmed = moved[..., : n_out * n]
    stacked = trimmed.reshape(*moved.shape[:-1], n_out, n).mean(axis=-1)
    return np.asarray(np.moveaxis(stacked, -1, axis))


def add_gaussian_noise(
    x: ArrayLike,
    sigma_fraction: float,
    rng: np.random.Generator | int | None = None,
) -> _Float:
    """Add zero-mean Gaussian noise with ``sigma = fraction * mean(|x|)``.

    The scale is a single scalar tied to the curve's mean absolute level.
    ``rng=None`` uses the deterministic ``default_rng(0)`` of the source
    article; pass a seed or Generator to vary.  Source: src2021_12/article01
    (Eqs. 9-10).
    """
    arr = _arr(x)
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(0 if rng is None else rng)
    sigma = sigma_fraction * np.abs(arr).mean()
    return np.asarray(arr + rng.normal(0.0, sigma, size=arr.shape))
