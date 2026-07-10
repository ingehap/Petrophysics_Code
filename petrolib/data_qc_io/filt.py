"""Depth-series filtering: smoothing, moving statistics, tool response,
2-D median filter, window feature stacks, bed-boundary detection.

Edge conventions differ per idiom and are preserved: kernel smoothing
(boxcar/Gaussian) zero-pads via ``np.convolve(mode='same')``; the median
smoother and the feature stack edge-pad (``np.pad(mode='edge')``, i.e.
scipy's ``'nearest'``); moving statistics shrink the window at the edges.
All spreads are population (``ddof=0``).

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2020_10/article3 -- Article 3: Automatic Detection of Anomalous Density Measurements due to
  Wellbore Cave-in. Sen, Ong, Kainkaryam, Sharma (2020). DOI: 10.30632/PJV61N5-2020a3. Petrophysics
  Vol. 61 No. 5 (Oct 2020).
src2023_04/article10 -- Article 10: Machine-Learning-Based Deconvolution Method Provides High-
  Resolution Fast Inversion of Induction Log Data. Hagiwara (2023). DOI: 10.30632/PJV64N2-2023a10.
  Petrophysics Vol. 64 No. 2 (Apr 2023) — SPWLA AI/ML Special Issue.
src2023_06/article9 -- Aerens, P., Torres-Verdin, C., Espinoza, N. "Experimental Time-Lapse
  Visualization of Mud-Filtrate Invasion and Mudcake Deposition Using X-Ray Radiography"
  Petrophysics, Vol. 64, No. 3 (June 2023), pp. 448-461 DOI: 10.30632/PJV64N3-2023a9.
src2023_08/article5 -- Merletti, G., Rabinovich, M., Al Hajri, S., Dawson, W., Farmer, R., Ambia,
  J., Torres-Verdín, C. (2023). "New Iterative Resistivity Modeling Workflow Reduces Uncertainty in
  the Assessment of Water Saturation in Deeply Invaded Reservoirs", Petrophysics, Vol. 64, No. 4,
  pp. 555-567. DOI: 10.30632/PJV64N4-2023a5.
src2023_10/article_03 -- Okwoli, E. and Potter, D. K. (2023). "Probe Screening Techniques for
  Rapid, High-Resolution Core Analysis and Their Potential Usefulness for Energy Transition
  Applications." Petrophysics, 64(5), 640-655. DOI: 10.30632/PJV64N5-2023a3.
src2023_10/article_09 -- Bennis, M. and Torres-Verdin, C. (2023). "Numerical Simulation of Well
  Logs Based on Core Measurements: An Effective Method for Data Quality Control and Improved
  Petrophysical Interpretation." Petrophysics, 64(5), 753-772. DOI: 10.30632/PJV64N5-2023a9.
src2024_06/article2 -- Strobel, J. (2024). "Petrophysical Analyses for Supporting the Search for a
  Claystone-Hosted Nuclear Repository." Petrophysics 65(3), 302-316. DOI: 10.30632/PJV65N3-2024a2.
src2024_10/ml_permeability -- Automatic Permeability Estimation: ML Methods vs Conventional Models.
  Based on: Raheem et al. (2024), "Best Practices in Automatic Permeability Estimation: Machine-
  Learning Methods vs. Conventional Petrophysical Models", Petrophysics, Vol. 65, No. 5, pp.
  789-812.
src2024_10/thin_bed_nmr -- Thin-Bed NMR Response Characterisation in Horizontal Wells. Based on:
  Ramadan et al. (2024), "Characterizing Thin-Bed Responses in Horizontal Wells Using LWD NMR
  Tools: Insights From a Water Tank Experiment", Petrophysics, Vol. 65, No. 5, pp. 765-771.
src2026_04/a04 -- Liu, Z., Zhang, X., Fan, Q., Zhou, L., Zhang, Y., Zhao, Z., and Zhang, Z. (2026).
  Intelligent Sensors and Algorithms for Diagnosing Downhole Operating Conditions of Wireline
  Logging Instruments. Petrophysics, 67(2), 295–317. DOI: 10.30632/PJV67N2-2026a4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]

_MOVING_STATS = ("mean", "std", "var", "mad", "cv")


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


def smooth(
    x: ArrayLike,
    window: int = 5,
    kind: str = "boxcar",
    *,
    sigma: float | None = None,
) -> _Float:
    """Smooth a 1-D curve with a boxcar, Gaussian, or median filter.

    ``'boxcar'``: centered moving average, ``np.convolve(mode='same')``
    (zero-padded edges); ``window<=1`` returns a copy.  ``'gaussian'``:
    normalized Gaussian kernel of standard deviation ``sigma`` samples
    (kernel length ``max(3, round(6*sigma)|1)``), same convolution.
    ``'median'``: running median over an odd ``window`` with edge-replicated
    padding.  Sources: src2023_10/article_03 (boxcar), src2026_04/a04
    (moving_average), src2024_06/article2 (median, mode='nearest').
    """
    arr = _arr(x)
    if kind == "boxcar":
        if window <= 1:
            return arr.copy()
        k = np.ones(window) / window
        return np.asarray(np.convolve(arr, k, mode="same"))
    if kind == "gaussian":
        if sigma is None:
            raise ValueError("smooth: sigma is required for kind='gaussian'")
        n = max(3, int(round(6.0 * sigma)) | 1)
        t = np.arange(n) - n // 2
        g = np.exp(-0.5 * (t / sigma) ** 2)
        g /= g.sum()
        return np.asarray(np.convolve(arr, g, mode="same"))
    if kind == "median":
        if window < 1 or window % 2 == 0:
            raise ValueError("smooth: kind='median' requires an odd window")
        pad = window // 2
        padded = np.pad(arr, pad, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, window)
        return np.asarray(np.median(windows, axis=-1))
    raise ValueError(f"smooth: unknown kind {kind!r}")


def moving_stat(
    x: ArrayLike,
    window: int,
    stat: str = "mean",
    *,
    center: bool = True,
) -> _Float:
    """Moving-window statistic with shrinking edge windows.

    ``center=True`` uses the symmetric window ``x[i-window//2 : i+window//2+1]``
    (clamped at the edges); ``center=False`` uses the causal/trailing window
    ``x[i-window+1 : i+1]``.  ``stat`` is ``'mean'``, ``'std'``, ``'var'``
    (all ``ddof=0``), ``'mad'`` (mean absolute deviation about the window
    mean), or ``'cv'`` (``std/mean``, 0 where the mean is 0).  Sources:
    src2020_10/article3 (rolling_cv, half-width convention), src2024_10/
    ml_permeability (centered mean/var), src2026_04/a04 (causal std/msd).
    """
    if stat not in _MOVING_STATS:
        raise ValueError(f"moving_stat: unknown stat {stat!r}")
    arr = _arr(x)
    n = arr.size
    half = window // 2
    out = np.zeros(n)
    for i in range(n):
        if center:
            seg = arr[max(0, i - half) : min(n, i + half + 1)]
        else:
            seg = arr[max(0, i - window + 1) : i + 1]
        if stat == "mean":
            out[i] = np.mean(seg)
        elif stat == "std":
            out[i] = np.std(seg)
        elif stat == "var":
            out[i] = np.var(seg)
        elif stat == "mad":
            out[i] = np.mean(np.abs(seg - np.mean(seg)))
        else:  # cv
            m = seg.mean()
            out[i] = seg.std() / m if m != 0 else 0.0
    return out


def tool_response(x: ArrayLike, dz: float, fwhm: float) -> _Float:
    """Convolve a fine-scale curve with a Gaussian tool vertical response.

    ``sigma = fwhm / (2*sqrt(2*ln 2))``; the kernel spans ~6 sigma with an
    odd length (``max(3, round(6*sigma/dz)|1)``), is unit-normalized, and is
    applied with ``np.convolve(mode='same')``.  This is the log-simulation /
    upscaling operator for a tool of aperture ``fwhm`` on a grid of spacing
    ``dz`` (same length units).  Source: src2023_10/article_09
    (vertical_response).
    """
    arr = _arr(x)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    n = max(3, int(round(6 * sigma / dz)) | 1)
    t = (np.arange(n) - n // 2) * dz
    g = np.exp(-0.5 * (t / sigma) ** 2)
    g /= g.sum()
    return np.asarray(np.convolve(arr, g, mode="same"))


def median_filter2d(img: ArrayLike, size: int = 3) -> _Float:
    """Pure-numpy 2-D median filter with edge-replicated padding.

    Builds the ``size*size`` stack of shifted views of the padded image and
    takes the median across it.  Source: src2023_06/article9
    (median_filter_3x3), generalized to odd ``size``.
    """
    arr = _arr(img)
    if arr.ndim != 2:
        raise ValueError("median_filter2d: img must be 2-D")
    if size < 1 or size % 2 == 0:
        raise ValueError("median_filter2d: size must be odd")
    pad = size // 2
    padded = np.pad(arr, pad, mode="edge")
    h, w = arr.shape
    stack = np.empty((size * size, h, w))
    k = 0
    for dy in range(size):
        for dx in range(size):
            stack[k] = padded[dy : dy + h, dx : dx + w]
            k += 1
    return np.asarray(np.median(stack, axis=0))


def window_features(x: ArrayLike, window: int = 11) -> _Float:
    """Edge-padded sliding-window feature stack for depth-indexed ML.

    A 1-D curve of length ``n`` becomes ``(n, window)`` — row ``i`` is the
    window centered on sample ``i``, with ``np.pad(mode='edge')`` handling
    the ends so every depth keeps a full window.  A 2-D ``(n, d)`` matrix
    stacks each column in turn to ``(n, d*window)``.  ``window`` must be odd.
    Sources: src2023_04/article10 & article11 (induction deconvolution
    feature stacks).
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("window_features: window must be odd")
    arr = _arr(x)
    if arr.ndim == 1:
        pad = window // 2
        padded = np.pad(arr, pad, mode="edge")
        return np.lib.stride_tricks.sliding_window_view(padded, window).copy()
    if arr.ndim == 2:
        return np.asarray(
            np.hstack([window_features(arr[:, j], window) for j in range(arr.shape[1])])
        )
    raise ValueError("window_features: x must be 1-D or 2-D")


def detect_bed_boundaries(
    curve: ArrayLike,
    window: int = 11,
    threshold: float | None = None,
) -> NDArray[np.int_]:
    """Bed-boundary indices from a derivative-times-local-variability metric.

    ``metric = |gradient(curve)| * sqrt(moving var)`` (variance over the
    centered ``window``, edges left 0); boundaries are the local maxima of
    the metric above ``threshold`` (default adaptive ``mean + 2*std``).
    Source: src2023_08/article5 (detect_bed_boundaries); simpler
    gradient-threshold and lag-variance variants exist in src2024_10/
    thin_bed_nmr and src2024_06/article2.
    """
    arr = _arr(curve)
    n = arr.size
    half = window // 2
    deriv = np.gradient(arr)
    variance = np.zeros(n)
    for i in range(half, n - half):
        variance[i] = np.var(arr[i - half : i + half + 1])
    metric = np.abs(deriv) * np.sqrt(variance)
    if threshold is None:
        threshold = float(np.mean(metric) + 2.0 * np.std(metric))
    boundaries = []
    for i in range(1, n - 1):
        if metric[i] > threshold and metric[i] >= metric[i - 1] and metric[i] >= metric[i + 1]:
            boundaries.append(i)
    return np.array(boundaries, dtype=int)
