"""Depth matching / curve alignment: DTW, cross-correlation shift, warping.

Alignment numerics shared by the depth-matching, core-to-log homing and
log-QC articles.  The corpus uses several conventions, which are exposed as
parameters so a caller can reproduce its exact variant:

* :func:`dtw` -- banded dynamic time warping with a selectable local cost
  (``'sqdiff'`` / ``'absdiff'``) and an optional square-root of the total cost.
* :func:`xcorr_shift` -- integer-lag bulk shift by maximum Pearson correlation,
  with ``edge='trim'`` (overlap only) or ``edge='wrap'`` (circular ``roll``).
* physical-unit core-to-log homing, windowed local lags, and shift application
  by integer roll/fill or by continuous interpolation.

Sign convention: a returned lag/shift is the value to feed back into
:func:`apply_integer_shift` / :func:`apply_depth_shift` to align the target onto
the reference; a positive integer lag moves the target to greater depth (index).

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2018_12/article10_ml_depth_matching -- Article 10: Machine-Learning-Based Automatic Well-Log
  Depth Matching. Zimmermann, Liang, Zeroug (2018). DOI: 10.30632/PJV59N6Y2018a9. Petrophysics Vol.
  59 No. 6 (Dec 2018) — Special Issue: Data-Driven Analytics in Logging and Petrophysics.
src2019_08/article1_ml_well_log_correlation -- Article 1: A Machine-Learning-Based Approach to
  Assistive Well-Log Correlation. Brazell, Bayeh, Ashby, Burton (2019). DOI:
  10.30632/PJV60N4-2019a1. Petrophysics Vol. 60 No. 4 (Aug 2019).
src2019_10/article3_ml_depth_matching -- Article 3: A Machine-Learning Framework for Automating
  Well-Log Depth Matching. Le, Liang, Zimmermann, Zeroug, Heliot (2019). DOI:
  10.30632/PJV60N5-2019a3. Petrophysics Vol. 60 No. 5 (Oct 2019).
src2022_02/article3_log_analytics_dtw_xcorr -- Article 3: Automated Log Data Analytics Workflow -
  The Value of Data Access and Management to Reduced Turnaround Time for Log Analysis. Torres
  Caceres, Duffaut, Westad, Stovas, Johansen, Jenssen (2022). DOI: 10.30632/PJV63N1-2022a3.
  Petrophysics Vol. 63 No. 1 (Feb 2022).
src2023_02/article9_depth_matching -- Article 9: Automated Well-Log Pattern Alignment and Depth-
  Matching Techniques: An Empirical Review and Recommendations. Ezenkwu, Guntoro, Starkey, Vaziri,
  Addario (2023). DOI: 10.30632/PJV64N1-2023a9. Petrophysics Vol. 64 No. 1 (Feb 2023).
src2025_06/toc_prediction -- Comparative Analysis of TOC Logging Evaluation Methods Using Machine
  Learning Implements the methodology from: Dong, M., Shang, J., Tian, L., Wu, M., and Nie, X.,
  2025, "Comparative Analysis of TOC Logging Evaluation Methods Using Machine Learning – A Case
  Study of the Ordos Basin-Yanchang Formation," Petrophysics, Vol. 66, No. 3, pp. 425–448.
src2026_02/depth_alignment -- Article 4: Dynamic Depth Alignment of Well Logs: A Continuous
  Optimization Framework for Enhanced Petrophysical and Rock Physics Interpretation. Westeng et al.
  (2026), Petrophysics, 67(1), 54-67. DOI: 10.30632/PJV67N1-2026a4.
src2026_04/a12_depth_shifting_ml -- Pan, W., Fu, L., Xu, C., Ashby, M., Lee, H., Lee, J., Meng, F.,
  Chen, S., Ye, Y., Jiang, H., Kim, H., Kong, H., Baek, I., Baek, J., Sun, X., Sun, H., Li, S.,
  Zhao, Z., Ke, Y., ... and Park, J. (2026). Automatic Well-Log Depth Shifting With Data-Driven
  Approaches: A Contest Summary. Petrophysics, 67(2), 437–462. DOI: 10.30632/PJV67N2-2026a12.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


class DtwResult(NamedTuple):
    """Result of :func:`dtw` / :func:`rddtw`."""

    distance: float
    path: list[tuple[int, int]]
    cost: _Float


class ShiftResult(NamedTuple):
    """Result of :func:`xcorr_shift`."""

    lag: int
    corr: float
    curve: _Float | None


def _pearson(a: _Float, b: _Float) -> float:
    if a.size < 2:
        return float("nan")
    sa = a.std()
    sb = b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# --- dynamic time warping -----------------------------------------------------


def dtw(
    ref: ArrayLike,
    target: ArrayLike,
    *,
    band: int | None = None,
    cost: str = "sqdiff",
    root: bool = False,
) -> DtwResult:
    """Banded dynamic time warping of ``target`` onto ``ref``.

    Accumulates ``D[i,j] = local(i,j) + min(D[i-1,j-1], D[i-1,j], D[i,j-1])`` on a
    padded matrix, where ``local`` is the squared (``cost='sqdiff'``) or absolute
    (``cost='absdiff'``) difference.  ``band`` is the Sakoe-Chiba half-width
    (``None`` = full).  Returns the total ``distance`` (``sqrt`` of it if
    ``root=True``), the backtracked 0-based ``path`` and the accumulated matrix.

    Sources: src2018_12/article10_ml_depth_matching,
    src2019_08/article1_ml_well_log_correlation, src2019_10/article3_ml_depth_matching,
    src2022_02/article3_log_analytics_dtw_xcorr, src2023_02/article9_depth_matching.
    """
    x = _arr(ref)
    y = _arr(target)
    n, m = x.size, y.size
    if cost not in ("sqdiff", "absdiff"):
        raise ValueError(f"unknown cost {cost!r}; use 'sqdiff' or 'absdiff'")
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = 1 if band is None else max(1, i - band)
        hi = m if band is None else min(m, i + band)
        for j in range(lo, hi + 1):
            diff = x[i - 1] - y[j - 1]
            local = diff * diff if cost == "sqdiff" else abs(diff)
            d[i, j] = local + min(d[i - 1, j - 1], d[i - 1, j], d[i, j - 1])
    total = d[n, m]
    distance = float(np.sqrt(total)) if root else float(total)
    path = _backtrack(d, n, m)
    return DtwResult(distance, path, d)


def _backtrack(d: _Float, n: int, m: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([d[i - 1, j - 1], d[i - 1, j], d[i, j - 1]]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


def _derivative(seq: _Float) -> _Float:
    d = np.zeros_like(seq)
    if seq.size >= 3:
        d[1:-1] = ((seq[1:-1] - seq[:-2]) + (seq[2:] - seq[:-2]) / 2.0) / 2.0
    if seq.size >= 2:
        d[0] = d[1]
        d[-1] = d[-2]
    return d


def rddtw(
    ref: ArrayLike,
    target: ArrayLike,
    *,
    tau: float = 4.0,
    lam: float = 0.5,
    band: int | None = None,
) -> DtwResult:
    """Regularized derivative DTW (RDDTW) with an excessive-warping penalty.

    The local cost combines shape and Keogh-Pazzani derivative (trend)
    absolute differences ``|s-t| + tau*|s'-t'|`` plus an off-diagonal penalty
    ``lam*|i/n - j/m|`` that discourages excessive warping.
    """
    s = _arr(ref)
    t = _arr(target)
    n, m = s.size, t.size
    d_shape = np.abs(s[:, None] - t[None, :])
    d_trend = np.abs(_derivative(s)[:, None] - _derivative(t)[None, :])
    d_joint = d_shape + tau * d_trend
    c = np.full((n + 1, m + 1), np.inf)
    c[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = 1 if band is None else max(1, i - band)
        hi = m if band is None else min(m, i + band)
        for j in range(lo, hi + 1):
            penalty = lam * abs(i / n - j / m)
            local = d_joint[i - 1, j - 1] + penalty
            c[i, j] = local + min(c[i - 1, j], c[i, j - 1], c[i - 1, j - 1])
    path = _backtrack(c, n, m)
    return DtwResult(float(c[n, m]), path, c)


def warp_to_reference(
    target: ArrayLike, path: list[tuple[int, int]], n_ref: int, *, reduce: str = "mean"
) -> _Float:
    """Collapse a DTW ``path`` into a ``target`` curve aligned to the reference grid.

    For each reference index the matched ``target`` samples are combined
    (``reduce='mean'``); reference indices absent from the path are filled from
    the nearest populated index.

    Sources: src2023_02/article9_depth_matching.
    """
    if reduce != "mean":
        raise ValueError(f"unknown reduce {reduce!r}; use 'mean'")
    t = _arr(target)
    acc = np.zeros(n_ref)
    counts = np.zeros(n_ref)
    for i, j in path:
        acc[i] += t[j]
        counts[i] += 1.0
    out = np.where(counts > 0, acc / np.where(counts > 0, counts, 1.0), np.nan)
    valid = np.flatnonzero(counts > 0)
    if valid.size:
        idx = np.arange(n_ref)
        nearest = valid[np.abs(idx[:, None] - valid[None, :]).argmin(axis=1)]
        out = np.where(counts > 0, out, out[nearest])
    return np.asarray(out)


def path_depth_shifts(
    path: list[tuple[int, int]], depth_ref: ArrayLike, depth_target: ArrayLike
) -> tuple[_Float, _Float]:
    """Per-depth correction from a DTW path: ``(depth_ref[i], depth_target[j]-depth_ref[i])``.

    Sources: src2026_04/a12_depth_shifting_ml.
    """
    dref = _arr(depth_ref)
    dtgt = _arr(depth_target)
    depths = np.array([dref[i] for i, _ in path])
    shifts = np.array([dtgt[j] - dref[i] for i, j in path])
    return depths, shifts


# --- correlation-optimised warping -------------------------------------------


def _segment_corr(a: _Float, b: _Float) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    b_rs = np.interp(np.linspace(0.0, b.size - 1, a.size), np.arange(b.size), b)
    return float(np.corrcoef(a, b_rs)[0, 1])


def cow(
    ref: ArrayLike, target: ArrayLike, *, n_segments: int = 10, slack: int = 8
) -> tuple[_Float, _Float]:
    """Correlation-optimised warping -> ``(aligned, warp)``.

    Greedily adjusts ``n_segments`` interior segment boundaries of ``target``
    within +/-``slack`` to maximise the piecewise segment correlation with
    ``ref``, then piecewise-linearly remaps ``target`` onto the reference grid.

    Sources: src2023_02/article9_depth_matching.
    """
    x = _arr(ref)
    y = _arr(target)
    n, m = x.size, y.size
    ref_bounds = np.linspace(0, n, n_segments + 1, dtype=int)
    tgt_bounds = np.linspace(0, m, n_segments + 1).astype(int)
    for k in range(1, n_segments):
        best_b = tgt_bounds[k]
        best_score = -np.inf
        for db in range(-slack, slack + 1):
            b = int(np.clip(tgt_bounds[k] + db, tgt_bounds[k - 1] + 2, tgt_bounds[k + 1] - 2))
            score = _segment_corr(
                x[ref_bounds[k - 1] : ref_bounds[k]], y[tgt_bounds[k - 1] : b]
            ) + _segment_corr(x[ref_bounds[k] : ref_bounds[k + 1]], y[b : tgt_bounds[k + 1]])
            if score > best_score:
                best_score = score
                best_b = b
        tgt_bounds[k] = best_b
    warp = np.interp(np.arange(n), ref_bounds, tgt_bounds)
    aligned = np.interp(warp, np.arange(m), y)
    return np.asarray(aligned), np.asarray(warp)


# --- cross-correlation shift --------------------------------------------------


def xcorr_shift(
    ref: ArrayLike,
    target: ArrayLike,
    *,
    max_lag: int = 50,
    edge: str = "trim",
    return_curve: bool = False,
) -> ShiftResult:
    """Integer-lag bulk shift of ``target`` onto ``ref`` by maximum correlation.

    Scans ``lag`` in ``[-max_lag, max_lag]`` maximising the Pearson correlation.
    ``edge='trim'`` correlates only the overlapping samples; ``edge='wrap'`` uses
    a circular ``np.roll``.  Returns the best ``lag``, its correlation and (if
    ``return_curve``) the full correlation-vs-lag curve.

    Sources: src2018_12/article10_ml_depth_matching,
    src2019_08/article1_ml_well_log_correlation, src2019_10/article3_ml_depth_matching,
    src2026_04/a12_depth_shifting_ml.
    """
    x = _arr(ref)
    y = _arr(target)
    n = x.size
    lags = np.arange(-max_lag, max_lag + 1)
    curve = np.full(lags.size, np.nan)
    for k, lag in enumerate(lags):
        if edge == "wrap":
            r, t = x, np.roll(y, lag)
        elif edge == "trim":
            if lag >= 0:
                r, t = x[lag:], y[: n - lag]
            else:
                r, t = x[: n + lag], y[-lag:]
        else:
            raise ValueError(f"unknown edge {edge!r}; use 'trim' or 'wrap'")
        curve[k] = _pearson(r, t)
    if np.all(np.isnan(curve)):
        best = max_lag  # lag 0
    else:
        best = int(np.nanargmax(curve))
    return ShiftResult(int(lags[best]), float(curve[best]), curve if return_curve else None)


def xcorr_shift_depth(
    depth_a: ArrayLike,
    a: ArrayLike,
    depth_b: ArrayLike,
    b: ArrayLike,
    *,
    max_shift: float = 2.0,
    step: float = 0.125,
    use_abs_corr: bool = False,
) -> tuple[float, float]:
    """Physical-unit core-to-log homing by interpolated correlation.

    Slides ``a`` (on ``depth_a``) against ``b`` (on ``depth_b``) over shifts in
    ``[-max_shift, max_shift]`` in ``step`` increments, interpolating ``b`` onto
    the shifted ``depth_a`` and maximising the Pearson correlation (its absolute
    value if ``use_abs_corr``).  ``b`` may be a single curve or ``(n, k)``
    multi-curve array.  Returns ``(best_shift, best_corr)``.

    Sources: src2025_06/toc_prediction.
    """
    da = _arr(depth_a)
    av = _arr(a)
    db = _arr(depth_b)
    bv = _arr(b)
    cols = [bv] if bv.ndim == 1 else [bv[:, c] for c in range(bv.shape[1])]
    shifts = np.arange(-max_shift, max_shift + step, step)
    best_score = -np.inf
    best_shift = 0.0
    best_corr = float("nan")
    for s in shifts:
        shifted = da + s
        for col in cols:
            interp_vals = np.interp(shifted, db, col)
            r = _pearson(av, interp_vals)
            if np.isnan(r):
                continue
            score = abs(r) if use_abs_corr else r
            if score > best_score:
                best_score = score
                best_shift = float(s)
                best_corr = float(r)
    return best_shift, best_corr


def local_shifts(
    ref: ArrayLike, target: ArrayLike, *, window: int = 60, step: int = 30, max_lag: int = 20
) -> _Float:
    """Windowed non-wrapping lag profile of ``target`` relative to ``ref``.

    Over sliding windows (length ``window``, stride ``step``) picks the lag that
    maximises the mean-removed full cross-correlation, clipped to
    ``[-max_lag, max_lag]``.

    Sources: src2018_12/article10_ml_depth_matching.
    """
    r = _arr(ref)
    t = _arr(target)
    length = r.size
    out = []
    for s in range(0, length - window, step):
        rw = r[s : s + window]
        tw = t[s : s + window]
        corr = np.correlate(rw - rw.mean(), tw - tw.mean(), mode="full")
        lag = int(corr.argmax() - (tw.size - 1))
        out.append(int(np.clip(lag, -max_lag, max_lag)))
    return np.asarray(out, dtype=float)


# --- shift application --------------------------------------------------------


def apply_integer_shift(log: ArrayLike, lag: int, *, fill: float = np.nan) -> _Float:
    """Shift ``log`` by an integer ``lag`` samples without wrapping; vacated ends get ``fill``.

    A positive ``lag`` moves samples to greater depth (index); the exposed edge
    is filled with ``fill``.

    Sources: src2026_04/a12_depth_shifting_ml.
    """
    x = _arr(log)
    n = x.size
    out = np.full(n, fill)
    if lag >= 0:
        out[lag:] = x[: n - lag]
    else:
        out[: n + lag] = x[-lag:]
    return out


def apply_depth_shift(values: ArrayLike, depth: ArrayLike, shift: ArrayLike) -> _Float:
    """Apply a continuous (possibly per-depth) ``shift`` to ``values`` by interpolation.

    Resamples ``values`` from ``depth + shift`` back onto ``depth`` with
    endpoint clamping; ``shift`` may be a scalar bulk shift or a per-sample
    array.

    Sources: src2026_02/depth_alignment.
    """
    v = _arr(values)
    z = _arr(depth)
    shifted_z = z + _arr(shift)
    return np.asarray(np.interp(z, shifted_z, v, left=v[0], right=v[-1]))
