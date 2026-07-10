"""Log cleaning: sentinels, gap imputation, outliers, compositional closure.

Conventions observed across the corpus and kept here: statistical spreads use
population std (``ddof=0``); comparison against outlier bounds is strict
(``>`` / ``<``); the robust MAD scale uses the Gaussian-consistency factor
1.4826.  Sources: src2021_12/article01 (sentinels, z-score/IQR masks),
src2018_12/article11 (gap imputation, IQR), src2024_08/magnetic_permeability
(MAD despike), src2019_02/article4 (closure), src2023_08/article3
(renormalize), src2016_02/article3 & src2017_10/article5 (discrepancy).

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2014_10/article2 -- Article 2: Application and Quality Control of Core Data for the Development
  and Validation of Elemental Spectroscopy Log Interpretation Susan Herron, Michael Herron, Iain
  Pirie, Pablo Saldungaray, Paul Craddock, Alyssa Charsky, Marina Polyakov, Frank Shray, Ting Li
  (2014). Petrophysics Vol. 55, No. 5 (October 2014), pp. 392-414. DOI: none assigned (this issue
  predates SPWLA DOI assignment).
src2016_02/article3 -- Article 3: Low-Permeability Measurements: Insights. Profice, Hamon, Nicot
  (2016). Petrophysics Vol. 57, No. 1 (February 2016), pp. 30-40. DOI: none assigned (this issue
  predates SPWLA DOI assignment).
src2017_10/article5 -- Article 5: Lessons Learned in Permian Core Analysis: Comparison Between
  Retort, GRI, and Routine Methodologies. Blount, Croft, Driskill, Tepper (2017). Petrophysics Vol.
  58, No. 5 (October 2017), pp. 517-527. DOI: none assigned (this issue predates SPWLA DOI
  assignment).
src2018_12/article11 -- Article 11: Data Preconditioning for Predictive and Interpretive
  Algorithms: Importance in Data-Driven Analytics and Methods for Application. Frost, Quinn (2018).
  DOI: 10.30632/PJV59N6Y2018a10. Petrophysics Vol. 59 No. 6 (Dec 2018) — Special Issue: Data-Driven
  Analytics in Logging and Petrophysics.
src2019_02/article4 -- Article 4: Maintaining and Reconstructing In-Situ Saturations: A Comparison
  Between Whole Core, Sidewall Core, and Pressurized Sidewall Core in the Permian Basin. Blount,
  McMullen, Durand, Driskill (2019). DOI: 10.30632/PJV60N1Y2019a3. Petrophysics Vol. 60 No. 1 (Feb
  2019).
src2021_12/article01 -- Article 1: Data Quality Considerations for Petrophysical Machine-Learning
  Models. McDonald (2021). DOI: 10.30632/PJV62N6-2021a1. Petrophysics Vol. 62 No. 6 (Dec 2021).
src2023_08/article3 -- Jácomo, M.H., Hartmann, G.A., Rebelo, T.B., Mattos, N.H., Batezelli, A.,
  Leite, E.P. (2023). "Mineralogical Modeling and Petrophysical Properties of the Barra Velha
  Formation, Santos Basin, Brazil", Petrophysics, Vol. 64, No. 4, pp. 518-543. DOI:
  10.30632/PJV64N4-2023a3.
src2024_08/magnetic_permeability -- Applying Magnetic Susceptibility to Estimate Permeability From
  Drill Cuttings. Based on: Banks, J.Y., Tugwell, A.G., and Potter, D.K. (2024), "Applying Magnetic
  Susceptibility to Estimate Permeability From Drill Cuttings: A Case Study Constraining
  Uncertainty in the Culzean Triassic Reservoir," Petrophysics, 65(4), pp. 604-623. DOI:
  10.30632/PJV65N4-2024a13.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]
_Bool = NDArray[np.bool_]

#: Common LAS/DLIS no-data sentinel values.
SENTINELS = (-999.0, -999.25, -9999.0)

#: Gaussian-consistency factor: sigma ~= 1.4826 * MAD for normal data.
MAD_TO_SIGMA = 1.4826


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


def sentinels_to_nan(
    x: ArrayLike,
    sentinels: Sequence[float] = SENTINELS,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> _Float:
    """Replace LAS/DLIS no-data sentinels (e.g. -999.25) with NaN.

    Matching uses ``np.isclose`` (defaults ``rtol=1e-5, atol=1e-8``) so float
    round-trips of the sentinel still match.  Source: src2021_12/article01.
    """
    out = _arr(x).copy()
    for s in sentinels:
        out[np.isclose(out, s, rtol=rtol, atol=atol)] = np.nan
    return out


def impute_gaps(x: ArrayLike, index: ArrayLike | None = None) -> _Float:
    """Fill NaN runs by linear interpolation over the valid samples.

    ``index`` gives the sample coordinates (default ``arange``); leading and
    trailing NaNs are held flat at the nearest valid value (``np.interp``
    endpoint behavior).  Source: src2018_12/article11.
    """
    out = _arr(x).copy()
    idx = np.arange(out.size) if index is None else _arr(index)
    good = ~np.isnan(out)
    if not good.any():
        raise ValueError("impute_gaps: input has no finite values to interpolate from")
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def outlier_mask(
    x: ArrayLike,
    method: str = "zscore",
    *,
    threshold: float = 3.0,
    k: float = 1.5,
    side: str = "both",
) -> _Bool:
    """Boolean outlier mask by z-score, IQR whiskers, or robust MAD.

    ``'zscore'``: ``|z| > threshold`` with population std (guarded ``>1e-12``).
    ``'iqr'``: outside ``(Q1 - k*IQR, Q3 + k*IQR)``.
    ``'mad'``: further than ``threshold * MAD * 1.4826`` from the median.
    ``side`` restricts to ``'upper'``/``'lower'`` exceedances (the
    ferromagnetic-contaminant screen is upper-only).  Sources:
    src2021_12/article01, src2018_12/article11, src2024_08/magnetic_permeability.
    """
    if side not in {"both", "upper", "lower"}:
        raise ValueError(f"outlier_mask: unknown side {side!r}")
    arr = _arr(x)
    if method == "zscore":
        s = arr.std()
        z = (arr - arr.mean()) / (s if s > 1e-12 else 1.0)
        if side == "both":
            return np.asarray(np.abs(z) > threshold)
        return np.asarray(z > threshold if side == "upper" else z < -threshold)
    if method == "iqr":
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        if side == "both":
            return np.asarray((arr < lo) | (arr > hi))
        return np.asarray(arr > hi if side == "upper" else arr < lo)
    if method == "mad":
        med = np.median(arr)
        h = threshold * np.median(np.abs(arr - med)) * MAD_TO_SIGMA
        if side == "both":
            return np.asarray((arr < med - h) | (arr > med + h))
        return np.asarray(arr > med + h if side == "upper" else arr < med - h)
    raise ValueError(f"outlier_mask: unknown method {method!r}")


def despike(
    x: ArrayLike,
    *,
    threshold: float = 3.0,
    side: str = "both",
    replace: str = "interp",
) -> _Float:
    """Remove spikes flagged by the robust MAD criterion.

    Spikes are samples further than ``threshold * MAD * 1.4826`` from the
    median (see :func:`outlier_mask`).  ``replace='interp'`` fills them by
    linear interpolation from the surviving samples, ``'median'`` with the
    global median (the ferromagnetic-contaminant convention), ``'nan'`` with
    NaN.  Source: src2024_08/magnetic_permeability (one-sided, median).
    """
    out = _arr(x).copy()
    spikes = outlier_mask(out, "mad", threshold=threshold, side=side)
    if not spikes.any():
        return out
    if replace == "interp":
        idx = np.arange(out.size)
        good = ~spikes
        out[spikes] = np.interp(idx[spikes], idx[good], out[good])
    elif replace == "median":
        out[spikes] = np.median(_arr(x))
    elif replace == "nan":
        out[spikes] = np.nan
    else:
        raise ValueError(f"despike: unknown replace {replace!r}")
    return out


def closure_residual(*fracs: ArrayLike, target: float = 1.0) -> _Float:
    """Closure residual ``sum(fracs) - target`` (e.g. ``Sw+So+Sg-1``).

    Zero means the fractions close exactly.  Source: src2019_02/article4
    (saturation closure); the oxide-closure factor lives in
    ``petrolib.geochem_fluids``.
    """
    if not fracs:
        raise ValueError("closure_residual: at least one fraction is required")
    total = _arr(fracs[0])
    for f in fracs[1:]:
        total = total + _arr(f)
    return np.asarray(total - target)


def renormalize(fracs: ArrayLike, axis: int = -1) -> _Float:
    """Rescale fractions to sum to one along ``axis`` (compositional closure).

    A zero sum yields NaN/inf rather than a guessed composition.  Sources:
    src2023_08/article3 (mineral volumes), src2014_10/article2 (oxides, via
    ``petrolib.geochem_fluids.core_geochem.oxide_closure``).
    """
    f = _arr(fracs)
    return np.asarray(f / np.sum(f, axis=axis, keepdims=True))


def relative_discrepancy(a: ArrayLike, b: ArrayLike) -> _Float:
    """Relative discrepancy ``2|a-b| / (a+b)`` (difference over the mean).

    Both corpus arrangements (``2|a-b|/(a+b)`` and ``|a-b|/(0.5(a+b))``)
    round identically in IEEE arithmetic.  Sources: src2016_02/article3
    (deviation indicator, Eq. 6), src2017_10/article5.
    """
    return np.asarray(2.0 * np.abs(_arr(a) - _arr(b)) / (_arr(a) + _arr(b)))
