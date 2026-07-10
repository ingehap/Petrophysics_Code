"""Curve normalization against a reference well or target moments.

Only the reference-based normalizations live here; plain feature scaling is
already canon in :mod:`petrolib.ml_stats` (``zscore``, ``minmax``,
``affine_rescale``).  Population std (``ddof=0``) throughout, matching every
article copy.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2015_04/article4 -- Article 4: Microresistivity Curve Extraction from Borehole Microimager Data.
  Roslin (2015). Petrophysics Vol. 56, No. 2 (April 2015), pp. 140-146. DOI: none assigned (this
  issue predates SPWLA DOI assignment).
src2016_12/article6 -- Article 6 (Technical Note): Normalizing Gamma-Ray Logs Acquired from a
  Mixture of Vertical and Horizontal Wells in the Haynesville Shale. Xu, Bayer, Wunderle, Bansal
  (2016). Petrophysics Vol. 57, No. 6 (December 2016), pp. 638-643. DOI: none assigned (this issue
  predates SPWLA DOI assignment).
src2021_12/article01 -- Article 1: Data Quality Considerations for Petrophysical Machine-Learning
  Models. McDonald (2021). DOI: 10.30632/PJV62N6-2021a1. Petrophysics Vol. 62 No. 6 (Dec 2021).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import ml_stats

_Float = NDArray[np.float64]


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


def normalize_to_reference(
    x: ArrayLike,
    ref_lo: float,
    ref_hi: float,
    *,
    in_lo: float | None = None,
    in_hi: float | None = None,
    pct: tuple[float, float] | None = (5.0, 95.0),
) -> _Float:
    """Two-point (Shier 2004) normalization onto reference-well endpoints.

    ``out = ref_lo + (ref_hi - ref_lo) * (x - in_lo) / (in_hi - in_lo)``.

    ``in_lo``/``in_hi`` are the input-well endpoints; when omitted they are
    taken as the ``pct`` percentiles of ``x`` (the Shier 5th/95th convention),
    or the min/max when ``pct=None`` (the image-histogram convention).  No
    clipping is applied.  The affine map itself is
    :func:`petrolib.ml_stats.affine_rescale` (bit-identical arrangement).
    Sources: src2021_12/article01 (normalize_reference), src2016_12/article6
    (histogram_normalize), src2015_04/article4 (histogram_scale).
    """
    v = _arr(x)
    if in_lo is None:
        in_lo = float(np.percentile(v, pct[0])) if pct is not None else float(np.min(v))
    if in_hi is None:
        in_hi = float(np.percentile(v, pct[1])) if pct is not None else float(np.max(v))
    return ml_stats.affine_rescale(v, src_lo=in_lo, src_hi=in_hi, dst_lo=ref_lo, dst_hi=ref_hi)


def match_moments(x: ArrayLike, target_mean: float, target_std: float) -> _Float:
    """Affine transform of ``x`` to the target mean and (population) std.

    ``out = (x - mean(x)) / std(x) * target_std + target_mean``; a constant
    input divides by zero rather than guessing a scale.  Source:
    src2016_12/article6 (affine_normalize).
    """
    v = _arr(x)
    return np.asarray((v - v.mean()) / v.std() * target_std + target_mean)
