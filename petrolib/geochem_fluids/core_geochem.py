"""Core geochemistry: Dean-Stark saturations, oxide closure, and OSI.

Dean-Stark porosity/saturation from retorted fluid volumes, sum-to-one closure of
an oxide (or elemental) suite, and the Jarvie oil/organic saturation index.

Unit-neutral volumes; oxides as weight fractions; ``S1`` in mg HC/g rock and
``TOC`` in wt%.  Sources: src2014_10/article2, src2015_12/article3,
src2016_04/article1, src2017_10/article5, src2019_02/article4, src2024_10.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]


def dean_stark(
    v_water: ArrayLike,
    v_hc: ArrayLike,
    *,
    v_bulk: ArrayLike | None = None,
    v_pore: ArrayLike | None = None,
) -> tuple[_Float, _Float, _Float]:
    """Dean-Stark porosity and saturations -> ``(phi, Sw, S_hc)``.

    Pore volume is ``v_pore`` if given, else ``v_water + v_hc``.  ``phi`` is
    ``pore/v_bulk`` when ``v_bulk`` is given (else NaN).  ``Sw = v_water/pore``,
    ``S_hc = v_hc/pore``.
    """
    vw = np.asarray(v_water, np.float64)
    vh = np.asarray(v_hc, np.float64)
    pore = vw + vh if v_pore is None else np.asarray(v_pore, np.float64)
    if v_bulk is None:
        phi = np.full(np.broadcast(vw, vh).shape, np.nan, dtype=np.float64)
    else:
        phi = pore / np.asarray(v_bulk, np.float64)
    return np.asarray(phi), np.asarray(vw / pore), np.asarray(vh / pore)


def oxide_closure(oxides: ArrayLike, *, axis: int = -1) -> tuple[_Float, _Float]:
    """Normalize an oxide (or elemental) suite to sum to one -> ``(closed, factor)``.

    ``factor = 1/sum(oxides)``; ``closed = factor*oxides``.
    """
    o = np.asarray(oxides, np.float64)
    factor = 1.0 / np.sum(o, axis=axis, keepdims=True)
    return np.asarray(factor * o), np.asarray(np.squeeze(factor, axis=axis))


def osi(s1_mg_g: ArrayLike, toc_wt_pct: ArrayLike) -> _Float:
    """Oil saturation index ``OSI = 100*S1/TOC`` (mg HC / g TOC; Jarvie producibility)."""
    return np.asarray(100.0 * np.asarray(s1_mg_g, np.float64) / np.asarray(toc_wt_pct, np.float64))
