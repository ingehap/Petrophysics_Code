"""Core geochemistry: Dean-Stark saturations, oxide closure, and OSI.

Dean-Stark porosity/saturation from retorted fluid volumes, sum-to-one closure of
an oxide (or elemental) suite, and the Jarvie oil/organic saturation index.

Unit-neutral volumes; oxides as weight fractions; ``S1`` in mg HC/g rock and
``TOC`` in wt%.  Sources: src2014_10/article2, src2015_12/article3,
src2016_04/article1, src2017_10/article5, src2019_02/article4, src2024_10.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2014_10/article2 -- Article 2: Application and Quality Control of Core Data for the Development
  and Validation of Elemental Spectroscopy Log Interpretation Susan Herron, Michael Herron, Iain
  Pirie, Pablo Saldungaray, Paul Craddock, Alyssa Charsky, Marina Polyakov, Frank Shray, Ting Li
  (2014). Petrophysics Vol. 55, No. 5 (October 2014), pp. 392-414. DOI: none assigned (this issue
  predates SPWLA DOI assignment).
src2014_10/article2_elemental_spectroscopy_qc -- Article 2: Application and Quality Control of Core
  Data for the Development and Validation of Elemental Spectroscopy Log Interpretation Susan
  Herron, Michael Herron, Iain Pirie, Pablo Saldungaray, Paul Craddock, Alyssa Charsky, Marina
  Polyakov, Frank Shray, Ting Li (2014). Petrophysics Vol. 55, No. 5 (October 2014), pp. 392-414.
  DOI: none assigned (this issue predates SPWLA DOI assignment).
src2015_12/article3 -- Article 3: Petrophysical Characterization of Bitumen-Saturated Karsted
  Carbonates: Case Study of the Multibillion Barrel Upper Devonian Grosmont Formation, Northern
  Alberta, Canada. MacNeil (2015). Petrophysics Vol. 56, No. 6 (December 2015), pp. 592-614. DOI:
  none assigned (this issue predates SPWLA DOI assignment).
src2015_12/article3_grosmont_bitumen_carbonates -- Article 3: Petrophysical Characterization of
  Bitumen-Saturated Karsted Carbonates: Case Study of the Multibillion Barrel Upper Devonian
  Grosmont Formation, Northern Alberta, Canada. MacNeil (2015). Petrophysics Vol. 56, No. 6
  (December 2015), pp. 592-614. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2016_04/article1 -- Article 1: The Reservoir Producibility Index: a Metric to Assess Reservoir
  Quality in Tight-Oil Plays from Logs. Reeder, Craddock, Rylander, Pirie, Lewis, Kausik,
  Kleinberg, Yang, Pomerantz (2016). Petrophysics Vol. 57, No. 2 (April 2016), pp. 83-95. DOI: none
  assigned (this issue predates SPWLA DOI assignment).
src2016_04/article1_reservoir_producibility_index -- Article 1: The Reservoir Producibility Index:
  a Metric to Assess Reservoir Quality in Tight-Oil Plays from Logs. Reeder, Craddock, Rylander,
  Pirie, Lewis, Kausik, Kleinberg, Yang, Pomerantz (2016). Petrophysics Vol. 57, No. 2 (April
  2016), pp. 83-95. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2017_10/article5 -- Article 5: Lessons Learned in Permian Core Analysis: Comparison Between
  Retort, GRI, and Routine Methodologies. Blount, Croft, Driskill, Tepper (2017). Petrophysics Vol.
  58, No. 5 (October 2017), pp. 517-527. DOI: none assigned (this issue predates SPWLA DOI
  assignment).
src2019_02/article4 -- Article 4: Maintaining and Reconstructing In-Situ Saturations: A Comparison
  Between Whole Core, Sidewall Core, and Pressurized Sidewall Core in the Permian Basin. Blount,
  McMullen, Durand, Driskill (2019). DOI: 10.30632/PJV60N1Y2019a3. Petrophysics Vol. 60 No. 1 (Feb
  2019).
src2019_02/article4_insitu_saturation_core_comparison -- Article 4: Maintaining and Reconstructing
  In-Situ Saturations: A Comparison Between Whole Core, Sidewall Core, and Pressurized Sidewall
  Core in the Permian Basin. Blount, McMullen, Durand, Driskill (2019). DOI:
  10.30632/PJV60N1Y2019a3. Petrophysics Vol. 60 No. 1 (Feb 2019).
src2024_10 -- Petrophysics Vol. 65 No. 5 (Oct 2024) (issue-level reference).
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

    Sources: src2015_12/article3_grosmont_bitumen_carbonates,
    src2019_02/article4_insitu_saturation_core_comparison.
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

    Sources: src2014_10/article2_elemental_spectroscopy_qc.
    """
    o = np.asarray(oxides, np.float64)
    factor = 1.0 / np.sum(o, axis=axis, keepdims=True)
    return np.asarray(factor * o), np.asarray(np.squeeze(factor, axis=axis))


def osi(s1_mg_g: ArrayLike, toc_wt_pct: ArrayLike) -> _Float:
    """Oil saturation index ``OSI = 100*S1/TOC`` (mg HC / g TOC; Jarvie producibility).

    Sources: src2016_04/article1_reservoir_producibility_index.
    """
    return np.asarray(100.0 * np.asarray(s1_mg_g, np.float64) / np.asarray(toc_wt_pct, np.float64))
