"""Mud-gas composition ratios and fluid-type classification.

Haworth wetness/balance/character ratios, Pixler and Bernard ratios, composition
normalization, extraction-efficiency correction, and GOR/wetness fluid typing.

The wetness ratio is a fraction by default; pass ``percent=True`` for the 0-100
convention (the corpus uses both, and classifier thresholds assume one or the
other -- the ``classify_*`` helpers below take a matching flag).  Components are
the light-alkane readings C1..C5 (any consistent unit).  Sources:
src2021_02/article1, src2023_12/cely, src2024_08/*, src2026_02, src2026_06/a08.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2021_02/article1 -- Article 1 (Tutorial): Maximizing Value From Mudlogs - Integrated Approach to
  Determine Net Pay. Malik, Hanson, Clinch (2021). DOI: 10.30632/PJV62N1-2021t1. Petrophysics Vol.
  62 No. 1 (Feb 2021).
src2023_12/cely -- Cely et al. (2023), Petrophysics 64(6): 919-930. Reservoir oil viscosity
  estimation in the Breidablikk Field from advanced mud-gas ratios + cuttings geochemistry,
  compared against PVT.
src2024_08 -- Petrophysics Vol. 65 No. 4 (Aug 2024) — Special Issue on Advancements in Mud Logging
  (issue-level reference).
src2026_02 -- Petrophysics Vol. 67 No. 1 (Feb 2026) — SPWLA 66th Annual Symposium best papers
  (issue-level reference).
src2026_06/a08 -- Luo, P., Li, W., Lu, P., and Qubaisi, K. (2026). An Improved Mud Gas Ratio Method
  for Enhanced Fluid Identification While Drilling. Petrophysics, 67(3), 582-593. DOI:
  10.30632/PJV67N3-2026a8.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]


def normalize_composition(comps: ArrayLike, *, axis: int = -1) -> _Float:
    """Sum-to-one (zero-safe) closure of a composition along ``axis``."""
    c = np.asarray(comps, np.float64)
    total = np.sum(c, axis=axis, keepdims=True)
    return np.asarray(np.divide(c, total, out=np.zeros_like(c), where=total != 0))


def _total(c1: ArrayLike, c2: ArrayLike, c3: ArrayLike, c4: ArrayLike, c5: ArrayLike) -> _Float:
    return np.asarray(
        np.asarray(c1, np.float64)
        + np.asarray(c2, np.float64)
        + np.asarray(c3, np.float64)
        + np.asarray(c4, np.float64)
        + np.asarray(c5, np.float64)
    )


def wetness_ratio(
    c1: ArrayLike,
    c2: ArrayLike,
    c3: ArrayLike,
    c4: ArrayLike,
    c5: ArrayLike,
    *,
    percent: bool = True,
) -> _Float:
    """Haworth wetness ``Wh = (C2+C3+C4+C5)/(C1+..+C5)`` (percent by default)."""
    heavy = _total(0.0, c2, c3, c4, c5)
    wh = heavy / _total(c1, c2, c3, c4, c5)
    return np.asarray(wh * 100.0 if percent else wh)


def balance_ratio(
    c1: ArrayLike, c2: ArrayLike, c3: ArrayLike, c4: ArrayLike, c5: ArrayLike
) -> _Float:
    """Haworth balance ``Bh = (C1+..+C5)/(C3+C4+C5)`` (inf where the denominator is 0)."""
    heavy = _total(0.0, 0.0, c3, c4, c5)
    return np.asarray(
        np.divide(
            _total(c1, c2, c3, c4, c5), heavy, out=np.full_like(heavy, np.inf), where=heavy != 0
        )
    )


def character_ratio(c3: ArrayLike, c4: ArrayLike, c5: ArrayLike) -> _Float:
    """Haworth character ``Ch = (C4+C5)/C3`` (0 where C3 is 0)."""
    c3a = np.asarray(c3, np.float64)
    top = np.asarray(c4, np.float64) + np.asarray(c5, np.float64)
    return np.asarray(np.divide(top, c3a, out=np.zeros_like(c3a), where=c3a != 0))


def pixler_ratios(
    c1: ArrayLike, c2: ArrayLike, c3: ArrayLike, c4: ArrayLike, c5: ArrayLike
) -> dict[str, _Float]:
    """Pixler ratios ``{C1/C2, C1/C3, C1/C4, C1/C5}`` (inf where a denominator is 0)."""
    c1a = np.asarray(c1, np.float64)
    out = {}
    for key, den in ("C1/C2", c2), ("C1/C3", c3), ("C1/C4", c4), ("C1/C5", c5):
        d = np.asarray(den, np.float64)
        out[key] = np.asarray(np.divide(c1a, d, out=np.full_like(d, np.inf), where=d != 0))
    return out


def bernard_ratio(c1: ArrayLike, c2: ArrayLike, c3: ArrayLike) -> _Float:
    """Bernard ratio ``C1/(C2+C3)`` (biogenic vs thermogenic indicator)."""
    den = np.asarray(c2, np.float64) + np.asarray(c3, np.float64)
    return np.asarray(
        np.divide(np.asarray(c1, np.float64), den, out=np.full_like(den, np.inf), where=den != 0)
    )


def apply_eec(comps: ArrayLike, alphas: ArrayLike) -> _Float:
    """Extraction-efficiency correction: divide each component by its ``alpha``."""
    return np.asarray(np.asarray(comps, np.float64) / np.asarray(alphas, np.float64))


def classify_fluid_gor(
    gor_sm3: float, *, thresholds: tuple[float, ...] = (180, 360, 640, 5000, 15000)
) -> str:
    """Fluid type from GOR (Sm3/Sm3) against ascending thresholds.

    Labels: black oil / volatile oil / gas condensate / wet gas / dry gas.
    """
    labels = ["black oil", "volatile oil", "gas condensate", "wet gas", "dry gas"]
    g = float(gor_sm3)
    for thr, label in zip(thresholds, labels, strict=False):
        if g < thr:
            return label
    return labels[-1]


def classify_fluid_wetness(wh_pct: float, bh: float) -> str:
    """Haworth wetness/balance fluid typing (``wh`` in PERCENT)."""
    wh = float(wh_pct)
    if wh < 0.5:
        return "dry gas"
    if wh < 17.5:
        return "gas"
    if wh < 40.0:
        return "gas condensate" if float(bh) > 100.0 else "oil"
    return "residual oil"
