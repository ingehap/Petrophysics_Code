"""Wellbore-survey trajectory geometry: dogleg, minimum curvature, MD to TVD.

Directional-survey math shared by the geosteering / well-positioning articles:
the dogleg severity between two stations, the minimum-curvature ratio (radius)
factor, one balanced min-curvature step, a cumulative trajectory, and the
MD -> TVD conversion in either the balanced minimum-curvature or the legacy
single-inclination tangential convention.

Inclination and azimuth are always taken in **degrees**; the dogleg is returned
in **radians** (that is the form the ratio factor consumes).  The horizontal
axes are North/East and TVD is positive downward, so a trajectory column order
is ``(TVD, North, East)`` throughout.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2019_06/article9_wellbore_positioning_lwd -- Article 9: Wellbore Positioning While Drilling With
  LWD Measurements. Poedjono, Nwosu, Martin (2019). DOI: 10.30632/PJV60N3-2019a8. Petrophysics Vol.
  60 No. 3 (Jun 2019).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


def dogleg_angle(inc1: float, azi1: float, inc2: float, azi2: float) -> float:
    """Dogleg (total curvature) angle between two survey stations, in radians.

    ``cos(DL) = cos(i2 - i1) - sin(i1) sin(i2) (1 - cos(a2 - a1))`` with the
    inclinations ``i`` and azimuths ``a`` supplied in degrees.  This is the
    numerically well-behaved rearrangement of the spherical law of cosines and
    is clipped to ``[-1, 1]`` before the ``arccos``.

    Sources: src2019_06/article9_wellbore_positioning_lwd.
    """
    i1, a1, i2, a2 = (np.radians(x) for x in (inc1, azi1, inc2, azi2))
    cos_dl = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1))
    return float(np.arccos(np.clip(cos_dl, -1.0, 1.0)))


def ratio_factor(dogleg_rad: float) -> float:
    """Minimum-curvature ratio (radius) factor ``RF = (2/DL) tan(DL/2)``.

    ``dogleg_rad`` is the dogleg angle in radians; for a near-straight step
    (``DL < 1e-9``) the factor is exactly ``1.0`` (the balanced-tangential
    limit).

    Sources: src2019_06/article9_wellbore_positioning_lwd.
    """
    if dogleg_rad < 1e-9:
        return 1.0
    return float((2.0 / dogleg_rad) * np.tan(dogleg_rad / 2.0))


def minimum_curvature_step(
    md1: float,
    inc1: float,
    azi1: float,
    md2: float,
    inc2: float,
    azi2: float,
) -> tuple[float, float, float]:
    """One minimum-curvature step, returning ``(dTVD, dNorth, dEast)``.

    ``half = (dMD/2) * RF`` scales the balanced-tangential increments
    ``dN = half (sin i1 cos a1 + sin i2 cos a2)``,
    ``dE = half (sin i1 sin a1 + sin i2 sin a2)``,
    ``dTVD = half (cos i1 + cos i2)`` with ``RF`` the :func:`ratio_factor` of the
    :func:`dogleg_angle`.  Angles in degrees, lengths in the units of ``md``.

    Sources: src2019_06/article9_wellbore_positioning_lwd.
    """
    dmd = md2 - md1
    dl = dogleg_angle(inc1, azi1, inc2, azi2)
    rf = ratio_factor(dl)
    i1, a1, i2, a2 = (np.radians(x) for x in (inc1, azi1, inc2, azi2))
    half = dmd / 2.0 * rf
    d_north = half * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2))
    d_east = half * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2))
    d_tvd = half * (np.cos(i1) + np.cos(i2))
    return float(d_tvd), float(d_north), float(d_east)


def survey_to_path(md: ArrayLike, inc: ArrayLike, azi: ArrayLike) -> _Float:
    """Cumulative minimum-curvature trajectory, shape ``(n, 3)``.

    Integrates :func:`minimum_curvature_step` from the origin ``(0, 0, 0)`` over
    consecutive stations of a survey given as parallel ``md``/``inc``/``azi``
    arrays (degrees).  Column order is ``(TVD, North, East)``.

    Sources: src2019_06/article9_wellbore_positioning_lwd.
    """
    md_a = _arr(md)
    inc_a = _arr(inc)
    azi_a = _arr(azi)
    n = md_a.size
    path = np.zeros((n, 3))
    for k in range(1, n):
        d_tvd, d_north, d_east = minimum_curvature_step(
            md_a[k - 1], inc_a[k - 1], azi_a[k - 1], md_a[k], inc_a[k], azi_a[k]
        )
        path[k] = path[k - 1] + (d_tvd, d_north, d_east)
    return path


def md_to_tvd(md: ArrayLike, inc_deg: ArrayLike, *, method: str = "min_curvature") -> _Float:
    """Cumulative true vertical depth from measured depth and inclination.

    ``method='min_curvature'`` uses the balanced minimum-curvature vertical
    increments (azimuth-independent, so any constant azimuth is used
    internally); ``method='tangential'`` reproduces the legacy single-
    inclination form ``TVD_k = sum_{j<=k} dMD_j cos(inc_j)`` (each segment takes
    the inclination at its lower station).  Returns a cumulative TVD array the
    same length as ``md``, starting at ``md[0] * cos(inc[0])`` for tangential and
    at ``0`` increment for the first sample.
    """
    md_a = _arr(md)
    inc_a = _arr(inc_deg)
    if method == "tangential":
        dmd = np.diff(md_a, prepend=md_a[0])
        return np.asarray(np.cumsum(dmd * np.cos(np.radians(inc_a))))
    if method == "min_curvature":
        azi = np.zeros_like(md_a)
        return np.asarray(survey_to_path(md_a, inc_a, azi)[:, 0])
    raise ValueError(f"unknown method {method!r}; use 'min_curvature' or 'tangential'")
