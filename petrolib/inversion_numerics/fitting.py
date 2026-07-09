"""Curve fitting: line / power-law / exponential-approach / cosine.

Small regression helpers that recur across the corpus: a transform-aware line fit
returning slope/intercept/R^2, power-law-decay and exponential-relaxation fits,
and the azimuthal cosine-harmonic fit.  scipy (``curve_fit``) is imported lazily
only where a nonlinear fit is genuinely needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


class LineFit(NamedTuple):
    """Result of :func:`fit_line` (in the transformed space)."""

    slope: float
    intercept: float
    r2: float


_XFORMS: dict[str | None, Callable[[_Float], _Float]] = {
    None: lambda v: v,
    "log10": np.log10,
    "log": np.log,
    "inv": lambda v: 1.0 / v,
    "sqrt": np.sqrt,
    "square": np.square,
}


def _transform(name: str | None, v: _Float) -> _Float:
    try:
        return np.asarray(_XFORMS[name](v))
    except KeyError:
        raise ValueError(
            f"unknown transform {name!r}; use {sorted(k for k in _XFORMS if k)}"
        ) from None


def fit_line(
    x: ArrayLike, y: ArrayLike, *, xform: str | None = None, yform: str | None = None
) -> LineFit:
    """Degree-1 least-squares fit ``y = slope*x + intercept`` with R^2.

    ``xform`` / ``yform`` optionally transform the axes first (e.g. ``'log10'``
    for a log-log or semilog fit); slope/intercept/R^2 are reported in that
    transformed space.
    """
    xt = _transform(xform, _arr(x))
    yt = _transform(yform, _arr(y))
    slope, intercept = np.polyfit(xt, yt, 1)
    yhat = slope * xt + intercept
    ss_res = float(np.sum((yt - yhat) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return LineFit(float(slope), float(intercept), float(r2))


def fit_powerlaw_decay(
    x: ArrayLike, y: ArrayLike, exponent: float | None = None
) -> tuple[float, float]:
    """Fit ``y = a * x**(-b)`` -> ``(a, b)``.

    With ``exponent`` given (``b`` known) the amplitude ``a`` is recovered by a
    linear least-squares projection; otherwise both are fit by ``curve_fit``.
    """
    x_arr = _arr(x)
    y_arr = _arr(y)
    if exponent is not None:
        basis = x_arr ** (-exponent)
        a = float(np.dot(basis, y_arr) / np.dot(basis, basis))
        return a, float(exponent)
    try:
        from scipy.optimize import curve_fit
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fit_powerlaw_decay without exponent requires scipy") from exc

    def model(xx: _Float, a: float, b: float) -> _Float:
        return np.asarray(a * xx ** (-b))

    popt, _ = curve_fit(model, x_arr, y_arr, p0=[float(y_arr[0]), 1.0], maxfev=10000)
    return float(popt[0]), float(popt[1])


def fit_exponential_approach(
    t: ArrayLike, y: ArrayLike, three_point: bool = False
) -> tuple[float, float]:
    """Fit an exponential approach ``y = asymptote + (y0-asymptote)*exp(-t/tau)``.

    Returns ``(asymptote, tau)``.  ``three_point=True`` uses the closed-form
    estimate from the first / middle / last (equally spaced) samples; otherwise
    ``curve_fit`` is used.
    """
    t_arr = _arr(t)
    y_arr = _arr(y)
    if three_point:
        mid = t_arr.size // 2
        y1, y2, y3 = float(y_arr[0]), float(y_arr[mid]), float(y_arr[-1])
        asymptote = (y1 * y3 - y2**2) / (y1 + y3 - 2.0 * y2)
        r = (y3 - y2) / (y2 - y1)
        tau = -(t_arr[mid] - t_arr[0]) / np.log(r)
        return float(asymptote), float(tau)
    try:
        from scipy.optimize import curve_fit
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fit_exponential_approach without three_point requires scipy") from exc

    def model(tt: _Float, asy: float, y0: float, tau: float) -> _Float:
        return np.asarray(asy + (y0 - asy) * np.exp(-tt / tau))

    span = float(t_arr[-1] - t_arr[0])
    popt, _ = curve_fit(
        model, t_arr, y_arr, p0=[float(y_arr[-1]), float(y_arr[0]), span / 3.0], maxfev=10000
    )
    return float(popt[0]), float(popt[2])


def fit_cosine(az: ArrayLike, y: ArrayLike) -> tuple[float, float, float]:
    """Azimuthal cosine-harmonic fit ``y = mean + amp*cos(az - phase)``.

    Least squares on ``[1, cos(az), sin(az)]`` (``az`` in degrees).  Returns
    ``(mean, amplitude, phase_deg)``.
    """
    theta = np.radians(_arr(az))
    g = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    coef, *_ = np.linalg.lstsq(g, _arr(y), rcond=None)
    mean, a, b = float(coef[0]), float(coef[1]), float(coef[2])
    amplitude = float(np.hypot(a, b))
    phase = float(np.degrees(np.arctan2(b, a)))
    return mean, amplitude, phase
