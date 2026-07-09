"""Relative permeability and displacement: Corey/Brooks-Corey/LET, mobility,
fractional flow, and the Buckley-Leverett/Welge construction.

The Corey relative-permeability pair ``kr = kr_end * Se**n`` is re-implemented
in ~21 article files under many names (``corey_krw``, ``corey_relperm``,
``krw_kro_brooks_corey``, ``relative_permeability_corey`` ...), each with its
own saturation normalization and clip policy.  This module is the one
canonical home; facades map their article's endpoints and exponents.

Hazards this module's API is designed around (LIBRARY_MERGE_PLAN.md
section 9):

- **Se-denominator convention.**  The normalized saturation is
  ``Se = (S - Sr)/(1 - Sr - Snr)`` — but articles subtract different
  residuals: ``(Sw-Swc)/(1-Swc-Sor)`` (standard), ``(Sw-Swc)/(1-Swc)``
  (drainage only), and the 4-endpoint gas form ``(Sg-Sgc)/(1-Swc-Sgc-Sorg)``.
  :func:`normalized_saturation` takes ``snr`` and ``clip`` explicitly.
- **Oil normalization.**  Most files write ``kro = kro_max*(1-Se)**no``; a few
  normalize oil independently as ``(1-Sw-Sor)/(1-Swir-Sor)``.  These are
  algebraically identical for a shared endpoint set (they agree to ~1 ULP),
  so :func:`corey_kro` uses the ``(1-Se)`` form.
- **Name != family.**  ``krg_brooks_corey``, ``krw_kro_brooks_corey``,
  ``brooks_corey_kr`` are all *free-exponent* Corey (:func:`corey_kr`), not
  the pore-size-index Burdine form.  The true fixed ``(2+3*lam)/lam``
  exponents (src2020_04/article7, src2020_06/article4) are
  :func:`brooks_corey_burdine_kr`.
- **Clip policy.**  Articles variously clip Se to [0,1], clip Sw instead,
  clip the output, or never clip.  ``clip=`` is explicit (default
  ``(0.0, 1.0)``, the mobile-window bound); pass ``clip=None`` to disable.

Saturations and relative permeabilities are fractions; viscosities in Pa*s
(any consistent unit cancels in ratios).  numpy-only at import; :func:`fit_corey`
imports scipy lazily.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]

_TINY = 1e-30  # zero-mobility / zero-denominator guard


def normalized_saturation(
    s: ArrayLike,
    sr: float,
    snr: float = 0.0,
    *,
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> _Float:
    """Normalized (effective) saturation ``Se = (S - Sr)/(1 - Sr - Snr)``.

    ``snr=0`` gives the ``(1-Sr)`` drainage convention; the residual of the
    other phase (Sor, Snwr, Sorg) goes in ``snr``.  ``clip=(0.0, 1.0)`` bounds
    Se to the mobile window (the common Corey behavior); ``clip=None`` disables
    it (some articles leave Se unclipped).  Sources: src2018_04/article9,
    src2016_06/article3, src2020_04/article7 (effective_saturation).
    """
    s_arr = np.asarray(s, np.float64)
    se = (s_arr - sr) / (1.0 - sr - snr)
    if clip is not None:
        se = np.clip(se, clip[0], clip[1])
    return np.asarray(se)


def corey_krw(
    sw: ArrayLike,
    *,
    swr: float,
    sor: float = 0.0,
    krw_max: float = 1.0,
    nw: float = 2.0,
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> _Float:
    """Corey water relative permeability ``krw = krw_max * Se**nw``.

    ``Se = (Sw - Swr)/(1 - Swr - Sor)``.  Sources: src2014_08/article1,
    src2016_02/article1, src2017_02, src2018_04/article9, src2024_10.
    """
    se = normalized_saturation(sw, swr, sor, clip=clip)
    return np.asarray(krw_max * se**nw)


def corey_kro(
    sw: ArrayLike,
    *,
    swr: float,
    sor: float = 0.0,
    kro_max: float = 1.0,
    no: float = 2.0,
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> _Float:
    """Corey oil relative permeability ``kro = kro_max * (1 - Se)**no``.

    Uses the ``(1-Se)`` form on the water-normalized Se; articles that
    normalize oil independently as ``(1-Sw-Sor)/(1-Swir-Sor)`` agree to ~1 ULP.
    Sources: src2014_08/article1, src2016_02, src2017_02/article5, src2024_10.
    """
    se = normalized_saturation(sw, swr, sor, clip=clip)
    return np.asarray(kro_max * (1.0 - se) ** no)


def corey_kr(
    sw: ArrayLike,
    *,
    swr: float,
    sor: float = 0.0,
    krw_max: float = 1.0,
    kro_max: float = 1.0,
    nw: float = 2.0,
    no: float = 2.0,
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> tuple[_Float, _Float]:
    """Two-phase Corey pair ``(krw, kro)`` on a shared Se.

    The single canonical replacement for the ``corey_kr`` / ``corey_relperm``
    / ``krw``+``kro`` pair variants.  Sources: src2017_02/article5,
    src2018_02/article5, src2022_10/article4, src2024_04, src2025_02.
    """
    se = normalized_saturation(sw, swr, sor, clip=clip)
    krw = krw_max * se**nw
    kro = kro_max * (1.0 - se) ** no
    return np.asarray(krw), np.asarray(kro)


def corey_krg(
    sg: ArrayLike,
    *,
    sgc: float = 0.0,
    swc: float = 0.0,
    sorg: float = 0.0,
    krg_max: float = 1.0,
    ng: float = 2.0,
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> _Float:
    """Corey gas relative permeability ``krg = krg_max * Sg*_**ng``.

    ``Sg* = (Sg - Sgc)/(1 - Swc - Sgc - Sorg)`` — the 4-endpoint gas
    denominator.  Sw-framework gas curves (``(1-Sw-Sgc)/(1-Swr-Sgc)``) map by
    passing ``sg=1-sw`` and ``swc=swr`` (bit-exact).  Sources:
    src2019_04/article11, src2020_04/article6, src2014_08/article1 (gas).
    """
    sg_arr = np.asarray(sg, np.float64)
    sge = (sg_arr - sgc) / (1.0 - swc - sgc - sorg)
    if clip is not None:
        sge = np.clip(sge, clip[0], clip[1])
    return np.asarray(krg_max * sge**ng)


def brooks_corey_burdine_kr(
    sw: ArrayLike,
    *,
    swr: float,
    lam: float,
    snwr: float = 0.0,
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> tuple[_Float, _Float]:
    """Brooks-Corey (Burdine) kr pair from the pore-size-distribution index.

    ``krw = Se**((2+3*lam)/lam)``,
    ``krnw = (1-Se)**2 * (1 - Se**((2+lam)/lam))`` — the fixed-exponent form
    (no endpoint factor; kr_max == 1).  This is NOT free-exponent Corey.
    Sources: src2020_04/article7, src2020_06/article4.
    """
    se = normalized_saturation(sw, swr, snwr, clip=clip)
    krw = se ** ((2.0 + 3.0 * lam) / lam)
    krnw = (1.0 - se) ** 2 * (1.0 - se ** ((2.0 + lam) / lam))
    return np.asarray(krw), np.asarray(krnw)


def let_kr(
    sw: ArrayLike,
    *,
    swr: float,
    L: float,
    E: float,
    T: float,
    kr_max: float = 1.0,
    phase: str = "wetting",
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> _Float:
    """Lomeland-Ebeltoft-Thomas (LET) relative permeability.

    ``phase="wetting"``:  ``kr = kr_max * Se**L / (Se**L + E*(1-Se)**T)``.
    ``phase="nonwetting"``: ``kr = kr_max * (1-Se)**L / ((1-Se)**L + E*Se**T)``.
    ``Se = (Sw - Swr)/(1 - Swr)`` (LET articles use the drainage denominator);
    the denominator is zero-guarded.  Call once per phase.  Sources:
    src2025_02/scal_model_ccs, src2025_02/enhanced_gas_recovery (gas).
    """
    se = normalized_saturation(sw, swr, 0.0, clip=clip)
    if phase == "wetting":
        num = se**L
        den = num + E * (1.0 - se) ** T
    elif phase == "nonwetting":
        num = (1.0 - se) ** L
        den = num + E * se**T
    else:
        raise ValueError(f"phase must be 'wetting' or 'nonwetting', got {phase!r}")
    den = np.where(den == 0.0, _TINY, den)
    return np.asarray(kr_max * num / den)


def phase_mobility(kr: ArrayLike, mu: float) -> _Float:
    """Phase mobility ``lambda = kr / mu``.  Sources: src2025_12/analog_kr."""
    return np.asarray(np.asarray(kr, np.float64) / mu)


def total_mobility(
    kr_w: ArrayLike,
    kr_nw: ArrayLike,
    *,
    mu_w: float,
    mu_nw: float,
) -> _Float:
    """Total two-phase mobility ``lambda_t = kr_w/mu_w + kr_nw/mu_nw``.

    Sources: src2025_12/analog_kr (total_mobility).
    """
    return np.asarray(phase_mobility(kr_w, mu_w) + phase_mobility(kr_nw, mu_nw))


def fractional_flow(
    kr_w: ArrayLike,
    kr_nw: ArrayLike,
    *,
    mu_w: float,
    mu_nw: float,
) -> _Float:
    """Buckley-Leverett fractional flow of the wetting phase (zero-safe).

    ``fw = (kr_w/mu_w) / (kr_w/mu_w + kr_nw/mu_nw)``; returns 0 where the total
    mobility is 0 (no gravity/capillary term — none of the article fw kernels
    include one).  Takes kr *values*; use :func:`fractional_flow_curve` from Sw.
    Sources: src2025_12/analog_kr, src2016_02/article4, src2019_02/article1.
    """
    lam_w = np.asarray(kr_w, np.float64) / mu_w
    lam_nw = np.asarray(kr_nw, np.float64) / mu_nw
    denom = lam_w + lam_nw
    return np.asarray(np.where(denom > 0.0, lam_w / denom, 0.0))


def fractional_flow_curve(
    *,
    swr: float,
    sor: float = 0.0,
    krw_max: float = 1.0,
    kro_max: float = 1.0,
    nw: float = 2.0,
    no: float = 2.0,
    mu_w: float,
    mu_nw: float,
    n: int = 2000,
) -> tuple[_Float, _Float]:
    """Corey fractional-flow curve ``(sw, fw)`` over the mobile window.

    Convenience wrapper: builds ``sw = linspace(swr, 1-sor, n)``, evaluates the
    Corey pair, and returns the wetting-phase fractional flow.  Sources:
    src2016_02/article4 (welge_shock_front builds the same grid).
    """
    sw = np.linspace(swr, 1.0 - sor, n)
    krw, kro = corey_kr(sw, swr=swr, sor=sor, krw_max=krw_max, kro_max=kro_max, nw=nw, no=no)
    fw = fractional_flow(krw, kro, mu_w=mu_w, mu_nw=mu_nw)
    return np.asarray(sw), fw


def welge_shock(
    sw: ArrayLike,
    fw: ArrayLike,
    swc: float,
) -> tuple[float, float, float]:
    """Welge tangent construction: ``(swf, fwf, sw_avg)``.

    The Buckley-Leverett shock front is the tangent to ``fw(Sw)`` drawn from
    ``(Swc, 0)``; found as the maximum secant slope ``fw/(Sw - Swc)`` over the
    interior.  ``sw_avg = swf + (1 - fwf)/slope`` is the average saturation
    behind the front at breakthrough.  Sources: src2016_02/article4
    (welge_shock_front), src2022_10/article4 (welge_tangent, front only).
    """
    sw_arr = np.asarray(sw, np.float64)
    fw_arr = np.asarray(fw, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        secant = np.where(sw_arr > swc, fw_arr / (sw_arr - swc), -np.inf)
    i = int(np.argmax(secant))
    swf = float(sw_arr[i])
    fwf = float(fw_arr[i])
    slope = float(secant[i])
    sw_avg = swf + (1.0 - fwf) / slope
    return swf, fwf, sw_avg


def endpoint_mobility_ratio(
    krw_max: float,
    kro_max: float,
    *,
    mu_w: float,
    mu_o: float,
) -> float:
    """Endpoint mobility ratio ``M = (krw_max/mu_w) / (kro_max/mu_o)``.

    The waterflood favorability index (M < 1 is favorable / piston-like).
    Sources: src2018_04/article9, src2019_08/article6 (mobility_ratio).
    """
    return (krw_max / mu_w) / (kro_max / mu_o)


def fit_corey(
    sw: ArrayLike,
    krw_obs: ArrayLike,
    krnw_obs: ArrayLike,
    *,
    swr: float,
    sor: float = 0.0,
) -> dict[str, float]:
    """Least-squares Corey endpoints and exponents from observed kr.

    Fits ``krw = krw_max*Se**nw`` and ``krnw = krnw_max*(1-Se)**nnw`` on
    ``Se = (Sw-Swr)/(1-Swr-Sor)`` and returns
    ``{swr, sor, krw_max, nw, krnw_max, nnw}``.  Requires scipy (imported
    lazily).  Sources: src2025_02/co2_brine_relperm (fit_corey).
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fit_corey requires scipy; install petrolib[scipy]") from exc

    se = normalized_saturation(sw, swr, sor, clip=(1e-6, 1.0 - 1e-6))
    krw_arr = np.asarray(krw_obs, np.float64)
    krnw_arr = np.asarray(krnw_obs, np.float64)
    (krw_max, nw), _ = curve_fit(lambda s, a, b: a * s**b, se, krw_arr, p0=(1.0, 2.0))
    (krnw_max, nnw), _ = curve_fit(lambda s, a, b: a * (1.0 - s) ** b, se, krnw_arr, p0=(1.0, 2.0))
    return {
        "swr": swr,
        "sor": sor,
        "krw_max": float(krw_max),
        "nw": float(nw),
        "krnw_max": float(krnw_max),
        "nnw": float(nnw),
    }
