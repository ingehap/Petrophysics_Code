"""Capillary pressure: Young-Laplace/Washburn, curve models, and scaling.

The Young-Laplace/Washburn family alone is re-implemented in ~24 article
files under nine different names, in four unit systems, with both signed and
absolute contact-angle conventions.  This module is strict-SI with explicit
conventions; facades map their article's units and signs.

Hazards this module's API is designed around (LIBRARY_MERGE_PLAN.md
section 9):

- **Brooks-Corey reciprocal exponents.**  ``Sw = Swirr + (1-Swirr)*
  (Pe/Pc)**lam`` in some articles but ``**(1/N)`` in others — the parameters
  are reciprocals under one function name.  The canonical exponent is
  ``lam`` (the pore-size-distribution index); ``1/N``-convention facades
  pass ``lam=1.0/N``.
- **Thomeer log base.**  ``Shg = Bv*exp(-G/log(Pc/Pd))`` uses log10 in some
  articles and the natural log in others — G values differ by ln(10) =
  2.303 and are NOT interchangeable.  ``log_base`` is an explicit parameter.
- **Contact-angle sign.**  Mercury's theta = 140 deg makes cos(theta)
  negative: MICP code needs |cos| while imbibition physics needs the sign
  (a wetting fluid rises, a non-wetting fluid is depressed).  Functions take
  ``absolute=`` explicitly where both conventions exist in the articles.
- **Name traps.**  src2024_12's ``leverett_j_function`` is an empirical
  J(Sw) correlation, not the normalization; src2017_02's ``pc_scaling``
  drops the sigma*cos(theta) factor.  Neither maps onto :func:`leverett_j`.

Units: SI throughout — Pa, N/m, m, kg/m3, Pa*s, rad/s; ``theta_deg`` in
degrees.  The one deliberate exception is :func:`buoyancy_pc_gradient`, the
oilfield psi/ft form, whose 0.433 psi/ft/SG gradient is a visible default.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import G_STD

_Float = NDArray[np.float64]


def _cos_theta(theta_deg: ArrayLike, absolute: bool) -> _Float:
    cos = np.cos(np.radians(np.asarray(theta_deg, np.float64)))
    return np.asarray(np.abs(cos) if absolute else cos)


# --------------------------------------------------------------------------
# Young-Laplace / Washburn
# --------------------------------------------------------------------------


def young_laplace_pc(
    radius: ArrayLike,
    *,
    sigma: ArrayLike,
    theta_deg: ArrayLike = 0.0,
    absolute: bool = True,
) -> _Float:
    """Capillary entry pressure of a tube: ``Pc = 2*sigma*cos(theta)/r``.

    ``theta_deg=0`` gives the pure spherical-meniscus jump ``2*sigma/r``.
    ``absolute=True`` (default) uses |cos| — the MICP convention; pass
    ``absolute=False`` for signed physics.  Sources: src2018_08/article1
    (tutorial, |cos|), src2014_02/article2 and src2020_04/article8 (signed).
    """
    r = np.asarray(radius, np.float64)
    sig = np.asarray(sigma, np.float64)
    return np.asarray(2.0 * sig * _cos_theta(theta_deg, absolute) / r)


def washburn_radius(
    pc: ArrayLike,
    *,
    sigma: ArrayLike,
    theta_deg: ArrayLike = 0.0,
    absolute: bool = True,
) -> _Float:
    """Pore-throat radius from capillary pressure: ``r = 2*sigma*cos(theta)/Pc``.

    Inverse of :func:`young_laplace_pc` (their round trip is a library
    property test).  Sources: src2014_02/article2 (pore_throat_radius,
    signed), src2018_08/article1, src2015_02/article4.
    """
    pc_arr = np.asarray(pc, np.float64)
    sig = np.asarray(sigma, np.float64)
    return np.asarray(2.0 * sig * _cos_theta(theta_deg, absolute) / pc_arr)


def washburn_diameter(
    pc: ArrayLike,
    *,
    sigma: ArrayLike,
    theta_deg: ArrayLike = 0.0,
    absolute: bool = True,
) -> _Float:
    """Pore-throat diameter ``d = 4*sigma*cos(theta)/Pc``.

    The diameter dialect of :func:`washburn_radius` (src2022_10 uses it with
    theta=180; src2018_02's ``-4*cos`` variant is this with the mercury
    angle's sign folded in — its facade passes ``absolute=True``).
    """
    pc_arr = np.asarray(pc, np.float64)
    sig = np.asarray(sigma, np.float64)
    return np.asarray(4.0 * sig * _cos_theta(theta_deg, absolute) / pc_arr)


# --------------------------------------------------------------------------
# Leverett J and system scaling
# --------------------------------------------------------------------------


def leverett_j(
    pc: ArrayLike,
    *,
    sigma: ArrayLike,
    theta_deg: ArrayLike = 0.0,
    k: ArrayLike,
    phi: ArrayLike,
    absolute: bool = True,
) -> _Float:
    """Leverett J-function ``J = Pc/(sigma*cos(theta)) * sqrt(k/phi)``.

    Use consistent units (Pc and sigma as pressure and force/length, k as an
    area, e.g. m2); J is dimensionless.  NOT src2024_12's empirical
    ``leverett_j_function(Sw)`` correlation, and NOT src2017_02's
    ``pc_scaling`` (which drops sigma*cos) — those stay article-local.
    Sources: src2014_02/article2 (signed), src2022_10 (|cos|, theta=180),
    src2016_02/article1 (omits cos entirely — its facade passes theta=0).
    """
    pc_arr = np.asarray(pc, np.float64)
    sig = np.asarray(sigma, np.float64)
    k_arr = np.asarray(k, np.float64)
    porosity = np.asarray(phi, np.float64)
    return np.asarray(pc_arr / (sig * _cos_theta(theta_deg, absolute)) * np.sqrt(k_arr / porosity))


def pc_from_leverett_j(
    j: ArrayLike,
    *,
    sigma: ArrayLike,
    theta_deg: ArrayLike = 0.0,
    k: ArrayLike,
    phi: ArrayLike,
    absolute: bool = True,
) -> _Float:
    """Invert :func:`leverett_j` for capillary pressure.

    Sources: src2016_02/article1 (pc_from_j).
    """
    j_arr = np.asarray(j, np.float64)
    sig = np.asarray(sigma, np.float64)
    k_arr = np.asarray(k, np.float64)
    porosity = np.asarray(phi, np.float64)
    return np.asarray(j_arr * sig * _cos_theta(theta_deg, absolute) / np.sqrt(k_arr / porosity))


def pc_convert_system(
    pc: ArrayLike,
    *,
    sigma_from: ArrayLike,
    theta_from_deg: ArrayLike,
    sigma_to: ArrayLike,
    theta_to_deg: ArrayLike,
    absolute: bool = True,
) -> _Float:
    """Convert Pc between fluid systems by the sigma*cos(theta) ratio.

    The lab-to-reservoir conversion (air-mercury -> gas-brine etc.);
    ``absolute=True`` (the src2018_10/src2018_02 convention) keeps mercury
    conversions positive; src2016_10 uses signed cos.  Sources:
    src2018_10/article1, src2016_10/article2, src2018_02 (rescale_ift).
    """
    pc_arr = np.asarray(pc, np.float64)
    num = np.asarray(sigma_to, np.float64) * _cos_theta(theta_to_deg, absolute)
    den = np.asarray(sigma_from, np.float64) * _cos_theta(theta_from_deg, absolute)
    return np.asarray(pc_arr * num / den)


# --------------------------------------------------------------------------
# Curve models
# --------------------------------------------------------------------------


def brooks_corey_sw(
    pc: ArrayLike,
    *,
    pc_entry: ArrayLike,
    lam: float,
    swirr: float,
    clip: tuple[float, float] | None = None,
) -> _Float:
    """Brooks-Corey drainage saturation ``Sw = Swirr + (1-Swirr)*(Pe/Pc)**lam``.

    Returns 1.0 at and below the entry pressure.  ``lam`` is the
    pore-size-distribution index — articles using the reciprocal ``(1/N)``
    convention pass ``lam=1.0/N`` (section 9 hazard; the parameters are NOT
    interchangeable).  Sources: src2018_06/article8, src2018_12,
    src2019_06/article8 (lam form); src2016_06/article1 (1/N form).
    """
    pc_arr = np.asarray(pc, np.float64)
    pe = np.asarray(pc_entry, np.float64)
    sw = swirr + (1.0 - swirr) * (pe / pc_arr) ** lam
    result = np.asarray(np.where(pc_arr <= pe, 1.0, sw))
    if clip is None:
        return result
    return np.asarray(np.clip(result, clip[0], clip[1]))


def brooks_corey_pc(
    sw: ArrayLike,
    *,
    pc_entry: ArrayLike,
    lam: float,
    swirr: float,
    snr: float = 0.0,
) -> _Float:
    """Brooks-Corey capillary pressure ``Pc = Pe * Swn**(-1/lam)``.

    ``Swn = (Sw - Swirr)/(1 - Swirr - Snr)`` — the normalized-saturation
    window is explicit (``snr`` defaults to 0, the drainage convention; the
    residual-nonwetting window some articles use must be passed, never
    assumed).  Sources: src2020_04/article7, src2022_10/article2,
    src2023_08, src2025_02.
    """
    sw_arr = np.asarray(sw, np.float64)
    pe = np.asarray(pc_entry, np.float64)
    swn = (sw_arr - swirr) / (1.0 - swirr - snr)
    return np.asarray(pe * swn ** (-1.0 / lam))


def thomeer_shg(
    pc: ArrayLike,
    *,
    bv_inf: float,
    g: float,
    pd: float,
    log_base: float = 10.0,
) -> _Float:
    """Thomeer hyperbola ``Shg = Bv_inf * exp(-G / log_base(Pc/Pd))``; 0 below Pd.

    ``log_base`` is explicit because the articles disagree — src2021_04 and
    src2016_10 use log10, src2021_08 uses the natural log — and G values
    differ by ln(10) = 2.303 between the conventions (section 9 hazard).
    Sources: src2021_04/article3, src2021_08/article1, src2016_10.
    """
    pc_arr = np.asarray(pc, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(pc_arr / pd) / np.log(log_base)
        shg = bv_inf * np.exp(-g / log_ratio)
    return np.asarray(np.where(pc_arr > pd, shg, 0.0))


# --------------------------------------------------------------------------
# Buoyancy / saturation-height kernels
# --------------------------------------------------------------------------


def buoyancy_pc(height: ArrayLike, *, delta_rho: ArrayLike, g: float = G_STD) -> _Float:
    """Buoyancy capillary pressure ``Pc = delta_rho * g * h`` (SI: Pa).

    ``delta_rho`` is the water-hydrocarbon density difference (kg/m3), ``h``
    the height above the free-water level (m).  Sources:
    src2019_06/article8, src2018_08/article1 (capillary_pressure_from_rise,
    g = 9.81 at the facade), src2023_10.
    """
    h = np.asarray(height, np.float64)
    drho = np.asarray(delta_rho, np.float64)
    return np.asarray(drho * g * h)


def height_above_fwl(pc: ArrayLike, *, delta_rho: ArrayLike, g: float = G_STD) -> _Float:
    """Height above the free-water level from Pc: inverse of :func:`buoyancy_pc`."""
    pc_arr = np.asarray(pc, np.float64)
    drho = np.asarray(delta_rho, np.float64)
    return np.asarray(pc_arr / (drho * g))


def buoyancy_pc_gradient(
    height_ft: ArrayLike,
    *,
    sg_water: ArrayLike,
    sg_hc: ArrayLike,
    gradient_psi_per_ft: float = 0.433,
) -> _Float:
    """Oilfield buoyancy Pc: ``Pc = gradient*(SGw - SGhc)*h`` (psi, ft, SG).

    The 0.433 psi/ft/SG freshwater gradient is the conventional rounded
    field constant, kept visible as a parameter (CONVENTIONS.md rule 5).
    Sources: src2016_06/article1, src2016_10/article2.
    """
    h = np.asarray(height_ft, np.float64)
    sg_w = np.asarray(sg_water, np.float64)
    sg_h = np.asarray(sg_hc, np.float64)
    return np.asarray(gradient_psi_per_ft * (sg_w - sg_h) * h)


# --------------------------------------------------------------------------
# Laboratory kernels
# --------------------------------------------------------------------------


def centrifuge_pc(
    omega: ArrayLike,
    *,
    delta_rho: ArrayLike,
    r1: ArrayLike,
    r2: ArrayLike,
) -> _Float:
    """Hassler-Brunner inlet-face Pc: ``0.5*delta_rho*omega**2*(r2**2 - r1**2)``.

    ``omega`` in rad/s (rpm-input articles convert in their facades:
    ``omega = rpm*2*pi/60``); radii in m from the rotation axis.  Sources:
    src2017_08/article3, src2023_06, src2025_12 (rad/s); src2020_06,
    src2020_08 (rpm at the facade).
    """
    w = np.asarray(omega, np.float64)
    drho = np.asarray(delta_rho, np.float64)
    r1_arr = np.asarray(r1, np.float64)
    r2_arr = np.asarray(r2, np.float64)
    return np.asarray(0.5 * drho * w**2 * (r2_arr**2 - r1_arr**2))


def capillary_rise_height(
    radius: ArrayLike,
    *,
    sigma: ArrayLike,
    theta_deg: ArrayLike = 0.0,
    delta_rho: ArrayLike,
    g: float = G_STD,
) -> _Float:
    """Equilibrium capillary rise ``h = 2*sigma*cos(theta)/(delta_rho*g*r)``.

    Signed by physics: a wetting fluid (theta < 90 deg) rises, a non-wetting
    fluid (mercury) is depressed (h < 0).  Sources: src2018_08/article1
    (g = 9.81 at the facade).
    """
    r = np.asarray(radius, np.float64)
    sig = np.asarray(sigma, np.float64)
    drho = np.asarray(delta_rho, np.float64)
    return np.asarray(2.0 * sig * _cos_theta(theta_deg, absolute=False) / (drho * g * r))


def lucas_washburn_length(
    t: ArrayLike,
    *,
    sigma: ArrayLike,
    radius: ArrayLike,
    theta_deg: ArrayLike = 0.0,
    mu: ArrayLike,
) -> _Float:
    """Lucas-Washburn imbibition front ``L(t) = sqrt(sigma*r*cos(theta)*t/(2*mu))``.

    Signed cos: an oil-wet system (cos < 0) yields NaN rather than a
    fabricated zero — src2020_04's clip-to-zero behavior lives in its
    facade.  Sources: src2020_04/article8, src2025_12/nanopore_adsorption.
    """
    t_arr = np.asarray(t, np.float64)
    sig = np.asarray(sigma, np.float64)
    r = np.asarray(radius, np.float64)
    mu_arr = np.asarray(mu, np.float64)
    with np.errstate(invalid="ignore"):
        return np.asarray(
            np.sqrt(sig * r * _cos_theta(theta_deg, absolute=False) * t_arr / (2.0 * mu_arr))
        )
