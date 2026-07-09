"""Porosity, shale volume, lithology mixing, Thomas-Stieber, TOC and net pay.

The single most re-implemented petrophysics domain in the corpus: density
porosity ``(rho_ma-rho_b)/(rho_ma-rho_fl)`` alone appears in ~18 files, the
gamma-ray shale index in ~17, and the Timur-Coates/Archie neighbours in dozens
more.  This module is the canonical home; facades keep their article's matrix
and fluid endpoints, cutoffs, and clip behaviour.

Hazards this module's API is designed around (LIBRARY_MERGE_PLAN.md
section 9), all confirmed by a three-era corpus survey:

- **No baked-in matrix/fluid endpoints.**  ``density_porosity`` appears with
  rho_ma in {2.65, 2.67, 2.71, 2.85} and rho_fl in {1.0, 1.01, 1.025, 1.03};
  the endpoints are explicit arguments, never invisible field defaults.
- **Argument-order collisions.**  ``(BV-GV)/BV`` porosity is written both
  ``helium_porosity(bv, gv)`` and ``porosity(grain, bulk)`` — operands
  reversed; density porosity is written ``(rho_b, rho_ma, rho_fl)`` and
  ``total_porosity(rho_ma, rho_b, rho_fl)``.  Data arrays are positional,
  endpoints keyword-only where the swap is silent.
- **Clip is opt-in.**  Historic copies clip Vsh/porosity to [0,1], [0,0.6],
  [-0.05,1] or not at all; the canonical default is unclipped with an explicit
  ``clip=`` (CONVENTIONS.md rule 3), and facades restore each local clip.
- **Two mixing laws under one name.**  Grain density is both the arithmetic
  volume-weighted ``sum(v_i*rho_i)`` and the harmonic mass-weighted
  ``1/sum(w_i/rho_i)`` (with an optional kerogen term); they are distinct
  functions here.
- **Vsh method family.**  linear / Larionov (tertiary & older) / Clavier /
  Steiber / generalised-Stieber share the GR index but differ in transform;
  ``method=`` selects, and each takes the raw GR + endpoints so the IGR-input
  copies map by passing ``gr_clean=0, gr_shale=1``.

Units follow the corpus convention: densities g/cc, porosity/volumes/
saturations v/v fractions, GR in API, resistivity ohm-m, TOC in the unit each
function's docstring states.  numpy-broadcastable; scipy is imported lazily
only by the nnls mineral solver.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]

# A clip range ``(lo, hi)``; either bound may be ``None`` (unbounded on that
# side), or the whole thing ``None`` for no clipping (CONVENTIONS.md rule 3).
_Clip = tuple[float | None, float | None] | None


def _clip(values: _Float, clip: _Clip) -> _Float:
    if clip is None:
        return values
    return np.asarray(np.clip(values, clip[0], clip[1]))


# ==========================================================================
# 1. Shale / clay volume
# ==========================================================================


def gamma_ray_index(
    gr: ArrayLike,
    gr_clean: ArrayLike,
    gr_shale: ArrayLike,
    *,
    clip: _Clip = (0.0, 1.0),
) -> _Float:
    """Gamma-ray shale index ``IGR = (GR-GR_clean)/(GR_shale-GR_clean)``.

    The clean and shale endpoints carry any GR unit (API/GAPI); the index is
    dimensionless and clipped to [0,1] by default.  Sources: src2015_06/article1,
    src2016_10/article3 (gr_sand naming), src2023_08/article1.
    """
    gr_a = np.asarray(gr, np.float64)
    lo = np.asarray(gr_clean, np.float64)
    hi = np.asarray(gr_shale, np.float64)
    return _clip(np.asarray((gr_a - lo) / (hi - lo)), clip)


def vshale_from_gr(
    gr: ArrayLike,
    gr_clean: ArrayLike,
    gr_shale: ArrayLike,
    *,
    method: str = "linear",
    gcur: float = 2.0,
    clip: _Clip = None,
) -> _Float:
    """Shale volume from the gamma-ray index by one of several transforms.

    ``method``: ``"linear"`` (Vsh=IGR), ``"larionov_tertiary"``
    ``0.083*(2**(3.7*IGR)-1)``, ``"larionov_older"`` ``0.33*(2**(2*IGR)-1)``,
    ``"clavier"`` ``1.7-sqrt(3.38-(IGR+0.7)**2)``, ``"steiber"``
    ``0.5*IGR/(1.5-IGR)``, or ``"stieber_gcur"`` the normalised
    ``(2**(gcur*IGR)-1)/(2**gcur-1)``.  The IGR is computed and clipped to
    [0,1] internally, so copies that pass an already-computed IGR call this
    with ``gr_clean=0, gr_shale=1``.  ``clip`` applies to the final Vsh (opt-in;
    the corpus Larionov copies leave it unclipped, Clavier clips [0,1]).
    Sources: src2021_12/article09 (larionov_older), src2023_08/article1
    (larionov tertiary/older), src2025_10/a6, src2015_06/article1 (clavier),
    src2025_10/a3 (steiber), src2025_06/toc_prediction (stieber_gcur).
    """
    igr = gamma_ray_index(gr, gr_clean, gr_shale, clip=(0.0, 1.0))
    if method == "linear":
        vsh = igr
    elif method == "larionov_tertiary":
        vsh = 0.083 * (2.0 ** (3.7 * igr) - 1.0)
    elif method == "larionov_older":
        vsh = 0.33 * (2.0 ** (2.0 * igr) - 1.0)
    elif method == "clavier":
        vsh = 1.7 - np.sqrt(3.38 - (igr + 0.7) ** 2)
    elif method == "steiber":
        igr_s = np.clip(igr, 0.001, 0.999)
        vsh = 0.5 * igr_s / (1.5 - igr_s)
    elif method == "stieber_gcur":
        vsh = (2.0 ** (gcur * igr) - 1.0) / (2.0**gcur - 1.0)
    else:
        raise ValueError(
            f"method must be one of linear/larionov_tertiary/larionov_older/"
            f"clavier/steiber/stieber_gcur, got {method!r}"
        )
    return _clip(np.asarray(vsh), clip)


def vshale_neutron_density(
    phi_n: ArrayLike,
    phi_d: ArrayLike,
    phi_n_sh: ArrayLike,
    phi_d_sh: ArrayLike,
    *,
    clip: _Clip = None,
) -> _Float:
    """Shale volume from neutron-density separation.

    ``Vsh = (phi_n-phi_d)/(phi_n_sh-phi_d_sh)`` — the sand separation scaled by
    the shale-point separation.  Sources: src2014_12/article1 (also the
    dry-clay ``phi_*_cldry`` endpoints), src2023_08/article3 (vclay_nmr uses the
    NMR analogue).
    """
    num = np.asarray(phi_n, np.float64) - np.asarray(phi_d, np.float64)
    den = np.asarray(phi_n_sh, np.float64) - np.asarray(phi_d_sh, np.float64)
    return _clip(np.asarray(num / den), clip)


def combine_clay_indicators(*vcl: ArrayLike, how: str = "mean") -> _Float:
    """Combine several clay-volume indicators into one estimate.

    ``how="mean"`` averages (e.g. ND and uranium-free CGR indicators),
    ``"min"`` takes the most optimistic, ``"max"`` the hybrid GR/NMR upper
    bound.  Sources: src2016_04/article2 (mean), src2023_08/article3 (max
    hybrid).
    """
    stack = np.stack([np.asarray(v, np.float64) for v in vcl], axis=0)
    if how == "mean":
        return np.asarray(stack.mean(axis=0))
    if how == "min":
        return np.asarray(stack.min(axis=0))
    if how == "max":
        return np.asarray(stack.max(axis=0))
    raise ValueError(f"how must be mean/min/max, got {how!r}")


# ==========================================================================
# 2. Porosity from logs
# ==========================================================================


def density_porosity(
    rho_b: ArrayLike,
    rho_ma: ArrayLike = 2.65,
    rho_fl: ArrayLike = 1.0,
    *,
    clip: _Clip = None,
) -> _Float:
    """Density porosity ``phi_D = (rho_ma-rho_b)/(rho_ma-rho_fl)``.

    ``rho_ma`` defaults to sandstone 2.65; carbonate 2.71, dolomite 2.85, and
    brine fluids 1.03/1.025 are passed explicitly by the caller (the corpus
    uses all of them).  Clipping is opt-in.  Sources: src2015_06/article1,
    src2017_06/article6 (2.71), src2015_12/article3 (2.85/1.01),
    src2016_10/article3 (required endpoints).
    """
    rb = np.asarray(rho_b, np.float64)
    rma = np.asarray(rho_ma, np.float64)
    rfl = np.asarray(rho_fl, np.float64)
    return _clip(np.asarray((rma - rb) / (rma - rfl)), clip)


def neutron_density_porosity(phi_n: ArrayLike, phi_d: ArrayLike, *, method: str = "rms") -> _Float:
    """Combine neutron and density porosity.

    ``method="rms"`` gives the gas-corrected root-mean-square
    ``sqrt((phi_n**2+phi_d**2)/2)``; ``"mean"`` the arithmetic average
    ``(phi_n+phi_d)/2`` (shale/clay corrections stay in the caller).  Sources:
    src2020_02/article6 (rms), src2025_10/a6 (rms), src2023_02/article4 (mean).
    """
    n = np.asarray(phi_n, np.float64)
    d = np.asarray(phi_d, np.float64)
    if method == "rms":
        return np.asarray(np.sqrt((n**2 + d**2) / 2.0))
    if method == "mean":
        return np.asarray((n + d) / 2.0)
    raise ValueError(f"method must be rms/mean, got {method!r}")


def effective_porosity(
    phi_t: ArrayLike,
    vsh: ArrayLike,
    phi_sh: ArrayLike,
    *,
    clip: _Clip = (0.0, None),
) -> _Float:
    """Shale-corrected effective porosity ``phi_e = phi_t - Vsh*phi_sh``.

    ``phi_sh`` (the shale total porosity) is explicit; the corpus uses 0.10,
    0.27, 0.30.  Clips to non-negative by default; pass ``clip=None`` for the
    structural-shale-porosity form.  Sources: src2019_06/article1,
    src2020_02/article6, src2018_04/article2 (unclipped), src2022_02/article5
    (phi_sh=0.27).
    """
    pt = np.asarray(phi_t, np.float64)
    return _clip(
        np.asarray(pt - np.asarray(vsh, np.float64) * np.asarray(phi_sh, np.float64)), clip
    )


# ==========================================================================
# 3. Porosity from core / digital rock
# ==========================================================================


def porosity_from_volumes(v_bulk: ArrayLike, v_grain: ArrayLike) -> _Float:
    """Core porosity ``phi = (V_bulk - V_grain)/V_bulk``.

    Bulk volume first, grain volume second (the corpus writes both orders under
    ``helium_porosity``/``porosity``; here the order is fixed).  Sources:
    src2017_12/article6 (helium), src2017_10/article5, src2019_12/article4,
    src2021_10/article4.
    """
    bv = np.asarray(v_bulk, np.float64)
    return np.asarray((bv - np.asarray(v_grain, np.float64)) / bv)


def boyle_porosity(rho_bulk: ArrayLike, rho_grain: ArrayLike) -> _Float:
    """Boyle's-law density porosity ``phi = 1 - rho_bulk/rho_grain``.
    Sources: src2022_02/article6 (boyles_law_porosity).
    """
    return np.asarray(1.0 - np.asarray(rho_bulk, np.float64) / np.asarray(rho_grain, np.float64))


def boyle_grain_volume(
    v_cell: ArrayLike, v_expansion: ArrayLike, p1: ArrayLike, p2: ArrayLike
) -> _Float:
    """Boyle's-law grain volume ``V_grain = V_cell - V_exp/(p1/p2 - 1)``.

    Helium-expansion (double-cell) grain volume.  Sources: src2021_10/article4.
    """
    vc = np.asarray(v_cell, np.float64)
    ve = np.asarray(v_expansion, np.float64)
    return np.asarray(vc - ve / (np.asarray(p1, np.float64) / np.asarray(p2, np.float64) - 1.0))


def fluid_summation_porosity(
    bv_oil: ArrayLike, bv_water: ArrayLike, bv_gas: ArrayLike = 0.0
) -> _Float:
    """Fluid-summation porosity ``phi = BV_oil + BV_water + BV_gas``.
    Sources: src2019_12/article4, src2022_02/article6.
    """
    return np.asarray(
        np.asarray(bv_oil, np.float64)
        + np.asarray(bv_water, np.float64)
        + np.asarray(bv_gas, np.float64)
    )


def porosity_from_voxel_count(pore_voxels: ArrayLike, total_voxels: ArrayLike) -> _Float:
    """Digital-rock porosity ``phi = pore_voxels/total_voxels``.

    The caller counts pore voxels under its own segmentation convention (some
    label pore==0, some pore==True).  Sources: src2017_12/article4,
    src2014_08/article3, src2022_04/article7, src2022_12/article7.
    """
    return np.asarray(np.asarray(pore_voxels, np.float64) / np.asarray(total_voxels, np.float64))


def gravimetric_porosity(
    m_dry: ArrayLike, m_sat: ArrayLike, v_bulk: ArrayLike, *, rho_fluid: float = 1.0
) -> _Float:
    """Mass-balance porosity ``phi = (m_sat-m_dry)/(rho_fluid*V_bulk)``.

    ``rho_fluid`` and the mass/volume units must be consistent (the corpus
    mixes g/cc·cm3 and kg/m3·m3).  Sources: src2023_06/article8 (rho_w=1000,
    kg/m3), src2026_04/a03.
    """
    ms = np.asarray(m_sat, np.float64) - np.asarray(m_dry, np.float64)
    return np.asarray(ms / (rho_fluid * np.asarray(v_bulk, np.float64)))


def ct_porosity(
    mu: ArrayLike,
    mu_grain: ArrayLike,
    mu_fluid: ArrayLike,
    *,
    clip: _Clip = None,
) -> _Float:
    """Porosity from CT attenuation ``phi = (mu_grain-mu)/(mu_grain-mu_fluid)``.
    Sources: src2018_10/article9, src2019_10/article6.
    """
    mg = np.asarray(mu_grain, np.float64)
    return _clip(
        np.asarray((mg - np.asarray(mu, np.float64)) / (mg - np.asarray(mu_fluid, np.float64))),
        clip,
    )


def ct_saturation(
    mu: ArrayLike,
    mu_dry: ArrayLike,
    mu_sat: ArrayLike,
    *,
    clip: _Clip = (0.0, 1.0),
) -> _Float:
    """Saturation from CT attenuation ``S = (mu-mu_dry)/(mu_sat-mu_dry)``.
    Clips to [0,1] by default.  Sources: src2019_10/article6.
    """
    md = np.asarray(mu_dry, np.float64)
    return _clip(
        np.asarray((np.asarray(mu, np.float64) - md) / (np.asarray(mu_sat, np.float64) - md)), clip
    )


# ==========================================================================
# 4. Mixing laws (density, fluids, tool response, fractions)
# ==========================================================================


def matrix_density_from_volumes(v: ArrayLike, rho: ArrayLike) -> _Float:
    """Grain density by volume-weighted (arithmetic) mixing ``sum(v_i*rho_i)``.

    ``v`` volume fractions and ``rho`` grain densities along the last axis.
    Sources: src2017_12/article5, src2021_12/article02, src2022_02/article5.
    """
    return np.asarray(np.sum(np.asarray(v, np.float64) * np.asarray(rho, np.float64), axis=-1))


def matrix_density_from_masses(
    w: ArrayLike, rho: ArrayLike, *, w_kerogen: float = 0.0, rho_kerogen: float = 1.43
) -> _Float:
    """Grain density by mass-weighted (harmonic) mixing.

    ``1/rho_ma = sum(w_i/rho_i) + w_kerogen/rho_kerogen`` — the reciprocal-mass
    average with an optional organic term.  Sources: src2018_06/article2
    (kerogen term, rho_k=1.43), src2016_04/article3, src2014_10/article2.
    """
    wa = np.asarray(w, np.float64)
    ra = np.asarray(rho, np.float64)
    recip = np.sum(wa / ra, axis=-1) + w_kerogen / rho_kerogen
    return np.asarray(1.0 / recip)


def fluid_density(saturations: ArrayLike, rhos: ArrayLike) -> _Float:
    """Pore-fluid density ``rho_fl = sum(S_i*rho_i)`` over the last axis.
    Sources: src2017_10/article4, src2023_02/article6, src2022_08/article4.
    """
    return np.asarray(
        np.sum(np.asarray(saturations, np.float64) * np.asarray(rhos, np.float64), axis=-1)
    )


def bulk_density(
    phi: ArrayLike,
    rho_ma: ArrayLike,
    rho_fl: ArrayLike = 1.0,
    *,
    v_k: ArrayLike = 0.0,
    rho_k: float = 1.30,
) -> _Float:
    """Forward bulk density ``rho_b = (1-phi-v_k)*rho_ma + v_k*rho_k + phi*rho_fl``.

    The three-component (matrix + kerogen + fluid) mass balance; ``v_k=0``
    reduces to the two-component form.  Keep units consistent (g/cc or kg/m3).
    Sources: src2018_10/article3 (kerogen), src2019_10/article1 (2-component),
    src2023_02/article6.
    """
    p = np.asarray(phi, np.float64)
    vk = np.asarray(v_k, np.float64)
    rma = np.asarray(rho_ma, np.float64)
    return np.asarray((1.0 - p - vk) * rma + vk * rho_k + p * np.asarray(rho_fl, np.float64))


def log_response(volumes: ArrayLike, endpoints: ArrayLike) -> _Float:
    """Linear tool-response mixing ``M = sum(V_j * R_j)`` = ``endpoints @ volumes``.

    Used for matrix GR, PEF, Sigma, and forward multi-mineral log synthesis.
    Sources: src2015_10/article3, src2015_08/article4 (linear_mixing_law),
    src2023_08/article3.
    """
    return np.asarray(np.asarray(endpoints, np.float64) @ np.asarray(volumes, np.float64))


def electron_density_to_bulk(rho_e: ArrayLike, *, a: float = 1.0704, b: float = -0.1883) -> _Float:
    """Electron- to bulk-density calibration ``rho_b = a*rho_e + b``.

    The (1.0704, -0.1883) constants are the standard tool calibration; the
    ``(0.1823, 1.07)`` inverse-form constants are passed by the caller.
    Sources: src2018_06/article2, src2019_10/article1, src2015_02/article5.
    """
    return np.asarray(a * np.asarray(rho_e, np.float64) + b)


def volume_to_weight_fractions(v: ArrayLike, rho: ArrayLike) -> _Float:
    """Convert volume fractions to weight fractions ``w_i = v_i*rho_i/sum(v*rho)``.
    Sources: src2015_08/article4.
    """
    mass = np.asarray(v, np.float64) * np.asarray(rho, np.float64)
    return np.asarray(mass / np.sum(mass, axis=-1, keepdims=True))


def weight_to_volume_fractions(w: ArrayLike, rho: ArrayLike) -> _Float:
    """Convert weight fractions to volume fractions ``v_i = (w_i/rho_i)/sum(w/rho)``.
    Sources: src2019_10/article1 (molar_fractions analogue).
    """
    vol = np.asarray(w, np.float64) / np.asarray(rho, np.float64)
    return np.asarray(vol / np.sum(vol, axis=-1, keepdims=True))


# ==========================================================================
# 5. Thomas-Stieber
# ==========================================================================


def thomas_stieber_phit(
    v_lam: ArrayLike, phi_sand: ArrayLike, phi_sh: ArrayLike, *, v_disp: ArrayLike = 0.0
) -> _Float:
    """Thomas-Stieber total porosity vs laminar shale volume.

    ``phi_t = phi_sand*(1-V_lam) + phi_sh*V_lam - V_disp`` — the laminated trend
    (dispersed shale subtracts pore-filling clay).  Sources:
    src2015_10/article6 (laminated/dispersed), src2022_02/article5.
    """
    vl = np.asarray(v_lam, np.float64)
    ps = np.asarray(phi_sand, np.float64)
    return np.asarray(
        ps * (1.0 - vl) + np.asarray(phi_sh, np.float64) * vl - np.asarray(v_disp, np.float64)
    )


def thomas_stieber_vlam(
    phi_t: ArrayLike,
    phi_sand: ArrayLike,
    phi_sh: ArrayLike,
    *,
    clip: _Clip = None,
) -> _Float:
    """Inverse Thomas-Stieber laminar shale ``V_lam=(phi_sand-phi_t)/(phi_sand-phi_sh)``.

    The net-to-grain fraction is ``1 - V_lam``.  Sources: src2015_10/article6,
    src2022_02/article5 (thomas_stieber_fntg).
    """
    ps = np.asarray(phi_sand, np.float64)
    return _clip(
        np.asarray((ps - np.asarray(phi_t, np.float64)) / (ps - np.asarray(phi_sh, np.float64))),
        clip,
    )


def thomas_stieber_sand_porosity(phi_t: ArrayLike, v_lam: ArrayLike, phi_sh: ArrayLike) -> _Float:
    """Sand (net) porosity ``phi_sand = (phi_t - V_lam*phi_sh)/(1 - V_lam)``.
    Sources: src2018_04/article2, src2015_10/article6, src2022_02/article5.
    """
    vl = np.asarray(v_lam, np.float64)
    num = np.asarray(phi_t, np.float64) - vl * np.asarray(phi_sh, np.float64)
    return np.asarray(num / (1.0 - vl))


# ==========================================================================
# 6. Organic matter / TOC / kerogen
# ==========================================================================


def kerogen_mass_fraction(toc: ArrayLike, *, k: float = 1.2) -> _Float:
    """Organic-matter mass fraction from TOC ``OM = k*TOC`` (carbon fraction 1/k).

    ``k`` is 1.2 (carbon 0.83) or 1.25 depending on source; explicit.  Sources:
    src2018_06/article2, src2014_10/article2.
    """
    return np.asarray(k * np.asarray(toc, np.float64))


def kerogen_volume_from_toc(
    toc: ArrayLike,
    rho_ref: ArrayLike,
    *,
    rho_k: float = 1.30,
    carbon_frac: float = 0.80,
) -> _Float:
    """Kerogen volume fraction ``V_k = (TOC/carbon_frac)*rho_ref/rho_k``.

    ``rho_ref`` is the bulk density (most copies) or the matrix density (TMALI
    form) per the source model; ``TOC`` is a fraction (divide wt% by 100 in the
    facade).  Sources: src2018_10/article3, src2019_06/article1,
    src2019_04/article1 (rho_b, 1.30/0.80), src2019_10/article1 (rho_ma).
    """
    om = np.asarray(toc, np.float64) / carbon_frac
    return np.asarray(om * np.asarray(rho_ref, np.float64) / rho_k)


def toc_schmoker(rho_b: ArrayLike, *, a: float = 154.497, b: float = 57.261) -> _Float:
    """Schmoker density TOC ``TOC[wt%] = a/rho_b - b``.

    The (154.497, 57.261) Schmoker-Hester constants are the corpus default;
    other basins pass their own.  The density-difference Schmoker variant
    ``(rho_min-rho_b)/(rho_min-rho_org)*100*fc`` is a distinct form kept in its
    facade.  Sources: src2016_04/article2, src2022_02/article6 (154.5/57.26).
    """
    return np.asarray(a / np.asarray(rho_b, np.float64) - b)


def toc_passey_dlogr(
    rt: ArrayLike,
    overlay: ArrayLike,
    rt_base: ArrayLike,
    overlay_base: ArrayLike,
    *,
    lom: float = 10.0,
    k_overlay: float = 0.02,
) -> _Float:
    """Passey Delta-log-R TOC ``TOC = dlogR * 10**(2.297 - 0.1688*LOM)``.

    ``dlogR = log10(Rt/Rt_base) + k_overlay*(overlay - overlay_base)``.  The
    overlay curve and its coefficient are explicit: sonic ``+0.02`` us/ft,
    density ``-2.50`` g/cc, neutron ``+4.0`` v/v, GR ``+0.01`` API.  Sources:
    src2019_08/article2 (sonic), src2025_06/toc_prediction (all overlays).
    """
    dlogr = np.log10(np.asarray(rt, np.float64) / np.asarray(rt_base, np.float64)) + k_overlay * (
        np.asarray(overlay, np.float64) - np.asarray(overlay_base, np.float64)
    )
    return np.asarray(dlogr * 10.0 ** (2.297 - 0.1688 * lom))


# ==========================================================================
# 7. Multi-mineral inversion
# ==========================================================================


def multimineral_solve(
    measured: ArrayLike,
    endpoints: ArrayLike,
    *,
    sigma: ArrayLike | None = None,
    closure: bool = True,
    closure_weight: float = 1e3,
    nonneg: bool = False,
    method: str = "lstsq",
) -> _Float:
    """Solve tool responses ``endpoints @ v = measured`` for mineral volumes ``v``.

    ``endpoints`` is the (n_logs, n_minerals) response matrix.  With
    ``closure`` a unity-sum row weighted by ``closure_weight`` (1e3, or 100 in
    some copies) is appended so ``sum(v)=1``.  ``sigma`` (per-log) applies
    inverse-variance weighting.  ``method``: ``"lstsq"`` (least squares),
    ``"nnls"`` (non-negative, lazy scipy), or ``"simplex"`` (projected-gradient
    onto ``v>=0, sum(v)=1``).  ``nonneg`` clips a lstsq result to >=0.
    Sources: src2016_04/article2 (w=1e3), src2014_06/article1 (w=100),
    src2020_12/article5 & src2021_12/article02 (simplex).
    """
    a = np.asarray(endpoints, np.float64)
    b = np.asarray(measured, np.float64)
    n_min = a.shape[1]

    if sigma is not None:
        w = 1.0 / np.asarray(sigma, np.float64)
        a = a * w[:, None]
        b = b * w

    if method == "simplex":
        v = np.full(n_min, 1.0 / n_min)
        ata = a.T @ a
        step = 1.0 / (np.linalg.norm(ata, ord=2) + 1e-12)
        for _ in range(8000):
            grad = a.T @ (a @ v - b)
            v = _project_simplex(v - step * grad)
        return np.asarray(v)

    if closure:
        a = np.vstack([a, np.full((1, n_min), closure_weight)])
        b = np.concatenate([b, [closure_weight]])

    if method == "nnls":
        from scipy.optimize import nnls

        v, _ = nnls(a, b)
        return np.asarray(v)
    if method == "lstsq":
        v = np.linalg.lstsq(a, b, rcond=None)[0]
        if nonneg:
            v = np.clip(v, 0.0, None)
        return np.asarray(v)
    raise ValueError(f"method must be lstsq/nnls/simplex, got {method!r}")


def _project_simplex(v: _Float) -> _Float:
    """Euclidean projection of a vector onto the probability simplex."""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.asarray(np.maximum(v - theta, 0.0))


# ==========================================================================
# 8. Volumetrics, cutoffs, net pay
# ==========================================================================


def bulk_volume_water(phi: ArrayLike, sw: ArrayLike) -> _Float:
    """Bulk volume water ``BVW = phi*Sw`` (constant in a Buckles zone).
    Sources: src2019_04/article1, src2022_02/article6.
    """
    return np.asarray(np.asarray(phi, np.float64) * np.asarray(sw, np.float64))


def hydrocarbon_pore_volume(phi: ArrayLike, sw: ArrayLike) -> _Float:
    """Hydrocarbon pore volume fraction ``HCPV = phi*(1-Sw)``.
    Sources: src2017_04/article2, src2017_10/article5, src2022_02/article5.
    """
    return np.asarray(np.asarray(phi, np.float64) * (1.0 - np.asarray(sw, np.float64)))


def pay_flag(
    phi: ArrayLike,
    vsh: ArrayLike | None = None,
    sw: ArrayLike | None = None,
    *,
    phi_cut: float = 0.08,
    vsh_cut: float = 0.40,
    sw_cut: float = 0.60,
) -> NDArray[np.bool_]:
    """Boolean net-pay flag from porosity, shale, and saturation cutoffs.

    ``phi>=phi_cut`` AND (``vsh<=vsh_cut`` if ``vsh`` given) AND
    (``sw<=sw_cut`` if ``sw`` given).  Cutoffs vary by study (sw 0.5-0.65,
    phi 0.05-0.15) and are explicit.  Sources: src2019_02/article2,
    src2021_02/article1, src2025_10/a6, src2022_02/article5.
    """
    flag = np.asarray(phi, np.float64) >= phi_cut
    if vsh is not None:
        flag = flag & (np.asarray(vsh, np.float64) <= vsh_cut)
    if sw is not None:
        flag = flag & (np.asarray(sw, np.float64) <= sw_cut)
    return np.asarray(flag)


def interval_thickness(depth: ArrayLike, flag: ArrayLike) -> float:
    """Summed thickness ``sum(|dz|)`` over samples where ``flag`` is true.

    ``dz`` is the absolute midpoint gradient of ``depth`` (works for irregular
    sampling).  Sources: src2021_02/article1 (net_pay), src2019_02/article2.
    """
    z = np.asarray(depth, np.float64)
    dz = np.abs(np.gradient(z)) if z.size > 1 else np.ones_like(z)
    return float(np.sum(dz[np.asarray(flag, bool)]))


def net_to_gross(
    depth: ArrayLike, net_flag: ArrayLike, gross_flag: ArrayLike | None = None
) -> float:
    """Net-to-gross ratio ``net_thickness / gross_thickness``.

    ``gross`` is the whole logged interval unless ``gross_flag`` (e.g. a
    reservoir flag) restricts it to a subset.  Sources: src2019_02/article2 &
    src2020_02/article6 (gross=whole), src2021_02/article1 (gross=reservoir).
    """
    z = np.asarray(depth, np.float64)
    net = interval_thickness(z, net_flag)
    if gross_flag is None:
        gross = interval_thickness(z, np.ones(z.shape, bool))
    else:
        gross = interval_thickness(z, gross_flag)
    return float(net / gross)
