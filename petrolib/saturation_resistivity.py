"""Water saturation from resistivity: Archie and shaly-sand models.

The first physics domain of the library migration — its Archie family alone
is re-implemented in ~39 article files with four different positional
argument orders, optional ``a``, and inconsistent clipping (the reason this
API is keyword-only with an explicit ``clip=``; CONVENTIONS.md rules 1-3).

Hazards this module's API is designed around (LIBRARY_MERGE_PLAN.md
section 9):

- **Argument order.**  ``archie_sw`` exists in the articles as
  ``(rt, rw, phi)``, ``(rw, rt, phi)``, ``(rt, phi, rw)`` and ``(phi, rt)``
  — all floats, so swaps are silent.  Data arrays first, everything else
  keyword-only.
- **F vs F*.**  The Archie formation factor ``F = a/phi**m`` and the
  Waxman-Smits ``F* = phi**-m*`` share a formula but not an exponent
  meaning; ``m*`` is fitted on shaly rock and is NOT interchangeable with
  Archie's ``m``.  The Waxman-Smits functions name their exponents
  explicitly to keep the two calibrations apart.
- **Two dual-water parameterizations.**  The Qv-based form (excess-clay
  term ``beta*Qv``, e.g. src2014_02/article1) and the bound-water form
  (``Sb*(Cb - Cw)``, Clavier et al., e.g. src2014_12/article1) are distinct
  functions here — their parameters cannot be mapped one-to-one.
- **Qv sources.**  Grain-based ``Qv = CEC*rho_g*(1-phi)/phi`` and the
  clay-based Juhasz normalization differ physically; both are provided
  under distinct names.
- **No baked-in locale calibrations.**  Textbook defaults ``a=1, m=2, n=2``
  are kept (they are the overwhelming convention and every docstring says
  so); field-study values (a basin's ``Rw``, ``B = 0.045``) are the
  caller's explicit arguments.

Units: resistivities in ohm-m and conductivities in S/m (or any mutually
consistent pair — the equations are ratio-based); ``phi`` and saturations
as fractions; ``qv`` in meq/ml (= meq/cm3); ``cec`` in meq/g; densities in
g/cm3.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2014_02/article1 -- Article 1: Solving Complex Dual-Water Equation using Dielectric-NMR-
  Spectroscopy and Conventional Logs. Willy Tan, Ryan Lafferty, Thomas J. Neville (2014).
  Petrophysics Vol. 55, No. 1 (February 2014), pp. 14-23. DOI: none assigned (this issue predates
  SPWLA DOI assignment).
src2014_02/article2 -- Article 2: Capillary Pressure and Resistivity Index Measurements in a Mixed-
  Wet Carbonate Reservoir Moustafa R. Dernaika, Mohamed S. Efnik, Safouh Koronfol, Svein M.
  Skjaeveland, Maisoon M. Al Mansoori, Hafez Hafez, Mohammed Z. Kalam (2014). Petrophysics Vol. 55,
  No. 1 (February 2014), pp. 24-30. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2014_12/article1 -- Article 1: Review of Existing Shaly-Sand Models and Introduction of a New
  Method Based on Dry-Clay Parameters. Max Peeters and Antony Holmes (2014). Petrophysics Vol. 55,
  No. 6 (December 2014), pp. 543-553. DOI: none assigned (this issue predates SPWLA DOI
  assignment).
src2015_06/article4 -- Article 4 (Technical Note): The Bateman-Konen Resistivity-Salinity
  Transform. Kennedy (2015). Petrophysics Vol. 56, No. 3 (June 2015), pp. 282-283. DOI: none
  assigned (this issue predates SPWLA DOI assignment).
src2015_08 -- Petrophysics Vol. 56 No. 4 (Aug 2015) (issue-level reference).
src2015_12/article4 -- Article 4: Petrophysical Challenges in Giant Carbonate Tengiz Field,
  Republic of Kazakhstan. Skalinski, Se, Playton, Theologou, Narr, Sullivan, Mallan (2015).
  Petrophysics Vol. 56, No. 6 (December 2015), pp. 615-647. DOI: none assigned (this issue predates
  SPWLA DOI assignment).
src2016_02/article5 -- Article 5: Graphical Solutions for Laminated and Dispersed Shaly Sands.
  Bootle (2016). Petrophysics Vol. 57, No. 1 (February 2016), pp. 51-59. DOI: none assigned (this
  issue predates SPWLA DOI assignment).
src2016_08/article3 -- Article 3: Drainage Capillary Pressure and Resistivity Index from Short-Wait
  Porous-Plate Experiments. Dernaika, Wilson, Skjaeveland, Ebeltoft (2016). Petrophysics Vol. 57,
  No. 4 (August 2016), pp. 369-376. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2017_10/article1 -- Article 1: Improved Assessment of Hydrocarbon Saturation in Mixed-Wet Rocks
  With Complex Pore Structure. Garcia, Heidari, Rostami (2017). Petrophysics Vol. 58, No. 5
  (October 2017), pp. 454-469. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2018_06/article4 -- Article 4: A New Resistivity-Based Model for Improved Hydrocarbon Saturation
  Assessment in Clay-Rich Formations Using Quantitative Geometry of the Clay Network. Garcia,
  Jagadisan, Rostami, Heidari (2018). DOI: 10.30632/PJV59N3-2018a3. Petrophysics Vol. 59 No. 3 (Jun
  2018).
src2018_10/article2 -- Article 2: A Novel X-Ray Tool for True Sourceless Density Logging. Simon,
  Tkabladze, Beekman, Atobatele, De Looz, Grover, Hamichi, Jundt, McFarland, Mlcak, Reijonen,
  Revol, Stewart, Yeboah, Zhang (2018). DOI: 10.30632/PJV59N5-2018a1. Petrophysics Vol. 59 No. 5
  (Oct 2018) - "Best of 2018 SPWLA Symposium" issue.
src2019_02/article1 -- Article 1 (Tutorial): Organic-Mudstone Petrophysics: Workflow to Estimate
  Storage Capacity (Part 1). Newsham, Comisky, Chemali (2019). DOI: 10.30632/PJV60N1Y2019t1.
src2019_04/article2 -- Article 2 (Tutorial): Introduction to Resistivity Principles for Formation
  Evaluation: A Tutorial Primer. Kennedy, Garcia (2019). DOI: 10.30632/PJV60N2-2019t2. Petrophysics
  Vol. 60 No. 2 (Apr 2019).
src2020_08 -- Petrophysics Vol. 61 No. 4 (Aug 2020) (issue-level reference).
src2020_10 -- Petrophysics Vol. 61 No. 5 (Oct 2020) (issue-level reference).
src2021_10 -- Petrophysics Vol. 62 No. 5 (Oct 2021) (issue-level reference).
src2021_12 -- Petrophysics Vol. 62 No. 6 (Dec 2021) (issue-level reference).
src2021_12/article10 -- Article 10: Enhanced Assessment of Fluid Saturation in the Wolfcamp
  Formation of the Permian Basin. Dash, Heidari (2021). DOI: 10.30632/PJV62N6-2021a10. Petrophysics
  Vol. 62 No. 6 (Dec 2021).
src2022_02/article5 -- Article 5: Evaluating Petrophysical Properties and Volumetrics Uncertainties
  of Sand Injectite Reservoirs - Norwegian North Sea. Kotwicki, Baig, Johansen, Leirdal, Aftret,
  Sandstad, Anthonsen, Gianotten, Hansen, Firinu (2022). DOI: 10.30632/PJV63N1-2022a5. Petrophysics
  Vol. 63 No. 1 (Feb 2022).
src2022_04/article6 -- Article 6: Rock Physics Modeling of Gas Hydrate Reservoirs Through
  Integrated Core and Well-Log Data in NGHP-02 Area, KG Offshore Basin, India. Kumar, Mishra,
  Chatterjee, Tiwari, Avadhani (2022). DOI: 10.30632/PJV63N2-2022a6. Petrophysics Vol. 63 No. 2
  (Apr 2022).
src2024_10 -- Petrophysics Vol. 65 No. 5 (Oct 2024) (issue-level reference).
src2024_10/water_saturation_equations -- Water Saturation Equations for Unconsolidated Reservoirs.
  Based on: Acosta et al. (2024), "Unveiling the Optimal Water Saturation Equation for
  Unconsolidated Reservoirs: A Case Study From the Tambaredjo Oil Field, Suriname", Petrophysics,
  Vol. 65, No. 5, pp. 739-764.
src2024_12 -- Petrophysics Vol. 65 No. 6 (Dec 2024) (issue-level reference).
src2025_10/a6 -- Article 6: Enhanced Learning Experience for New Petrophysicists Using Open-Source
  Carbonate Data and Python Programming Author: Imran M. Fadhil Ref: Petrophysics, Vol. 66, No. 5
  (October 2025), pp. 807-838. DOI: 10.30632/PJV66N5-2025a6.
src2025_12 -- Petrophysics Vol. 66 No. 6 (Dec 2025) — Best Papers of the 2024 SCA International
  Symposium (issue-level reference).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]


def _clip(values: _Float, clip: tuple[float, float] | None) -> _Float:
    if clip is None:
        return values
    return np.clip(values, clip[0], clip[1])


# --------------------------------------------------------------------------
# Archie family
# --------------------------------------------------------------------------


def formation_factor(phi: ArrayLike, *, a: float = 1.0, m: float = 2.0) -> _Float:
    """Archie formation resistivity factor ``F = a / phi**m``.

    NOT the Waxman-Smits ``F* = phi**-m*`` — ``m*`` is a different
    calibration (module docstring).  Sources: src2019_04/article2,
    src2014_02/article2, src2015_06/article4, and ~15 further modules.
    """
    porosity = np.asarray(phi, np.float64)
    return np.asarray(a / porosity**m)


def archie_sw(
    rt: ArrayLike,
    rw: ArrayLike,
    *,
    phi: ArrayLike,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    clip: tuple[float, float] | None = None,
) -> _Float:
    """Archie water saturation ``Sw = (a*Rw / (phi**m * Rt))**(1/n)``.

    ``clip`` is explicit and off by default — hidden [0, 1] clipping biases
    SCAL fits (CONVENTIONS.md rule 3); facades pass their article's
    historical clip.  Sources: src2014_02/article2,
    src2024_10/water_saturation_equations (clips to [0, 1]), and the ~35
    further members of the repository's largest duplication family.
    """
    rt_arr = np.asarray(rt, np.float64)
    rw_arr = np.asarray(rw, np.float64)
    porosity = np.asarray(phi, np.float64)
    sw = (a * rw_arr / (porosity**m * rt_arr)) ** (1.0 / n)
    return _clip(np.asarray(sw), clip)


def archie_rt(
    sw: ArrayLike,
    rw: ArrayLike,
    *,
    phi: ArrayLike,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
) -> _Float:
    """Forward Archie resistivity ``Rt = a*Rw / (phi**m * Sw**n)``.

    Inverse of :func:`archie_sw` (their round trip is a library property
    test).  Sources: src2021_12/article10, src2019_04/article2.
    """
    sw_arr = np.asarray(sw, np.float64)
    rw_arr = np.asarray(rw, np.float64)
    porosity = np.asarray(phi, np.float64)
    return np.asarray(a * rw_arr / (porosity**m * sw_arr**n))


def archie_conductivity(
    sw: ArrayLike,
    cw: ArrayLike,
    *,
    phi: ArrayLike,
    m: float = 2.0,
    n: float = 2.0,
) -> _Float:
    """Conductivity-domain Archie ``Ct = Cw * phi**m * Sw**n``.

    Sources: src2017_10/article1, src2022_04/article6.
    """
    sw_arr = np.asarray(sw, np.float64)
    cw_arr = np.asarray(cw, np.float64)
    porosity = np.asarray(phi, np.float64)
    return np.asarray(cw_arr * porosity**m * sw_arr**n)


def archie_sw_from_conductivity(
    ct: ArrayLike,
    cw: ArrayLike,
    *,
    phi: ArrayLike,
    m: float = 2.0,
    n: float = 2.0,
    clip: tuple[float, float] | None = None,
) -> _Float:
    """Invert conductivity-domain Archie: ``Sw = (Ct/(Cw*phi**m))**(1/n)``.

    The closed-form inverse of :func:`archie_conductivity`.  Sources:
    src2018_06/article4 (clay-network model saturation).
    """
    ct_arr = np.asarray(ct, np.float64)
    cw_arr = np.asarray(cw, np.float64)
    porosity = np.asarray(phi, np.float64)
    sw = (ct_arr / (cw_arr * porosity**m)) ** (1.0 / n)
    return _clip(np.asarray(sw), clip)


def resistivity_index(rt: ArrayLike, ro: ArrayLike) -> _Float:
    """Resistivity index by definition, ``RI = Rt/Ro``.

    Sources: src2014_02/article2, src2019_04/article2, src2025_12.
    """
    return np.asarray(np.asarray(rt, np.float64) / np.asarray(ro, np.float64))


def resistivity_index_from_sw(sw: ArrayLike, *, n: float = 2.0, b: float = 1.0) -> _Float:
    """Resistivity-index power law ``RI = b * Sw**-n`` (b=1 is Archie).

    Sources: src2014_02/article2, src2016_08/article3, src2020_08 (general b).
    """
    return np.asarray(b * np.asarray(sw, np.float64) ** (-n))


def sw_from_resistivity_index(ri: ArrayLike, *, n: float = 2.0, b: float = 1.0) -> _Float:
    """Water saturation from the resistivity index, ``Sw = (RI/b)**(-1/n)``.

    Sources: src2014_02/article2 (water_saturation_from_ri),
    src2016_08/article3.
    """
    return np.asarray((np.asarray(ri, np.float64) / b) ** (-1.0 / n))


# --------------------------------------------------------------------------
# Exponent fitting
# --------------------------------------------------------------------------


def fit_cementation_exponent(phi: ArrayLike, ff: ArrayLike) -> tuple[float, float]:
    """Fit ``(m, a)`` from a log-log formation-factor vs porosity regression.

    ``log10(F) = log10(a) - m*log10(phi)``; returns ``(m, a)`` — in that
    order, matching the src2014_02/article2 convention.  Sources:
    src2014_02/article2, src2019_04/article2.
    """
    log_phi = np.log10(np.asarray(phi, np.float64))
    log_ff = np.log10(np.asarray(ff, np.float64))
    slope, intercept = np.polyfit(log_phi, log_ff, 1)
    return float(-slope), float(10.0**intercept)


def fit_saturation_exponent(sw: ArrayLike, ri: ArrayLike) -> float:
    """Fit ``n`` from a log-log resistivity-index vs saturation regression.

    ``log10(RI) = -n*log10(Sw)``.  Sources: src2014_02/article2,
    src2016_08/article3.
    """
    log_sw = np.log10(np.asarray(sw, np.float64))
    log_ri = np.log10(np.asarray(ri, np.float64))
    slope, _ = np.polyfit(log_sw, log_ri, 1)
    return float(-slope)


def cementation_exponent_at_point(phi: ArrayLike, ff: ArrayLike, *, a: float = 1.0) -> _Float:
    """Single-point inversion ``m = log(F/a) / log(1/phi)``.

    The single-sample alternative to :func:`fit_cementation_exponent`.
    Sources: src2018_10/article2, src2021_10, src2020_10.
    """
    porosity = np.asarray(phi, np.float64)
    ff_arr = np.asarray(ff, np.float64)
    return np.asarray(np.log(ff_arr / a) / np.log(1.0 / porosity))


# --------------------------------------------------------------------------
# Quick-look derivatives
# --------------------------------------------------------------------------


def bulk_volume_water(phi: ArrayLike, sw: ArrayLike) -> _Float:
    """Bulk volume water ``BVW = phi * Sw``.

    Sources: src2015_12/article4, src2019_02/article1, src2022_02/article5.
    """
    return np.asarray(np.asarray(phi, np.float64) * np.asarray(sw, np.float64))


def apparent_water_resistivity(
    rt: ArrayLike, phi: ArrayLike, *, a: float = 1.0, m: float = 2.0
) -> _Float:
    """Apparent water resistivity ``Rwa = Rt * phi**m / a``.

    Equals ``Rw`` in a clean wet zone; the quick-look Rw picker.  Sources:
    src2015_08 (apparent_water_resistivity), src2024_12 (compute_rwa,
    m=1.776 locale default NOT inherited here).
    """
    rt_arr = np.asarray(rt, np.float64)
    porosity = np.asarray(phi, np.float64)
    return np.asarray(rt_arr * porosity**m / a)


# --------------------------------------------------------------------------
# Shaly-sand models
# --------------------------------------------------------------------------


def waxman_smits_conductivity(
    sw: ArrayLike,
    *,
    cw: ArrayLike,
    qv: ArrayLike,
    b: ArrayLike,
    phi: ArrayLike,
    m_star: float = 2.0,
    n_star: float = 2.0,
) -> _Float:
    """Waxman-Smits total conductivity.

    ``Ct = Sw**n* * Cw * phi**m* + B*Qv * phi**m* * Sw**(n*-1)`` — the
    Archie term plus the cation-exchange term.  Algebraically identical to
    the ``(Sw**n*/F*)(Cw + B*Qv/Sw)`` form some articles use.  ``b`` is the
    equivalent cation conductance (S/m per meq/ml) and is an explicit
    argument — the articles use fixed constants, Waxman-Thomas temperature
    laws, and saturating correlations, so no default is safe.  ``m_star``
    and ``n_star`` are the shaly-rock calibrations (NOT Archie's m, n).
    Sources: src2014_12/article1, src2016_02/article5, src2024_10
    (B=0.045 locale default not inherited).
    """
    sw_arr = np.asarray(sw, np.float64)
    cw_arr = np.asarray(cw, np.float64)
    qv_arr = np.asarray(qv, np.float64)
    b_arr = np.asarray(b, np.float64)
    porosity = np.asarray(phi, np.float64)
    return np.asarray(
        sw_arr**n_star * cw_arr * porosity**m_star
        + b_arr * qv_arr * porosity**m_star * sw_arr ** (n_star - 1.0)
    )


def dual_water_conductivity(
    sw: ArrayLike,
    *,
    cw: ArrayLike,
    cb: ArrayLike,
    swb: ArrayLike,
    phi: ArrayLike,
    m: float = 2.0,
    n: float = 2.0,
) -> _Float:
    """Dual-water total conductivity, bound-water form (Clavier et al.).

    ``Ct = Sw**n * phi**m * Cw + Sw**(n-1) * Swb * phi**m * (Cb - Cw)``
    with bound-water saturation ``swb`` and conductivity ``cb``.  This is
    NOT the Qv-based dual-water form (:func:`dual_water_conductivity_qv`)
    — the parameters do not map one-to-one.  Sources: src2014_12/article1,
    src2016_02/article5.
    """
    sw_arr = np.asarray(sw, np.float64)
    cw_arr = np.asarray(cw, np.float64)
    cb_arr = np.asarray(cb, np.float64)
    swb_arr = np.asarray(swb, np.float64)
    porosity = np.asarray(phi, np.float64)
    return np.asarray(
        sw_arr**n * porosity**m * cw_arr
        + sw_arr ** (n - 1.0) * swb_arr * porosity**m * (cb_arr - cw_arr)
    )


def dual_water_conductivity_qv(
    sw: ArrayLike,
    *,
    cw: ArrayLike,
    qv: ArrayLike,
    alpha_vqh: float,
    beta: float,
    phi: ArrayLike,
    m0: float = 2.0,
    n: float = 2.0,
) -> _Float:
    """Dual-water total conductivity, Qv-based form.

    ``Ct = phi**m0 * (Sw**n * (1 - alpha_vqh*Qv) * Cw + Sw**(n-1) * beta*Qv)``
    where ``alpha_vqh`` is the product of the salinity-expansion factor and
    the clay-water volume factor vQH (ml/meq), and ``beta`` the counterion
    equivalent conductance.  Sources: src2014_02/article1 (which exposes
    alpha and vqh separately — its facade passes their product).
    """
    sw_arr = np.asarray(sw, np.float64)
    cw_arr = np.asarray(cw, np.float64)
    qv_arr = np.asarray(qv, np.float64)
    porosity = np.asarray(phi, np.float64)
    effective_cw = (1.0 - alpha_vqh * qv_arr) * cw_arr
    return np.asarray(
        porosity**m0 * (sw_arr**n * effective_cw + sw_arr ** (n - 1.0) * beta * qv_arr)
    )


def _bisect_increasing(
    forward: Callable[[_Float], _Float],
    target: _Float,
    *,
    lo: float,
    hi: float,
    n_iter: int,
    tol: float,
) -> _Float:
    """Vectorized bisection for a forward model increasing in Sw."""
    low = np.full_like(target, lo, dtype=np.float64)
    high = np.full_like(target, hi, dtype=np.float64)
    mid = 0.5 * (low + high)
    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        too_high = forward(mid) > target
        high = np.where(too_high, mid, high)
        low = np.where(too_high, low, mid)
        if np.all(high - low < tol):
            break
    return np.asarray(0.5 * (low + high))


def waxman_smits_sw(
    ct: ArrayLike,
    *,
    cw: ArrayLike,
    qv: ArrayLike,
    b: ArrayLike,
    phi: ArrayLike,
    m_star: float = 2.0,
    n_star: float = 2.0,
    lo: float = 1e-6,
    hi: float = 1.0,
    n_iter: int = 100,
    tol: float = 1e-12,
    clip: tuple[float, float] | None = None,
) -> _Float:
    """Invert :func:`waxman_smits_conductivity` for Sw by bisection.

    ``ct`` is the measured total conductivity (S/m).  The bracket
    ``[lo, hi]`` and iteration controls are explicit so facades can
    reproduce their article's solver settings exactly.  Sources:
    src2016_02/article5, src2024_10 (iterative Rt-domain form).
    """
    ct_arr = np.asarray(ct, np.float64)
    cw_arr = np.asarray(cw, np.float64)
    qv_arr = np.asarray(qv, np.float64)
    b_arr = np.asarray(b, np.float64)
    porosity = np.asarray(phi, np.float64)

    def forward(sw: _Float) -> _Float:
        return waxman_smits_conductivity(
            sw, cw=cw_arr, qv=qv_arr, b=b_arr, phi=porosity, m_star=m_star, n_star=n_star
        )

    broadcast = np.broadcast_arrays(ct_arr, cw_arr, qv_arr, b_arr, porosity)[0]
    target = np.asarray(np.broadcast_to(ct_arr, broadcast.shape), np.float64)
    sw = _bisect_increasing(forward, target, lo=lo, hi=hi, n_iter=n_iter, tol=tol)
    return _clip(sw, clip)


def dual_water_sw(
    ct: ArrayLike,
    *,
    cw: ArrayLike,
    cb: ArrayLike,
    swb: ArrayLike,
    phi: ArrayLike,
    m: float = 2.0,
    n: float = 2.0,
    lo: float = 1e-6,
    hi: float = 1.0,
    n_iter: int = 100,
    tol: float = 1e-12,
    clip: tuple[float, float] | None = None,
) -> _Float:
    """Invert the bound-water :func:`dual_water_conductivity` for Sw.

    Same bisection contract as :func:`waxman_smits_sw`.  Sources:
    src2014_12/article1, src2014_02/article1 (whose Qv-form solver is the
    same bisection against :func:`dual_water_conductivity_qv`).
    """
    ct_arr = np.asarray(ct, np.float64)
    cw_arr = np.asarray(cw, np.float64)
    cb_arr = np.asarray(cb, np.float64)
    swb_arr = np.asarray(swb, np.float64)
    porosity = np.asarray(phi, np.float64)

    def forward(sw: _Float) -> _Float:
        return dual_water_conductivity(
            sw, cw=cw_arr, cb=cb_arr, swb=swb_arr, phi=porosity, m=m, n=n
        )

    broadcast = np.broadcast_arrays(ct_arr, cw_arr, cb_arr, swb_arr, porosity)[0]
    target = np.asarray(np.broadcast_to(ct_arr, broadcast.shape), np.float64)
    sw = _bisect_increasing(forward, target, lo=lo, hi=hi, n_iter=n_iter, tol=tol)
    return _clip(sw, clip)


def dual_water_sw_qv(
    ct: ArrayLike,
    *,
    cw: ArrayLike,
    qv: ArrayLike,
    alpha_vqh: float,
    beta: float,
    phi: ArrayLike,
    m0: float = 2.0,
    n: float = 2.0,
    lo: float = 1e-6,
    hi: float = 1.0,
    n_iter: int = 100,
    tol: float = 1e-12,
    clip: tuple[float, float] | None = None,
) -> _Float:
    """Invert the Qv-form :func:`dual_water_conductivity_qv` for Sw.

    Same bisection contract as :func:`waxman_smits_sw` — the bracket and
    iteration controls are explicit so facades reproduce their article's
    solver exactly (src2014_02/article1 uses lo=1e-4, tol=1e-10).
    """
    ct_arr = np.asarray(ct, np.float64)
    cw_arr = np.asarray(cw, np.float64)
    qv_arr = np.asarray(qv, np.float64)
    porosity = np.asarray(phi, np.float64)

    def forward(sw: _Float) -> _Float:
        return dual_water_conductivity_qv(
            sw,
            cw=cw_arr,
            qv=qv_arr,
            alpha_vqh=alpha_vqh,
            beta=beta,
            phi=porosity,
            m0=m0,
            n=n,
        )

    broadcast = np.broadcast_arrays(ct_arr, cw_arr, qv_arr, porosity)[0]
    target = np.asarray(np.broadcast_to(ct_arr, broadcast.shape), np.float64)
    sw = _bisect_increasing(forward, target, lo=lo, hi=hi, n_iter=n_iter, tol=tol)
    return _clip(sw, clip)


def simandoux_sw(
    rt: ArrayLike,
    rw: ArrayLike,
    *,
    phi: ArrayLike,
    vsh: ArrayLike,
    rsh: ArrayLike,
    a: float = 1.0,
    m: float = 2.0,
    clip: tuple[float, float] | None = None,
) -> _Float:
    """Simandoux (1963) water saturation, closed-form for n = 2.

    Solves ``1/Rt = Sw**2 * phi**m/(a*Rw) + Vsh*Sw/Rsh`` — the quadratic in
    Sw.  The equation is only quadratic for n = 2, which is why this
    function has no ``n`` parameter (some article variants accept an ``n``
    kwarg they silently ignore; the canonical API refuses to).  Sources:
    src2024_10/water_saturation_equations, src2025_10/a6, src2021_12.
    """
    rt_arr = np.asarray(rt, np.float64)
    rw_arr = np.asarray(rw, np.float64)
    porosity = np.asarray(phi, np.float64)
    vsh_arr = np.asarray(vsh, np.float64)
    rsh_arr = np.asarray(rsh, np.float64)
    sand = porosity**m / (a * rw_arr)
    shale = vsh_arr / rsh_arr
    sw = (-shale + np.sqrt(shale**2 + 4.0 * sand / rt_arr)) / (2.0 * sand)
    return _clip(np.asarray(sw), clip)


def indonesia_sw(
    rt: ArrayLike,
    rw: ArrayLike,
    *,
    phi: ArrayLike,
    vcl: ArrayLike,
    rcl: ArrayLike,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    clip: tuple[float, float] | None = None,
) -> _Float:
    """Indonesia (Poupon-Leveaux) water saturation, closed form.

    Solves ``1/sqrt(Rt) = (Vcl**(1-Vcl/2)/sqrt(Rcl)
    + sqrt(phi**m/(a*Rw))) * Sw**(n/2)`` for Sw.  Sources:
    src2025_10/a6, src2024_10 (whose iterative variant's facade must be
    checked for the same equation before delegation).
    """
    rt_arr = np.asarray(rt, np.float64)
    rw_arr = np.asarray(rw, np.float64)
    porosity = np.asarray(phi, np.float64)
    vcl_arr = np.asarray(vcl, np.float64)
    rcl_arr = np.asarray(rcl, np.float64)
    clay_term = vcl_arr ** (1.0 - 0.5 * vcl_arr) / np.sqrt(rcl_arr)
    sand_term = np.sqrt(porosity**m / (a * rw_arr))
    sw = (1.0 / (np.sqrt(rt_arr) * (clay_term + sand_term))) ** (2.0 / n)
    return _clip(np.asarray(sw), clip)


# --------------------------------------------------------------------------
# Cation exchange (Qv)
# --------------------------------------------------------------------------


def qv_from_cec(cec: ArrayLike, *, rho_grain: ArrayLike, phi: ArrayLike) -> _Float:
    """Cation exchange capacity per pore volume, grain-based.

    ``Qv = CEC * rho_grain * (1 - phi) / phi`` with CEC in meq/g,
    grain density in g/cm3, Qv in meq/ml.  Sources: src2014_02/article1,
    src2020_08 (mind its /100 for mmol/100g units at the call site).
    """
    cec_arr = np.asarray(cec, np.float64)
    rho = np.asarray(rho_grain, np.float64)
    porosity = np.asarray(phi, np.float64)
    return np.asarray(cec_arr * rho * (1.0 - porosity) / porosity)


def cec_from_qv(qv: ArrayLike, *, rho_grain: ArrayLike, phi: ArrayLike) -> _Float:
    """Inverse of :func:`qv_from_cec`.  Sources: src2014_02/article1."""
    qv_arr = np.asarray(qv, np.float64)
    rho = np.asarray(rho_grain, np.float64)
    porosity = np.asarray(phi, np.float64)
    return np.asarray(qv_arr * porosity / (rho * (1.0 - porosity)))


def qv_juhasz(
    v_clay_dry: ArrayLike,
    *,
    rho_clay: ArrayLike,
    cec_clay: ArrayLike,
    phit: ArrayLike,
) -> _Float:
    """Juhasz clay-based Qv normalization.

    ``Qv = Vcl_dry * rho_clay * CEC_clay / phit`` — physically distinct
    from the grain-based :func:`qv_from_cec` (dry-clay properties, total
    porosity).  Sources: src2014_12/article1, src2016_02/article5.
    """
    vcl = np.asarray(v_clay_dry, np.float64)
    rho = np.asarray(rho_clay, np.float64)
    cec = np.asarray(cec_clay, np.float64)
    porosity = np.asarray(phit, np.float64)
    return np.asarray(vcl * rho * cec / porosity)
