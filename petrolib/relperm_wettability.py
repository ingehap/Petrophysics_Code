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

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2014_02/article3 -- Article 3: An Evaluation of Spontaneous Imbibition of Water into Oil-Wet
  Carbonate Reservoir Cores Using Nanofluid. Abbas Roustaei (2014). Petrophysics Vol. 55, No. 1
  (February 2014), pp. 31-37. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2014_08 -- Petrophysics Vol. 55 No. 4 (Aug 2014) (issue-level reference).
src2014_08/article1 -- Article 1: Drainage Three-Phase Flow Relative Permeability on Oil-Wet
  Carbonate Reservoir Rock Types: Experiments, Interpretation and Comparison with Standard
  Correlations. P. Egermann, K. Mejdoub, J.-M. Lombard, O. Vizika, Z. Kalam (2014). Petrophysics
  Vol. 55, No. 4 (August 2014), pp. 287-293. DOI: none assigned (this issue predates SPWLA DOI
  assignment).
src2014_08/article4 -- Article 4: Impact of Wettability on Residual Oil Saturation and Capillary
  Desaturation Curves K. J. Humphry, B. M. J. M. Suijkerbuijk, H. A. van der Linde, S. G. J.
  Pieterse, S. K. Masalmeh (2014). Petrophysics Vol. 55, No. 4 (August 2014), pp. 313-318. DOI:
  none assigned (this issue predates SPWLA DOI assignment).
src2014_12/article3 -- Article 3: Experimental Study of the Effects of Wettability and Fluid
  Saturation on NMR and Dielectric Measurements in Limestone Lalitha Venkataramanan, Martin D.
  Hurlimann, Jeffrey A. Tarvin, Kamilla Fellah, Diana Acero-Allard, Nikita V. Seleznev (2014).
  Petrophysics Vol. 55, No. 6 (December 2014), pp. 572-586. DOI: none assigned (this issue predates
  SPWLA DOI assignment).
src2016_02 -- Petrophysics Vol. 57 No. 1 (Feb 2016) (issue-level reference).
src2016_02/article1 -- Article 1: The Impact of Reservoir Conditions and Rock Heterogeneity on
  CO2-Brine Multiphase Flow in Permeable Sandstone. Krevor, Reynolds, Al-Menhali, Niu (2016).
  Petrophysics Vol. 57, No. 1 (February 2016), pp. 12-18. DOI: none assigned (this issue predates
  SPWLA DOI assignment).
src2016_02/article4 -- Article 4: Low-Salinity Waterflooding: Facts, Inconsistencies and the Way
  Forward. Hamon (2016). Petrophysics Vol. 57, No. 1 (February 2016), pp. 41-50. DOI: none assigned
  (this issue predates SPWLA DOI assignment).
src2016_06/article3 -- Article 3: Permeability Interpretation from Wireline Formation Testing
  Measurements with Consideration of Effective Thickness. Yang, Yang (2016). Petrophysics Vol. 57,
  No. 3 (June 2016), pp. 251-269. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2016_10/article2 -- Article 2: How Pore-Scale Attributes May Be Used to Derive Robust Drainage
  and Imbibition Water-Saturation Models in Complex Tight-Gas Reservoirs. Merletti, Gramin,
  Salunke, Hamman, Spain, Shabro, Armitage, Torres-Verdin, Salter, Dacy (2016). Petrophysics Vol.
  57, No. 5 (October 2016), pp. 447-464. DOI: none assigned (this issue predates SPWLA DOI
  assignment).
src2017_02 -- Petrophysics Vol. 58 No. 1 (Feb 2017) (issue-level reference).
src2017_02/article1 -- Article 1: Flow Regimes During Immiscible Displacement. Armstrong, McClure,
  Berrill, Rucker, Schluter, Berg (2017). Petrophysics Vol. 58, No. 1 (February 2017), pp. 10-18.
  DOI: none assigned (this issue predates SPWLA DOI assignment).
src2017_02/article5 -- Article 5: Comparison of Four Numerical Simulators for SCAL Experiments.
  Lenormand, Lorentzen, Maas, Ruth (2017). Petrophysics Vol. 58, No. 1 (February 2017), pp. 48-56.
  DOI: none assigned (this issue predates SPWLA DOI assignment).
src2018_02/article5 -- Article 5: Using Digital Rock Technology to Quality Control and Reduce
  Uncertainty in Relative Permeability Measurements. Schembre-McCabe, Kamath (2018). DOI:
  10.30632/petro_059_1_a4. Petrophysics Vol. 59 No. 1 (Feb 2018).
src2018_02/article8 -- Article 8: Application of an Optimization Method for the Restoration of Core
  Samples for SCAL Experiments. Sripal, James (2018). DOI: 10.30632/petro_059_1_a7. Petrophysics
  Vol. 59 No. 1 (Feb 2018).
src2018_04/article9 -- Article 9: Downhole Estimation of Relative Permeability With Integration of
  Formation-Tester Measurements and Advanced Well Logs. Hadibeik, Azari, Kalawina, Ramakrishna,
  Eyuboglu, Khan, Al-Rushaid, Al-Rashidi, Ahmad (2018). DOI: 10.30632/PJV59N2-2018a8 (inferred;
  body beyond extraction). Petrophysics Vol. 59 No. 2 (Apr 2018).
src2018_06/article3 -- Article 3: Water-Wet or Oil-Wet: is it Really That Simple in Shales?. Gupta,
  Jernigen, Curtis, Rai, Sondergeld (2018). DOI: 10.30632/PJV59N3-2018a2. Petrophysics Vol. 59 No.
  3 (Jun 2018).
src2019_02/article1 -- Article 1 (Tutorial): Organic-Mudstone Petrophysics: Workflow to Estimate
  Storage Capacity (Part 1). Newsham, Comisky, Chemali (2019). DOI: 10.30632/PJV60N1Y2019t1.
src2019_04/article11 -- Article 11: Loading Effects on Gas Relative Permeability of a Low-
  Permeability Sandstone. Agostini, Egermann, Jeannin, Portier, Skoczylas, Wang (2019). DOI:
  10.30632/PJV60N2-2019a9. Petrophysics Vol. 60 No. 2 (Apr 2019).
src2019_04/article3 -- Article 3: Pore-Scale Insights on Trapped Oil During Waterflooding of
  Sandstone Rocks of Varying Wettability States. Berthet, Hebert, Barbouteau, Andriamananjaona,
  Rivenq (2019). DOI: 10.30632/PJV60N2-2019a1. Petrophysics Vol. 60 No. 2 (Apr 2019).
src2019_08/article4 -- Article 4: Practical Approach to Derive Wettability Index by NMR in Core
  Analysis Experiments. Looyestijn (2019). DOI: 10.30632/PJV60N4-2019a4. Petrophysics Vol. 60 No. 4
  (Aug 2019).
src2019_08/article6 -- Article 6: Novel Coupling Smart Water-CO2 Flooding for Sandstone Reservoirs.
  Al-Saedi, Flori (2019). DOI: 10.30632/PJV60N4-2019a6. Petrophysics Vol. 60 No. 4 (Aug 2019).
src2020_04/article5 -- Article 5: Workflow for Upscaling Wettability From the Nanoscale to Core
  Scale. Rucker, Bartels, Bultreys, Boone, Singh, Garfi, Scanziani, Spurin, Yesufu-Rufai, Krevor,
  Blunt, Wilson, Mahani, Cnudde, Luckham, Georgiadis, Berg (2020). DOI: 10.30632/PJV61N2-2020a5.
  Petrophysics Vol. 61 No. 2 (Apr 2020).
src2020_04/article6 -- Article 6: Estimation of Gas-Condensate Relative Permeability Using a
  Lattice Boltzmann Modeling Approach. Schembre-McCabe, Kamath, Fager, Crouse (2020). DOI:
  10.30632/PJV61N2-2020a6. Petrophysics Vol. 61 No. 2 (Apr 2020).
src2020_04/article7 -- Article 7: Effect of Injection Pressure on the Imbibition Relative
  Permeability and Capillary Pressure Curves of Shale Gas Matrix. Al-Ameri, Mazeel (2020). DOI:
  10.30632/PJV61N2-2020a7. Petrophysics Vol. 61 No. 2 (Apr 2020).
src2020_06/article4 -- Article 4: Evaluation of Relative Permeability From Resistivity Data for
  Fractal Porous Media. Shi, Meng, Liu, Zhang, Wang (2020). DOI: 10.30632/PJV61N3-2020a4.
  Petrophysics Vol. 61 No. 3 (Jun 2020).
src2021_10/article2 -- Article 2: Enhanced Learning of Fundamental Petrophysical Concepts Through
  Image Processing and 3D Printing. Alyafei, Al Musleh, Bautista, Idris, Seers (2021). DOI:
  10.30632/PJV62N5-2021a2. Petrophysics Vol. 62 No. 5 (Oct 2021).
src2022_06/article9 -- Article 9: NMR-Based Wettability Index for Unconventional Rocks. Dick,
  Veselinovic, Bonnie, Kelly (2022). DOI: 10.30632/PJV63N3-2022a9. Petrophysics Vol. 63 No. 3 (Jun
  2022) — NMR SIG Special Issue.
src2022_10/article4 -- Article 4: Mud-Filtrate Invasion in Laminated and Spatially Heterogeneous
  Rocks: High-Resolution In-Situ Visualization and Analysis Using Time-Lapse X-Ray Microcomputed
  Tomography (Micro-CT). Schroeder, Torres-Verdin (2022). DOI: 10.30632/PJV63N5-2022a4.
  Petrophysics Vol. 63 No. 5 (Oct 2022).
src2023_06/article4 -- Gao, Y., Sorop, T., Brussee, N., van der Linde, H., Coorn, A., Appel, M.,
  Berg, S. "Advanced Digital-SCAL Measurements of Gas Trapped in Sandstone" Petrophysics, Vol. 64,
  No. 3 (June 2023), pp. 368-383 DOI: 10.30632/PJV64N3-2023a4.
src2024_04 -- Petrophysics Vol. 65 No. 2 (Apr 2024) (issue-level reference).
src2024_10 -- Petrophysics Vol. 65 No. 5 (Oct 2024) (issue-level reference).
src2025_02 -- Petrophysics Vol. 66 No. 1 (Feb 2025) (issue-level reference).
src2025_02/co2_brine_relperm -- CO2/Brine Relative Permeability: Reconciling SS and USS Methods.
  Mascle et al., 2025, Petrophysics 66(1), 26-43. DOI:10.30632/PJV66N1-2025a2.
src2025_02/dopant_impact_scal -- Impact of Dopants (NaI) on SCAL Experiments. Pairoys et al., 2025,
  Petrophysics 66(1), 123-133. DOI:10.30632/PJV66N1-2025a9.
src2025_02/enhanced_gas_recovery -- Enhanced Gas Recovery (EGR) by CO2 Injection. Jones et al.,
  2025, Petrophysics 66(1), 54-66. DOI:10.30632/PJV66N1-2025a4.
src2025_02/scal_model_ccs -- SCAL Model for CCS: LET Correlations for Relative Permeability and
  Capillary Pressure. Ebeltoft et al., 2025, Petrophysics 66(1), 10-25.
  DOI:10.30632/PJV66N1-2025a1.
src2025_12/analog_kr -- Analog Two-Phase Relative Permeability for CO₂/Brine Estimation.
  Petrophysics, 66(6), 969–981. DOI: 10.30632/PJV66N6-2025a4.
src2026_04/a02 -- Aljishi, M.K., Chitrala, Y., Dang, S.T., and Rai, C. (2026). Wettability-Based
  Pore Partitioning and Its Effects on Oil Recovery and Formation Damage in Unconventional
  Reservoirs. Petrophysics, 67(2), 263–279. DOI: 10.30632/PJV67N2-2026a2.
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


# --------------------------------------------------------------------------
# Dimensionless numbers and desaturation
# --------------------------------------------------------------------------


def capillary_number(*, mu: float, v: ArrayLike, sigma: float) -> _Float:
    """Capillary number ``Nca = mu*v/sigma`` (viscous / capillary).

    Keyword-only on purpose: the ~14 article definitions disagree on positional
    order (four put velocity first) though all compute the same ratio.  For the
    volumetric-rate form ``Q*mu/(A*sigma)`` pass ``v=Q/A``.  Sources:
    src2025_12/analog_kr, src2017_02/article1, src2014_08 (velocity-first).
    """
    return np.asarray(mu * np.asarray(v, np.float64) / sigma)


def bond_number(*, drho: ArrayLike, k: ArrayLike, sigma: float, g: float = 9.81) -> _Float:
    """Bond number ``Nb = drho*g*k/sigma`` (gravity / capillary).

    ``k`` is the characteristic area — permeability (m2) in the Darcy form, or
    a squared length (pass ``k=R**2``) in the pore-radius form.  Sources:
    src2014_08/article4 (k form, g passed as acceleration), src2022_10/article4
    (R**2 form, g=9.81).
    """
    drho_arr = np.asarray(drho, np.float64)
    k_arr = np.asarray(k, np.float64)
    return np.asarray(drho_arr * g * k_arr / sigma)


def trapping_number(nca: ArrayLike, nb: ArrayLike) -> _Float:
    """Total trapping number ``Nt = Nca + Nb`` (the additive form used repo-wide).

    Sources: src2014_08/article4 (trapping_number).
    """
    return np.asarray(np.asarray(nca, np.float64) + np.asarray(nb, np.float64))


def capillary_desaturation(
    n: ArrayLike,
    *,
    sor_max: float,
    sor_min: float,
    n_crit: float,
    exponent: float = 1.0,
) -> _Float:
    """Capillary desaturation curve ``Sor(N)``.

    ``Sor = Sor_min + (Sor_max - Sor_min)/(1 + (N/N_crit)**exponent)`` — the
    residual falls from the plateau ``Sor_max`` toward the floor ``Sor_min`` as
    the trapping number rises past ``N_crit``.  Articles that write the exponent
    as ``1/width`` pass ``exponent=1.0/width``.  Sources:
    src2019_04/article3, src2014_08/article4 (exponent = 1/width).
    """
    n_arr = np.asarray(n, np.float64)
    return np.asarray(sor_min + (sor_max - sor_min) / (1.0 + (n_arr / n_crit) ** exponent))


# --------------------------------------------------------------------------
# Land trapping
# --------------------------------------------------------------------------


def land_c(s_i_max: float, s_r_max: float) -> float:
    """Land trapping coefficient ``C = 1/Sr_max - 1/Si_max``.

    The single-endpoint form ``1/Sgt_max - 1`` (Si_max = 1) is ``land_c(1.0,
    Sgt_max)``.  Sources: src2016_02/article1, src2023_06/article4,
    src2025_02/scal_model_ccs, src2014_12/article3 (Si_max default 1).
    """
    return 1.0 / s_r_max - 1.0 / s_i_max


def land_trapped(
    s_i: ArrayLike,
    *,
    C: float | None = None,
    s_r_max: float | None = None,
) -> _Float:
    """Land trapped saturation ``Sr = Si/(1 + C*Si)``.

    Pass the coefficient ``C`` directly, or ``s_r_max`` to use the
    ``C = 1/Sr_max - 1`` (Si_max = 1) reduced form.  Sources:
    src2016_10/article2 (s_r_max form), src2025_02 (C form).
    """
    if C is None:
        if s_r_max is None:
            raise ValueError("land_trapped needs either C or s_r_max")
        C = 1.0 / s_r_max - 1.0
    si = np.asarray(s_i, np.float64)
    return np.asarray(si / (1.0 + C * si))


# --------------------------------------------------------------------------
# Wettability indices
# --------------------------------------------------------------------------


def amott_indices(
    vw_spont: float,
    vw_forced: float,
    vo_spont: float,
    vo_forced: float,
) -> tuple[float, float, float]:
    """Amott-Harvey indices ``(Iw, Io, Iah)`` from displaced volumes.

    ``Iw = Vw_spont/(Vw_spont+Vw_forced)``, ``Io`` likewise, and the
    Amott-Harvey index ``Iah = Iw - Io``.  Accepts imbibed volumes or the
    equivalent saturation changes; each ratio is zero-guarded.  Sources:
    src2016_02/article4 (amott_harvey_index), src2025_02/dopant_impact_scal.
    """
    dw = vw_spont + vw_forced
    do = vo_spont + vo_forced
    iw = vw_spont / dw if dw > 0 else 0.0
    io = vo_spont / do if do > 0 else 0.0
    return iw, io, iw - io


def usbm_index(area_drainage: float, area_imbibition: float) -> float:
    """USBM wettability index ``W = log10(A_drainage / A_imbibition)``.

    Positive is water-wet.  Sources: src2018_02/article8 (usbm_index).
    """
    return float(np.log10(area_drainage / area_imbibition))


def nmr_wettability_index(w_signal: ArrayLike, o_signal: ArrayLike) -> _Float:
    """NMR wettability index ``Iw = (w - o)/(w + o)``.

    Works on any matched water/oil pair — signal amplitudes, surface relaxation
    rates, or T2 peak areas.  Sources: src2018_06/article3, src2019_08/article4
    (relaxation rates), src2022_06/article9 (T2 areas).
    """
    w = np.asarray(w_signal, np.float64)
    o = np.asarray(o_signal, np.float64)
    return np.asarray((w - o) / (w + o))


# --------------------------------------------------------------------------
# Contact angle
# --------------------------------------------------------------------------


def young_contact_angle(sigma_so: float, sigma_sw: float, sigma_wo: float) -> _Float:
    """Young's-law contact angle (degrees) from the three interfacial tensions.

    ``cos(theta) = (sigma_so - sigma_sw)/sigma_wo``; the cosine is clipped to
    [-1, 1] before ``arccos``.  Sources: src2014_02/article3.
    """
    cos_t = np.clip((sigma_so - sigma_sw) / sigma_wo, -1.0, 1.0)
    return np.asarray(np.degrees(np.arccos(cos_t)))


def wenzel_angle(theta_young_deg: ArrayLike, roughness: float) -> _Float:
    """Wenzel apparent contact angle (degrees): ``cos(theta_app) = r*cos(theta)``.

    Roughness ``r >= 1`` amplifies the intrinsic wettability; the cosine is
    clipped to [-1, 1].  Sources: src2020_04/article5 (wenzel_contact_angle).
    """
    c = np.clip(roughness * np.cos(np.radians(np.asarray(theta_young_deg, np.float64))), -1.0, 1.0)
    return np.asarray(np.degrees(np.arccos(c)))


def work_of_adhesion(sigma_wo: float, contact_angle_deg: ArrayLike) -> _Float:
    """Young-Dupre work of adhesion ``W = sigma_wo*(1 + cos(theta))``.

    Sources: src2014_02/article3 (work_of_adhesion).
    """
    theta = np.asarray(contact_angle_deg, np.float64)
    return np.asarray(sigma_wo * (1.0 + np.cos(np.radians(theta))))


# --------------------------------------------------------------------------
# Classification and displacement efficiency
# --------------------------------------------------------------------------


def classify_wettability_angle(
    theta_deg: float, *, cuts: tuple[float, float] = (75.0, 105.0)
) -> str:
    """Classify wettability from the contact angle: water-wet / intermediate / oil-wet.

    Default cuts 75 deg / 105 deg (src2014_02/article3); pass ``cuts`` to shift
    the boundaries.  Articles using different labels (e.g. wetting/non-wetting)
    keep their own classifier.
    """
    if theta_deg < cuts[0]:
        return "water-wet"
    if theta_deg <= cuts[1]:
        return "intermediate"
    return "oil-wet"


def classify_wettability_index(i: float, *, scheme: str = "3class") -> str:
    """Classify wettability from an Amott/USBM-style index in [-1, 1].

    ``scheme="3class"``: water-wet / mixed-wet / oil-wet at +-0.3.
    ``scheme="5class"``: adds weakly water-/oil-wet bands at +-0.1.  Sources:
    src2026_04/a02 (3class), src2025_02/dopant_impact_scal (5class).
    """
    if scheme == "3class":
        if i > 0.3:
            return "water-wet"
        if i < -0.3:
            return "oil-wet"
        return "mixed-wet"
    if scheme == "5class":
        if i > 0.3:
            return "water-wet"
        if i > 0.1:
            return "weakly water-wet"
        if i >= -0.1:
            return "intermediate"
        if i >= -0.3:
            return "weakly oil-wet"
        return "oil-wet"
    raise ValueError(f"scheme must be '3class' or '5class', got {scheme!r}")


def displacement_efficiency(soi: ArrayLike, sor: ArrayLike) -> _Float:
    """Microscopic displacement efficiency ``Ed = (Soi - Sor)/Soi``.

    The same ``(Soi-Sor)/Soi`` form is also written ``recovery_factor`` in
    several articles.  Sources: src2019_08/article6, src2021_10/article2.
    """
    soi_arr = np.asarray(soi, np.float64)
    sor_arr = np.asarray(sor, np.float64)
    return np.asarray((soi_arr - sor_arr) / soi_arr)
