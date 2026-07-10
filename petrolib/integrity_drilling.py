"""Well integrity and drilling: cement bond, casing condition, leaks, mud gas.

The recurring cased-hole / drilling physics across the corpus (including the
Aug-2025 Well Integrity and Aug-2024 mud-gas special issues):

* cement-bond acoustics -- impedance (MRayl), reflection/transmission, CBL
  attenuation, the bond index in its linear (amplitude or attenuation) and
  logarithmic conventions, impedance-based annulus classification and cement
  quality scoring;
* casing condition -- plate thickness resonance, metal loss, condition bands,
  remaining life, and the diffusion-limited corrosion front;
* microannulus leak rates -- the annular-gap moment with gravity-corrected
  Hagen-Poiseuille (liquid) and isothermal compressible (gas) flow, and the
  parallel-plate cubic law;
* mud-gas interpretation -- Haworth wetness/balance/character ratios, Pixler
  component ratios with the Bernard ratio, fluid classification, and gas
  normalization for drilling parameters;
* the pore-pressure / drilling window -- hydrostatic and overburden pressure
  (SI plus the oilfield psi/bar wrappers), Eaton and Bowers pore pressure, and
  the ECD window test;
* mudcake growth in the empirical sqrt-t, Dewan closed-form, and Chin-ODE
  conventions.

SI in/out unless a suffix says otherwise (``*_psi``, ``*_bar``); acoustic
impedance is in MRayl; keyword defaults are the most common values observed in
the articles.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2014_04/article1_deepwater_gom_overview -- Article 1: Deepwater Exploration and Production in
  the Gulf of Mexico - Challenges and Opportunities. Hani Elshahawi (2014). Petrophysics Vol. 55,
  No. 2 (April 2014), pp. 81-87. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2014_04/article5_deepwater_formation_evaluation -- Article 5: Formation-Evaluation Challenges
  and Opportunities in Deepwater Roland Chemali, Wade Samec, Ron Balliet, Paul Cooper, David
  Torres, Chris Jones (2014). Petrophysics Vol. 55, No. 2 (April 2014), pp. 124-135. DOI: none
  assigned (this issue predates SPWLA DOI assignment).
src2019_02/article5_composite_cement_well_integrity -- Article 5: Novel Composite Cement for
  Improved Well Integrity Evaluation. Elshahawi, Huang, Pollock, Veedu (2019). DOI:
  10.30632/PJV60N1Y2019a4. Petrophysics Vol. 60 No. 1 (Feb 2019).
src2019_10/article6_microct_invasion_mudcake -- Article 6: Experimental Method for Time-Lapse
  Micro-CT Imaging of Mud-Filtrate Invasion and Mudcake Deposition. Schroeder, Torres-Verdin
  (2019). DOI: 10.30632/PJV60N5-2019a6. Petrophysics Vol. 60 No. 5 (Oct 2019).
src2020_08/article1_flexural_attenuation_casing -- Article 1: A Study of the Flexural Attenuation
  Technique Through Laboratory Measurements and Numerical Simulations. Sirevaag, Johansen, Larsen,
  Holt (2020). DOI: 10.30632/PJV61N4-2020a1. Petrophysics Vol. 61 No. 4 (Aug 2020).
src2021_02/article1_mudlog_net_pay_tutorial -- Article 1 (Tutorial): Maximizing Value From Mudlogs
  - Integrated Approach to Determine Net Pay. Malik, Hanson, Clinch (2021). DOI:
  10.30632/PJV62N1-2021t1. Petrophysics Vol. 62 No. 1 (Feb 2021).
src2021_12/article07_multistring_isolation_acoustic -- Article 7: Case Studies on Multistring
  Isolation Evaluation in P&A Operations. Zhang, Mueller, Bryce, Brockway, Iskander (2021). DOI:
  10.30632/PJV62N6-2021a7. Petrophysics Vol. 62 No. 6 (Dec 2021).
src2021_12/article08_overbalanced_drilling_correction -- Article 8: The Impact of Overbalanced
  Drilling From Exploration/Appraisal Wells to Field Development Plan. Mohammadlou, Reppert, Del
  Negro, Jones (2021). DOI: 10.30632/PJV62N6-2021a8. Petrophysics Vol. 62 No. 6 (Dec 2021).
src2022_02/article4_ultrasonic_creeping_shale -- Article 4: Ultrasonic Logging of Creeping Shale.
  Diez, Johansen, Larsen (2022). DOI: 10.30632/PJV63N1-2022a4. Petrophysics Vol. 63 No. 1 (Feb
  2022).
src2022_04/article4_microct_filtercake -- Article 4: In-Situ Visualization and Characterization of
  Filter-Cake Deposition Using Time-Lapse Micro-CT Imaging. Schroeder, Torres-Verdin (2022). DOI:
  10.30632/PJV63N2-2022a4. Petrophysics Vol. 63 No. 2 (Apr 2022).
src2022_10/article5_cement_acid_gas_corrosion -- Article 5: Corrosion Behavior and Mechanism
  Analysis of Oilwell Cement Under CO2 and H2S Conditions. Zhou, Zeng, Sun, Zhou, Lei, Wan, Luo,
  Wu, Zhang, Xiao (2022). DOI: 10.30632/PJV63N5-2022a5. Petrophysics Vol. 63 No. 5 (Oct 2022).
src2023_08/article2_invasion_simulation -- Implements the rock/fluid/mudcake equations and the
  radial 1-D mud-filtrate invasion + Archie-resistivity workflow described in: Merletti, G., Al
  Hajri, S., Rabinovich, M., Farmer, R., Bennis, M., Torres-Verdín, C. (2023). "Assessment of True
  Formation Resistivity and Water Saturation in Deeply Invaded Tight-Gas Sandstones Based on the
  Combined Numerical Simulation of Mud-Filtrate Invasion and Resistivity Logs", Petrophysics, Vol.
  64, No. 4, pp. 502-517. DOI: 10.30632/PJV64N4-2023a2.
src2024_08/mudgas_response -- What Causes Mud-Logging Mud Gas Response to Vary and Two Techniques
  to Quantify. Based on: Donovan, W.S. (2024), "What Causes Mud-Logging Mud Gas Response to Vary
  and Two Techniques to Quantify Mud Gas," Petrophysics, 65(4), pp. 565-584. DOI:
  10.30632/PJV65N4-2024a10.
src2024_12/m04_well_integrity_ccs -- Well Integrity Measurements Throughout the CCS Project Life
  Cycle. Based on: Valstar, Nettleton, Borchardt, Costeno, Landry, and Laronga (2024), Petrophysics
  65(6), pp. 896-912. DOI: 10.30632/PJV65N6-2024a4.
src2024_12/m05_casing_cement_inspection -- Casing and Cement Inspection: Logging Two Casing Sizes
  Simultaneously. Based on: Hawthorn, Ingebretson, Girneata, Delabroy, Winther, Steinsiek, and
  Leslie (2024), Petrophysics 65(6), pp. 913-918. DOI: 10.30632/PJV65N6-2024a5.
src2025_04/microannuli_leak_rate -- Advanced Ultrasonic Log Analysis and Mechanistic Modeling for
  Leak Rate Quantification Through Microannuli. Based on: Machicote et al., "The Road Through
  Microannuli: Advanced Ultrasonic Log Analysis and Mechanistic Modeling for Leak Rate
  Quantification", Petrophysics, Vol. 66, No. 2, April 2025, pp. 331–347.
src2025_04/overpressure_isotope -- Genetic Analysis of Overpressure While Drilling Based on Isotope
  Logging. Based on: Hu et al., "Genetic Analysis of Overpressure While Drilling Based on Isotope
  Logging Technology", Petrophysics, Vol. 66, No. 2, April 2025, pp. 283–293.
src2025_08/cement_snhr_emi -- Module 10: Through-Tubing Cement Evaluation (SNHR + EMI + ML)
  Implements ideas from: Zeghlache et al., "Challenges and Solutions for Advanced Through- Tubing
  Cement Evaluation," Petrophysics, vol. 66, no. 4, pp. 677–688, August 2025.
src2025_08/seven_pipe_em_corrosion -- Module 3: Seven-Pipe Electromagnetic Corrosion Evaluation
  Implements ideas from: Fouda et al., "First-Ever Seven-Pipe Corrosion Evaluation for
  Comprehensive Assessment of Pipe Integrity in Complex Well Completions," Petrophysics, vol. 66,
  no. 4, pp. 566–577, August 2025.
src2026_04/a11_awi_cement_evaluation -- Zhang, X., Zhang, X., Zhang, T., Li, X., Mei, C., Li, B.,
  Jiang, M., Liu, K., and Bai, Y. (2026). New Equipment and Method for Evaluating Anti-Water-
  Invasion Ability of Cement Slurry. Petrophysics, 67(2), 421–435. DOI: 10.30632/PJV67N2-2026a11.
src2026_06/a08_mud_gas_ratio_fluid_id -- Luo, P., Li, W., Lu, P., and Qubaisi, K. (2026). An
  Improved Mud Gas Ratio Method for Enhanced Fluid Identification While Drilling. Petrophysics,
  67(3), 582-593. DOI: 10.30632/PJV67N3-2026a8.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import acoustic_geomech

_Float = NDArray[np.float64]

#: Compressional velocity of casing steel, m/s.
V_STEEL = 5900.0
#: Fresh-water pressure gradient per specific gravity, psi/ft.
PSI_PER_FT_PER_SG = 0.433
#: Hydrostatic gradient per specific gravity, bar/m (rho_w * g / 1e5).
BAR_PER_SG_M = 0.0980665
#: Standard gravity, m/s^2.
G_STD = 9.80665


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


# --- cased-hole acoustics / cement ---------------------------------------------


def acoustic_impedance(rho: ArrayLike, v: ArrayLike, *, rho_unit: str = "kg/m3") -> _Float:
    """Acoustic impedance ``Z = rho*v`` in MRayl.

    ``rho_unit='kg/m3'`` takes SI density; ``'g/cc'`` takes g/cm^3 (the CBL
    articles' convention, ``Z = rho_gcc * v / 1000``).

    Sources: src2019_02/article5_composite_cement_well_integrity.
    """
    if rho_unit == "kg/m3":
        return acoustic_geomech.acoustic_impedance(rho, v, out="mrayl")
    if rho_unit == "g/cc":
        return np.asarray(acoustic_geomech.acoustic_impedance(rho, v) / 1000.0)
    raise ValueError(f"rho_unit must be 'kg/m3' or 'g/cc', got {rho_unit!r}")


def reflection_coefficient(z1: ArrayLike, z2: ArrayLike) -> _Float:
    """Normal-incidence reflection coefficient ``R = (Z2 - Z1)/(Z2 + Z1)`` (z1 incident side)."""
    return acoustic_geomech.reflection_coefficient(z1, z2)


def transmission_energy(z1: ArrayLike, z2: ArrayLike) -> _Float:
    """Transmitted energy fraction across an interface, ``1 - R^2``."""
    return acoustic_geomech.transmission_energy(z1, z2)


def attenuation_db(a_near: ArrayLike, a_far: ArrayLike, spacing_m: float | None = None) -> _Float:
    """Amplitude attenuation ``20*log10(A_near/A_far)`` in dB (dB/m if ``spacing_m`` given).

    Sources: src2020_08/article1_flexural_attenuation_casing.
    """
    db = 20.0 * np.log10(_arr(a_near) / _arr(a_far))
    if spacing_m is not None:
        db = db / spacing_m
    return np.asarray(db)


def attenuation_coefficient(a0: ArrayLike, ax: ArrayLike, x_m: ArrayLike) -> _Float:
    """Exponential attenuation coefficient ``ln(A0/Ax)/x`` (1/m) from ``A = A0*exp(-alpha*x)``.

    Sources: src2020_08/article1_flexural_attenuation_casing.
    """
    return acoustic_geomech.attenuation_coefficient(a0, ax, x_m)


def bond_index(
    measured: ArrayLike,
    free_pipe: float,
    well_bonded: float,
    *,
    method: str = "linear",
    input_kind: str = "amplitude",
    clip: tuple[float, float] = (0.0, 1.0),
) -> _Float:
    """Cement bond index in [0, 1] from a CBL measurement and its two anchors.

    ``method='linear'`` interpolates between the ``free_pipe`` (BI=0) and
    ``well_bonded`` (BI=1) anchors: for ``input_kind='amplitude'`` the amplitude
    falls with bond, ``BI = (free_pipe - A)/(free_pipe - well_bonded)``; for
    ``'attenuation'`` it rises, ``BI = (att - free_pipe)/(well_bonded -
    free_pipe)``.  ``method='log'`` (amplitude only) uses the logarithmic CBL
    convention ``BI = log(free_pipe/A) / log(free_pipe/well_bonded)`` with
    0.01 floors on the amplitudes.  The result is clipped to ``clip``.

    Sources: src2019_02/article5_composite_cement_well_integrity,
    src2025_04/microannuli_leak_rate.
    """
    m = _arr(measured)
    if method == "linear":
        if input_kind == "amplitude":
            bi = (free_pipe - m) / (free_pipe - well_bonded)
        elif input_kind == "attenuation":
            bi = (m - free_pipe) / (well_bonded - free_pipe)
        else:
            raise ValueError(f"input_kind must be 'amplitude' or 'attenuation', got {input_kind!r}")
        return np.asarray(np.clip(bi, clip[0], clip[1]))
    if method == "log":
        if input_kind != "amplitude":
            raise ValueError("method='log' supports input_kind='amplitude' only")
        if free_pipe <= well_bonded:
            return np.asarray(np.full_like(m, 1.0) if m.ndim else np.float64(1.0))
        num = np.log(free_pipe / np.maximum(m, 0.01))
        den = np.log(free_pipe / max(well_bonded, 0.01))
        if den <= 0:
            return np.asarray(np.zeros_like(m) if m.ndim else np.float64(0.0))
        return np.asarray(np.clip(num / den, clip[0], clip[1]))
    raise ValueError(f"method must be 'linear' or 'log', got {method!r}")


def bond_index_combined(
    bi_a: ArrayLike,
    bi_b: ArrayLike,
    *,
    weights: tuple[float, float] = (0.6, 0.4),
    corrector: Any = None,
) -> _Float:
    """Weighted combination of two bond indicators, clipped to [0, 1].

    ``corrector`` (optional callable) maps the raw combination to a corrected
    value before clipping -- e.g. an eccentricity-correction model.

    Sources: src2025_08/cement_snhr_emi.
    """
    combined = weights[0] * _arr(bi_a) + weights[1] * _arr(bi_b)
    if corrector is not None:
        combined = _arr(corrector(combined))
    return np.asarray(np.clip(combined, 0.0, 1.0))


def classify_annulus(
    z_mrayl: float, *, gas_max: float = 0.5, liquid_max: float = 2.6, cement_min: float = 3.0
) -> str:
    """Classify the annulus fill from its acoustic impedance (MRayl).

    ``'gas'`` below ``gas_max``, ``'liquid'`` below ``liquid_max``, ``'cement'``
    at or above ``cement_min``, else ``'transition'``.  Pass
    ``cement_min=liquid_max`` to remove the transition band.

    Sources: src2019_02/article5_composite_cement_well_integrity,
    src2021_12/article07_multistring_isolation_acoustic.
    """
    if z_mrayl < gas_max:
        return "gas"
    if z_mrayl < liquid_max:
        return "liquid"
    if z_mrayl >= cement_min:
        return "cement"
    return "transition"


#: Cement-quality (good, fair) impedance thresholds in MRayl by cement type.
CEMENT_QUALITY_THRESHOLDS = {
    "Portland": (4.0, 2.5),
    "CO2_resistant": (3.5, 2.0),
    "epoxy_resin": (2.5, 1.5),
}


def cement_quality_score(
    z_mrayl: ArrayLike,
    *,
    cement_type: str = "Portland",
    thresholds: tuple[float, float] | None = None,
) -> _Float:
    """Continuous cement-quality score in [0, 1] from annulus impedance (MRayl).

    Above the ``good`` threshold the score is 1; between ``fair`` and ``good``
    it ramps 0.5..1; below ``fair`` it ramps toward 0.  Thresholds default per
    ``cement_type`` (:data:`CEMENT_QUALITY_THRESHOLDS`).

    Sources: src2024_12/m04_well_integrity_ccs.
    """
    imp = _arr(z_mrayl)
    good, fair = (
        thresholds
        if thresholds is not None
        else CEMENT_QUALITY_THRESHOLDS.get(cement_type, (4.0, 2.5))
    )
    quality = np.where(
        imp >= good,
        1.0,
        np.where(imp >= fair, 0.5 + 0.5 * (imp - fair) / (good - fair), 0.25 * imp / fair),
    )
    return np.asarray(np.clip(quality, 0, 1))


def classify_cement_from_cbl(
    relative_amp: ArrayLike, *, good_max: float = 0.15, medium_max: float = 0.30
) -> NDArray[Any]:
    """CBL relative-amplitude cement classes ``'Good'``/``'Medium'``/``'Poor'``.

    ``relative_amp <= good_max`` is Good, ``<= medium_max`` Medium, else Poor.
    """
    amp = _arr(relative_amp)
    return np.asarray(
        np.where(amp <= good_max, "Good", np.where(amp <= medium_max, "Medium", "Poor"))
    )


# --- casing condition -----------------------------------------------------------


def casing_resonance_frequency(
    thickness_m: ArrayLike, *, v: float = V_STEEL, n: int = 1, correction: float = 1.0
) -> _Float:
    """Casing plate thickness-resonance frequency ``f = corr * n * v / (2 d)`` (Hz).

    ``correction`` absorbs mode corrections (e.g. 0.95 for the S1
    negative-group-velocity minimum).

    Sources: src2021_12/article07_multistring_isolation_acoustic,
    src2022_02/article4_ultrasonic_creeping_shale.
    """
    return np.asarray(correction * n * v / (2.0 * _arr(thickness_m)))


def casing_thickness_from_resonance(
    freq_hz: ArrayLike, *, v: float = V_STEEL, n: int = 1, correction: float = 1.0
) -> _Float:
    """Casing wall thickness (m) from the thickness-resonance frequency (Hz)."""
    return np.asarray(correction * n * v / (2.0 * _arr(freq_hz)))


def metal_loss_pct(measured: ArrayLike, nominal: ArrayLike) -> _Float:
    """Casing metal loss ``(1 - measured/nominal)*100``, clipped to [0, 100] %.

    Sources: src2025_08/seven_pipe_em_corrosion.
    """
    return np.asarray(np.clip((1.0 - _arr(measured) / _arr(nominal)) * 100.0, 0, 100))


def casing_condition(
    loss_pct: ArrayLike, *, bands: tuple[float, float, float] = (10.0, 25.0, 42.5)
) -> NDArray[Any]:
    """Casing condition class from metal loss %: good / fair / poor / critical.

    Strict ``<`` at each band edge, matching the corpus convention.

    Sources: src2024_12/m05_casing_cement_inspection.
    """
    loss = _arr(loss_pct)
    return np.asarray(
        np.where(
            loss < bands[0],
            "good",
            np.where(loss < bands[1], "fair", np.where(loss < bands[2], "poor", "critical")),
        )
    )


def remaining_life_years(
    thickness_mm: float, min_acceptable_mm: float, rate_mm_per_yr: float
) -> float:
    """Remaining casing life ``(t - t_min)/rate`` (years), floored at 0.

    A non-positive corrosion rate returns the 999.0 sentinel (no wear).
    """
    if rate_mm_per_yr > 0:
        return max(0, (thickness_mm - min_acceptable_mm) / rate_mm_per_yr)
    return 999.0


def corrosion_front_depth(t: ArrayLike, K: float = 2.5) -> _Float:
    """Diffusion-limited corrosion front ``x = K*sqrt(t)`` (units follow ``K``).

    Sources: src2022_10/article5_cement_acid_gas_corrosion.
    """
    return np.asarray(K * np.sqrt(np.maximum(_arr(t), 0.0)))


# --- microannulus leaks ---------------------------------------------------------


def microannulus_omega(r_casing_m: float, aperture_m: float) -> float:
    """Annular-gap fourth moment ``Omega`` (m^4) of a microannulus.

    For the gap between ``R1 = r_casing_m`` and ``R2 = R1 + aperture``:
    ``Omega = R2^4 - R1^4 - (R2^2 - R1^2)^2 / ln(R2/R1)``; zero for a closed
    (non-positive) aperture.

    Sources: src2025_04/microannuli_leak_rate.
    """
    if aperture_m <= 0:
        return 0.0
    r1 = r_casing_m
    r2 = r_casing_m + aperture_m
    ln_ratio = np.log(r2 / r1)
    if ln_ratio < 1e-12:
        return 0.0
    return float(r2**4 - r1**4 - (r2**2 - r1**2) ** 2 / ln_ratio)


def leak_rate_liquid(
    aperture_m: float,
    r_casing_m: float,
    dp_pa: float,
    length_m: float,
    mu_pa_s: float,
    *,
    rho: float = 1000.0,
    inclination_deg: float = 90.0,
) -> float:
    """Liquid microannulus leak rate (m^3/s), gravity-corrected Hagen-Poiseuille.

    ``Q = (pi / (8 mu L)) * Omega * (dP - rho g L cos(theta))`` with ``theta``
    the well inclination from horizontal (90 deg = vertical).  Returns 0 when
    the applied pressure cannot overcome gravity (or the gap is closed).

    Sources: src2025_04/microannuli_leak_rate.
    """
    if aperture_m <= 0 or length_m <= 0:
        return 0.0
    g = 9.81
    theta = np.radians(inclination_deg)
    gravity_dp = rho * g * length_m * np.cos(theta)
    effective_dp = dp_pa - gravity_dp
    if effective_dp <= 0:
        return 0.0
    omega = microannulus_omega(r_casing_m, aperture_m)
    return float((np.pi / (8.0 * mu_pa_s * length_m)) * omega * effective_dp)


def leak_rate_gas(
    aperture_m: float,
    r_casing_m: float,
    p_in_pa: float,
    p_out_pa: float,
    length_m: float,
    mu_pa_s: float,
) -> float:
    """Gas microannulus leak rate (m^3/s at outlet), isothermal compressible flow.

    ``Q = (pi / (16 mu L P2)) * Omega * (P1^2 - P2^2)``; gravity neglected.

    Sources: src2025_04/microannuli_leak_rate.
    """
    if aperture_m <= 0 or length_m <= 0 or p_out_pa <= 0:
        return 0.0
    omega = microannulus_omega(r_casing_m, aperture_m)
    q = (np.pi / (16.0 * mu_pa_s * length_m * p_out_pa)) * omega * (p_in_pa**2 - p_out_pa**2)
    return float(max(q, 0.0))


def cubic_law_conductivity(
    aperture_m: ArrayLike, *, rho: float = 1000.0, mu: float = 1e-3, g: float = 9.81
) -> _Float:
    """Parallel-plate crack hydraulic conductivity ``K = rho g w^3 / (12 mu)`` (m/s).

    Sources: src2026_04/a11_awi_cement_evaluation.
    """
    return np.asarray(rho * g * _arr(aperture_m) ** 3 / (12.0 * mu))


# --- mud gas --------------------------------------------------------------------


def haworth_ratios(
    c1: float, c2: float, c3: float, c4: float, c5: float, *, percent: bool = True
) -> tuple[float, float, float]:
    """Haworth wetness / balance / character ratios ``(Wh, Bh, Ch)``.

    ``Wh = (C2+..+C5)/total`` (x100 if ``percent``), ``Bh = (C1+C2)/(C3+C4+C5)``,
    ``Ch = (C4+C5)/C3``.  Any consistent gas units.  Empty gas returns NaN
    wetness; zero denominators return +inf.

    Sources: src2026_06/a08_mud_gas_ratio_fluid_id.
    """
    total = c1 + c2 + c3 + c4 + c5
    if total <= 0.0:
        wh = float("nan")
    else:
        wh = (c2 + c3 + c4 + c5) / total * 100.0 if percent else (c2 + c3 + c4 + c5) / total
    heavy = c3 + c4 + c5
    bh = (c1 + c2) / heavy if heavy > 0.0 else float("inf")
    ch = (c4 + c5) / c3 if c3 > 0.0 else float("inf")
    return wh, bh, ch


def pixler_ratios(c1: float, c2: float, c3: float, c4: float, c5: float) -> dict[str, float]:
    """Pixler component ratios ``C1/C2 .. C1/C5`` plus the Bernard ratio ``C1/(C2+C3)``.

    Zero denominators return +inf.

    Sources: src2021_02/article1_mudlog_net_pay_tutorial.
    """
    c1f = float(c1)
    out: dict[str, float] = {}
    for name, x in (("C1/C2", c2), ("C1/C3", c3), ("C1/C4", c4), ("C1/C5", c5)):
        xf = float(x)
        out[name] = c1f / xf if xf > 0 else np.inf
    denom = float(c2) + float(c3)
    out["bernard"] = c1f / denom if denom > 0 else np.inf
    return out


def classify_fluid_haworth(
    wh: float, bh: float, ch: float | None = None, *, n_classes: int = 4
) -> str:
    """Classify reservoir fluid from Haworth ratios (``wh`` in percent).

    ``n_classes=4`` uses the classic bands (dry gas < 0.5 %, gas /
    gas-condensate < 17.5 % split on ``bh > wh``, oil < 40 %, else residual
    oil).  ``n_classes=8`` uses the extended 2026 scheme (requires ``ch``).

    Sources: src2021_02/article1_mudlog_net_pay_tutorial,
    src2026_06/a08_mud_gas_ratio_fluid_id.
    """
    if n_classes == 4:
        if wh < 0.5:
            return "dry gas"
        if wh < 17.5:
            return "gas" if bh > wh else "gas-condensate"
        if wh < 40:
            return "oil"
        return "residual oil"
    if n_classes == 8:
        if np.isnan(wh):
            return "no gas"
        if wh < 0.5:
            return "very dry gas"
        if wh < 5.0:
            return "dry gas" if bh > 100 else "wet gas"
        if wh < 17.5:
            return "gas condensate" if (ch is not None and ch >= 0.5 and bh < 30) else "wet gas"
        if wh < 25.0:
            return "volatile oil"
        if wh < 32.5:
            return "light oil"
        if wh < 40.0:
            return "medium/black oil"
        return "heavy/residual oil"
    raise ValueError(f"n_classes must be 4 or 8, got {n_classes}")


def normalize_gas(
    total_gas: ArrayLike,
    rop: ArrayLike,
    flow: ArrayLike,
    bit_diameter: ArrayLike,
    *,
    mud_weight: ArrayLike | None = None,
    units: str = "metric",
    reference: tuple[float, float, float] = (30.0, 500.0, 10.0),
) -> _Float:
    """Normalize a total-gas reading for drilling parameters.

    ``units='metric'`` returns the drilled-rock gas index ``G*Q/(ROP*A_bit)``
    (ROP m/hr, flow L/min, bit inches; ROP floored at 1e-6).  ``units='field'``
    divides the reading by the ratio-to-``reference`` factors
    ``(ROP/ROP0)*(Q/Q0)*(MW/MW0)^2`` (ft/hr, gpm, ppg; ``bit_diameter`` unused)
    to bring it to reference conditions.

    Sources: src2021_02/article1_mudlog_net_pay_tutorial, src2024_08/mudgas_response.
    """
    if units == "metric":
        area = np.pi * (_arr(bit_diameter) * 0.0254 / 2.0) ** 2
        rop_f = np.maximum(_arr(rop), 1e-6)
        return np.asarray(_arr(total_gas) * _arr(flow) / (rop_f * area))
    if units == "field":
        if mud_weight is None:
            raise ValueError("units='field' requires mud_weight (ppg)")
        rop_factor = _arr(rop) / reference[0]
        flow_factor = _arr(flow) / reference[1]
        mw_factor = (_arr(mud_weight) / reference[2]) ** 2
        correction = rop_factor * flow_factor * mw_factor
        return np.asarray(_arr(total_gas) / correction)
    raise ValueError(f"units must be 'metric' or 'field', got {units!r}")


# --- pressures / drilling window ------------------------------------------------


def hydrostatic_pressure(tvd_m: ArrayLike, *, rho: float = 1030.0) -> _Float:
    """Hydrostatic pressure ``rho * g * TVD`` (Pa), g = 9.80665."""
    return np.asarray(rho * G_STD * _arr(tvd_m))


def hydrostatic_pressure_psi(tvd_ft: ArrayLike, *, sg: float = 1.0) -> _Float:
    """Hydrostatic pressure in psi from TVD in ft: ``0.433 * SG * TVD``.

    Sources: src2014_04/article1_deepwater_gom_overview.
    """
    return np.asarray(PSI_PER_FT_PER_SG * sg * _arr(tvd_ft))


def hydrostatic_pressure_bar(tvd_m: ArrayLike, *, sg: float = 1.0) -> _Float:
    """Hydrostatic pressure in bar from TVD in m: ``0.0980665 * SG * TVD``.

    Sources: src2021_12/article08_overbalanced_drilling_correction.
    """
    return np.asarray(BAR_PER_SG_M * sg * _arr(tvd_m))


def overburden_pressure(
    tvd_m: ArrayLike, rho_bulk: ArrayLike, *, water_depth_m: float = 0.0, rho_sw: float = 1030.0
) -> _Float:
    """Overburden pressure (Pa): seawater column plus constant-density sediment.

    ``P = g * (rho_sw * water_depth + rho_bulk * tvd)`` with ``tvd`` below
    mudline and g = 9.80665.
    """
    return np.asarray(G_STD * (rho_sw * water_depth_m + _arr(rho_bulk) * _arr(tvd_m)))


def overburden_pressure_psi(
    water_depth_ft: ArrayLike,
    sediment_depth_ft: ArrayLike,
    *,
    sw_sg: float = 1.025,
    sediment_sg: float = 2.3,
) -> _Float:
    """Deepwater overburden in psi: ``0.433 * (SG_sw*wd + SG_sed*sd)``.

    Sources: src2014_04/article1_deepwater_gom_overview.
    """
    return np.asarray(
        PSI_PER_FT_PER_SG * (sw_sg * _arr(water_depth_ft) + sediment_sg * _arr(sediment_depth_ft))
    )


def eaton_pore_pressure(
    overburden: ArrayLike,
    hydrostatic: ArrayLike,
    observed: ArrayLike,
    normal: ArrayLike,
    *,
    exponent: float = 3.0,
    log_type: str = "sonic",
    clip_ratio: tuple[float, float] | None = None,
) -> _Float:
    """Eaton pore pressure ``Pp = OB - (OB - Pn) * ratio^exponent``.

    ``log_type='sonic'`` uses ``ratio = normal/observed`` (slowness; slow,
    undercompacted rock gives ratio < 1 and overpressure); ``'resistivity'``
    uses ``ratio = observed/normal``.  Works in pressure or gradient space (the
    output follows the input units).  ``clip_ratio`` optionally bounds the
    ratio (e.g. ``(0.01, 100.0)``).

    Sources: src2014_04/article5_deepwater_formation_evaluation,
    src2025_04/overpressure_isotope.
    """
    ob = _arr(overburden)
    hyd = _arr(hydrostatic)
    if log_type == "sonic":
        ratio = _arr(normal) / _arr(observed)
    elif log_type == "resistivity":
        ratio = _arr(observed) / _arr(normal)
    else:
        raise ValueError(f"log_type must be 'sonic' or 'resistivity', got {log_type!r}")
    if clip_ratio is not None:
        ratio = np.clip(ratio, clip_ratio[0], clip_ratio[1])
    return np.asarray(ob - (ob - hyd) * ratio**exponent)


def bowers_pore_pressure(
    velocity: ArrayLike,
    overburden: ArrayLike,
    *,
    A: float = 10.0,
    B: float = 0.7,
    unloading: bool = False,
    U: float = 3.0,
    v0: float = 5000.0,
    sigma_max: float = 5000.0,
) -> _Float:
    """Bowers pore pressure from sonic velocity (ft/s) and overburden (psi).

    Inverts the loading curve ``V = V0 + A*sigma^B`` for the effective stress
    (floored at 0.01) and subtracts it from the overburden; ``unloading=True``
    uses the flatter unloading branch with maximum stress ``sigma_max`` and
    exponent ``U``.

    Sources: src2025_04/overpressure_isotope.
    """
    v = _arr(velocity)
    ob = _arr(overburden)
    if not unloading:
        sigma_eff = np.clip((v - v0) / A, 0.01, None) ** (1.0 / B)
    else:
        sigma_eff = sigma_max * ((v - v0) / (A * sigma_max**B)) ** U
    return np.asarray(ob - sigma_eff)


def drilling_window_margin(pore: ArrayLike, frac: ArrayLike) -> _Float:
    """Drilling-window width ``frac - pore`` (same units as the inputs).

    Sources: src2014_04/article5_deepwater_formation_evaluation.
    """
    return np.asarray(_arr(frac) - _arr(pore))


def within_drilling_window(ecd: float, pore: float, frac: float) -> bool:
    """True when the ECD sits strictly inside the pore/frac window.

    Sources: src2014_04/article5_deepwater_formation_evaluation.
    """
    return bool(pore < ecd < frac)


# --- drilling fluid / mudcake ---------------------------------------------------


def mudcake_thickness(
    t_s: ArrayLike,
    *,
    k_mc_m2: float = 0.0,
    dp_pa: float = 0.0,
    mu_pa_s: float = 1e-3,
    solids_ratio: float = 0.6,
    model: str = "dewan",
    rate_const: float = 0.0,
    v: float = 0.5,
) -> _Float:
    """Mudcake thickness growth (m) in one of the corpus conventions.

    ``model='dewan'`` is the closed-form filtration solution
    ``h = sqrt(2 k dP t / (mu * S))`` with ``S = solids_ratio`` the cake solids
    term.  ``model='sqrt_k'`` is the empirical ``h = rate_const * sqrt(t)``.
    ``model='chin_ode'`` integrates the Chin thickness ODE with a
    pressure-decaying cake permeability ``K(t) = k_mc * ((t + 1e-3)/1e-3)^-v``
    (10-micron seed cake, ``solids_ratio`` is the mud solid fraction ``fs``).

    Sources: src2019_10/article6_microct_invasion_mudcake,
    src2022_04/article4_microct_filtercake, src2023_08/article2_invasion_simulation.
    """
    t = _arr(t_s)
    if model == "sqrt_k":
        return np.asarray(rate_const * np.sqrt(t))
    if model == "dewan":
        return np.asarray(np.sqrt(2.0 * k_mc_m2 * dp_pa * t / (mu_pa_s * solids_ratio)))
    if model == "chin_ode":
        t0 = 1.0e-3
        rmc = np.zeros_like(t)
        rmc[0] = 1.0e-5
        for i in range(1, t.size):
            dt = t[i] - t[i - 1]
            kmc = k_mc_m2 * np.power((t[i] + t0) / t0, -v)
            rmc[i] = np.sqrt(rmc[i - 1] ** 2 + 2.0 * solids_ratio * kmc * dp_pa * dt / mu_pa_s)
        return rmc
    raise ValueError(f"model must be 'sqrt_k', 'dewan' or 'chin_ode', got {model!r}")
