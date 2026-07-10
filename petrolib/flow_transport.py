"""Single-phase flow, poro-permeability transforms, rock typing, and diffusion.

Darcy's law, Klinkenberg gas slippage, stress-dependent permeability, the
Kozeny-Carman / Winland / Swanson / RQI-FZI poro-perm transforms, permeability
averaging, and the molecular-diffusion group are each re-implemented across the
article corpus.  This module is the one canonical home; facades keep their
article's unit adapters and defaults.

Two-phase relative permeability, fractional flow, and the capillary/Bond
numbers live in :mod:`petrolib.relperm_wettability`; the NMR permeability
transforms (Timur-Coates, SDR, Timur) belong to the ``nmr`` domain.

Hazards this module's API is designed around (LIBRARY_MERGE_PLAN.md
section 9):

- **Klinkenberg apparent vs corrected.**  Nine files compute the *apparent*
  gas permeability ``k_app = k_inf*(1 + b/P_mean)``; src2021_12 computes the
  *inverse* (liquid-equivalent ``k_inf`` from a measured ``k_app``).  These are
  distinct functions — :func:`klinkenberg_apparent` and
  :func:`klinkenberg_corrected` — never one name with a flag.
- **Mean-free-path physics.**  The kinetic-theory form
  ``kB*T/(sqrt(2)*pi*d**2*P)`` (collision diameter) and the viscosity form
  ``(mu/P)*sqrt(pi*R*T/(2*M))`` are different inputs; :func:`mean_free_path`
  branches on which keywords are supplied and never mixes them.
- **Kozeny-Carman branch and grain term.**  The surface-area form varies in
  whether it carries a tortuosity ``tau**2`` and a grain-volume ``(1-phi)**2``
  factor, and articles emit m2, um2 or mD.  :func:`kozeny_carman` takes ``tau``,
  ``c`` and ``grain_term`` explicitly and returns m2 (specific surface in 1/m).
- **Winland phi scaling.**  All nine article copies take phi as a *fraction*
  and multiply by 100 internally (the 1972/1992 correlation is in percent).
  :func:`winland_r35` keeps that convention — pass phi as a fraction.
- **RQI unit constant.**  ``RQI = 0.0314*sqrt(k_md/phi)`` — the 0.0314 (=
  ``sqrt(1/1014)``) converts mD to um so RQI is in um; it is the explicit ``c``
  parameter so the equivalent ``sqrt(k/(1014*phi))`` callers map exactly.
- **Diffusion geometry factor.**  ``L = sqrt(f*D*t)`` uses ``f=1`` in seven
  files and ``f=2`` in one; :func:`diffusion_length` / :func:`diffusion_time`
  take ``geometry_factor`` explicitly.

Units: SI unless a name says otherwise.  Permeability is millidarcy (``k_md``)
in the empirical poro-perm / rock-typing transforms — that is how their
correlation constants are defined — and square metres in the Darcy-law and
Kozeny-Carman functions; porosity is a fraction, RQI/r35 in um.  numpy-only at
import; the ``fit_*`` routines use ``numpy.polyfit``.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2014_04/article3 -- Article 3: The Dynamics of Reservoir Fluids and their Substantial Systematic
  Variations Oliver C. Mullins, Julian Y. Zuo, Kang Wang, Paul S. Hammond, Ilaria De Santo, Hadrien
  Dumont, Vinay K. Mishra, Li Chen, Andrew E. Pomerantz, Chengli Dong, Hani Elshahawi, Douglas J.
  Seifert (2014). Petrophysics Vol. 55, No. 2 (April 2014), pp. 96-112. DOI: none assigned (this
  issue predates SPWLA DOI assignment).
src2014_10/article5 -- Article 5: Quantifying the Impact of Petrophysical Properties on Spatial
  Distribution of Contrasting Nanoparticle Agents in the Near-Wellbore Region. Kai Cheng, Aderonke
  Aderibigbe, Masoud Alfi, Zoya Heidari, John Killough (2014). Petrophysics Vol. 55, No. 5 (October
  2014), pp. 447-460. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2015_02/article2 -- Article 2: CO2 EOR by Diffusive Mixing in Fractured Reservoirs. Eide,
  Ersland, Brattekas, Haugen, Graue, Ferno (2015). Petrophysics Vol. 56, No. 1 (February 2015), pp.
  23-31. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2015_02/article4 -- Article 4: Petrophysical Characterization of the Pore Space in Permian
  Wolfcamp Rocks. Rafatian, Capsan (2015). Petrophysics Vol. 56, No. 1 (February 2015), pp. 45-57.
  DOI: none assigned (this issue predates SPWLA DOI assignment).
src2015_04/article2 -- Article 2: Steady-State Stress-Dependent Permeability Measurements of Tight
  Oil-Bearing Rocks. Chhatre, Braun, Sinha, Determan, Passey, Zirkle, Wood, Boros, Berry, Leonardi,
  Kudva (2015). Petrophysics Vol. 56, No. 2 (April 2015), pp. 116-124. DOI: none assigned (this
  issue predates SPWLA DOI assignment).
src2015_04/article3 -- Article 3: Estimation of Permeability in the McMurray Formation Using High-
  Resolution Data Sources. Manchuk, Garner, Deutsch (2015). Petrophysics Vol. 56, No. 2 (April
  2015), pp. 125-139. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2015_08/article3 -- Article 3: Gas Diffusion into Oil, Reservoir Baffling and Tar Mats Analyzed
  by Downhole Fluid Analysis, Pressure Transients, Core Extracts and DSTs. Achourov, Pfeiffer,
  Kollien, Betancourt, Zuo, di Primio, Mullins (2015). Petrophysics Vol. 56, No. 4 (August 2015),
  pp. 346-357. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2015_10/article2 -- Article 2: Differing Equilibration Times of GOR, Asphaltenes and Biomarkers
  as Determined by Charge History and Reservoir Fluid Geodynamics. Wang, Kauerauf, Zuo, Chen, Dong,
  Elshahawi, Mullins (2015). Petrophysics Vol. 56, No. 5 (October 2015), pp. 440-456. DOI: none
  assigned (this issue predates SPWLA DOI assignment).
src2015_10/article5 -- Article 5: Integrated Petrophysical Rock Classification in the McElroy
  Field, West Texas, USA. Saneifar, Skalinski, Theologou, Kenter, Cuffey, Salazar-Tio (2015).
  Petrophysics Vol. 56, No. 5 (October 2015), pp. 493-510. DOI: none assigned (this issue predates
  SPWLA DOI assignment).
src2015_12/article4 -- Article 4: Petrophysical Challenges in Giant Carbonate Tengiz Field,
  Republic of Kazakhstan. Skalinski, Se, Playton, Theologou, Narr, Sullivan, Mallan (2015).
  Petrophysics Vol. 56, No. 6 (December 2015), pp. 615-647. DOI: none assigned (this issue predates
  SPWLA DOI assignment).
src2016_02/article3 -- Article 3: Low-Permeability Measurements: Insights. Profice, Hamon, Nicot
  (2016). Petrophysics Vol. 57, No. 1 (February 2016), pp. 30-40. DOI: none assigned (this issue
  predates SPWLA DOI assignment).
src2016_06/article1 -- Article 1: Heterogeneous Carbonate Reservoirs: Ensuring Consistency of
  Subsurface Models by Maximizing the use of Saturation-Height Models and Dynamic Data. Hulea,
  Frese, Ramaswami (2016). Petrophysics Vol. 57, No. 3 (June 2016), pp. 223-232. DOI: none assigned
  (this issue predates SPWLA DOI assignment).
src2017_02/article3 -- Article 3: Wettability Effects on Osmosis as an Oil-Mobilization Mechanism
  During Low-Salinity Waterflooding. Fredriksen, Rognmo, Sandengen, Ferno (2017). Petrophysics Vol.
  58, No. 1 (February 2017), pp. 28-35. DOI: none assigned (this issue predates SPWLA DOI
  assignment).
src2017_02/article5 -- Article 5: Comparison of Four Numerical Simulators for SCAL Experiments.
  Lenormand, Lorentzen, Maas, Ruth (2017). Petrophysics Vol. 58, No. 1 (February 2017), pp. 48-56.
  DOI: none assigned (this issue predates SPWLA DOI assignment).
src2018_02/article3 -- Article 3: Stress Sensitivity of Mercury-Injection Measurements. Guise,
  Grattoni, Allshorn, Fisher, Schiffer (2018). DOI: 10.30632/petro_059_1_a2. Petrophysics Vol. 59
  No. 1 (Feb 2018).
src2018_02/article4 -- Article 4: Microstructural Investigation of Stress-Dependent Permeability in
  Tight-Oil Rocks. King, Sansone, Kortunov, Xu, Callen, Chhatre, Sahoo, Buono (2018). DOI:
  10.30632/petro_059_1_a3. Petrophysics Vol. 59 No. 1 (Feb 2018).
src2018_06/article8 -- Article 8: Saturation-Height Modeling: Assessing Capillary Pressure Stress
  Corrections. Hulea (2018). DOI: 10.30632/PJV59N3-2018a7 (inferred - see note). Petrophysics Vol.
  59 No. 3 (Jun 2018).
src2018_08/article8 -- Article 8: Incorporating Flow Regimes Into Crushed-Rock Analysis to Better
  Understand Matrix Permeability and Pore Structure in Shales. Royer, Hobbs, Bonar (2018). DOI:
  10.30632/PJV59V4-2018a7. Petrophysics Vol. 59 No. 4 (Aug 2018) - Special Issue on Flow
  Diagnostics.
src2019_02/article2 -- Article 2: Defining Net-Pay Cutoffs in Carbonates Using Advanced
  Petrophysical Methods. Skalinski, Mallan, Edwards, Sun, Toumelin, Kelly, Wushur, Sullivan (2019).
  DOI: 10.30632/PJV60N1Y2019a1. Petrophysics Vol. 60 No. 1 (Feb 2019).
src2019_04/article11 -- Article 11: Loading Effects on Gas Relative Permeability of a Low-
  Permeability Sandstone. Agostini, Egermann, Jeannin, Portier, Skoczylas, Wang (2019). DOI:
  10.30632/PJV60N2-2019a9. Petrophysics Vol. 60 No. 2 (Apr 2019).
src2019_06/article4 -- Article 4: Finite-Volume Computations of Shale Tortuosity and Permeability
  From 3D Pore Networks Extracted From Scanning Electron Tomographic Images. Almasoodi, Reza
  (2019). DOI: 10.30632/PJV60N3-2019a3. Petrophysics Vol. 60 No. 3 (Jun 2019).
src2019_06/article6 -- Article 6: Reconsidering Klinkenberg's Permeability Data. Ruth, Arabjamaloei
  (2019). DOI: 10.30632/PJV60N3-2019a5. Petrophysics Vol. 60 No. 3 (Jun 2019).
src2019_10/article8 -- Article 8: Presenting a Multifaceted Approach to Unconventional Rock Typing
  and Technical Validation - Case Study in the Permian Basin and Impacts on Reservoir
  Characterization Workflows. Perry, Hayes (2019). DOI: 10.30632/PJV60N5-2019a8. Petrophysics Vol.
  60 No. 5 (Oct 2019).
src2019_12/article10 -- Article 10: Review of Micro/Nanofluidic Insights on Fluid Transport
  Controls in Tight Rocks. Mehmani, Kelly, Torres-Verdin (2019). DOI: 10.30632/PJV60N6-2019a10.
  Petrophysics Vol. 60 No. 6 (Dec 2019).
src2020_04/article3 -- Article 3: Low-Permeability Measurement on Crushed Rock: Insights. Profice,
  Lenormand (2020). DOI: 10.30632/PJV61N2-2020a3. Petrophysics Vol. 61 No. 2 (Apr 2020).
src2020_04/article9 -- Article 9: Chemically Induced Formation Damage in Shale. Wick, Taneja,
  Gupta, Sondergeld, Rai (2020). DOI: 10.30632/PJV61N2-2020a9. Petrophysics Vol. 61 No. 2 (Apr
  2020).
src2020_06/article3 -- Article 3: Reliable Measurement of Saturation-Dependent Relative
  Permeability in Tight Gas Sand Formations. Gonzalez, Tandon, Heidari, Gramin, Merle (2020). DOI:
  10.30632/PJV61N3-2020a3. Petrophysics Vol. 61 No. 3 (Jun 2020).
src2021_12 -- Petrophysics Vol. 62 No. 6 (Dec 2021) (issue-level reference).
src2021_12/article08 -- Article 8: The Impact of Overbalanced Drilling From Exploration/Appraisal
  Wells to Field Development Plan. Mohammadlou, Reppert, Del Negro, Jones (2021). DOI:
  10.30632/PJV62N6-2021a8. Petrophysics Vol. 62 No. 6 (Dec 2021).
src2022_04/article2 -- Article 2: Permeability Modeling in Clay-Rich Carbonate Reservoir. Storebo,
  Meireles, Fabricius (2022). DOI: 10.30632/PJV63N2-2022a2. Petrophysics Vol. 63 No. 2 (Apr 2022).
src2022_04/article4 -- Article 4: In-Situ Visualization and Characterization of Filter-Cake
  Deposition Using Time-Lapse Micro-CT Imaging. Schroeder, Torres-Verdin (2022). DOI:
  10.30632/PJV63N2-2022a4. Petrophysics Vol. 63 No. 2 (Apr 2022).
src2022_12/article7 -- Article 7: Using Digital Rock Physics to Evaluate Novel Percussion Core
  Quality. Lakshtanov, Zapata, Saucier, Cook, Eve, Lancaster, Lane, Gettemy, Sincock, Liu, Geetan,
  Draper, Gill (2022). DOI: 10.30632/PJV63N6-2022a7. Petrophysics Vol. 63 No. 6 (Dec 2022) — Best
  Papers of the 2022 SPWLA Annual Symposium special issue.
src2023_02/article2 -- Article 2: Modeling Permeability in Different Carbonate Rock Types.
  Dernaika, Masalmeh, Mansour, Al Jallad, Koronfol (2023). DOI: 10.30632/PJV64N1-2023a2.
  Petrophysics Vol. 64 No. 1 (Feb 2023).
src2023_10/article_11 -- Shafiq, M. U., Ben Mahmud, H., Khan, M., Gishkori, S. N., Wang, L., and
  Jamil, M. (2023). "Effect of Chelating Agents on Tight Sandstone Formation Mineralogy During
  Sandstone Acidizing." Petrophysics, 64(5), 796-817. DOI: 10.30632/PJV64N5-2023a11.
src2024_10/permeability_anisotropy -- Permeability Anisotropy in Carbonates via Digital Rock
  Petrophysics. Based on: Silva Junior et al. (2024), "Permeability Anisotropy in Brazilian Presalt
  Carbonates at Core Scale Using Digital Rock Petrophysics", Petrophysics, Vol. 65, No. 5, pp.
  711-738.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import KB, R_GAS

_Float = NDArray[np.float64]

_TINY = 1e-30  # zero-denominator guard


# ==========================================================================
# 1. Single-phase Darcy & gas flow
# ==========================================================================


def darcy_permeability(q: ArrayLike, *, mu: float, length: float, area: float, dp: float) -> _Float:
    """Darcy permeability of a core plug ``k = q*mu*L/(A*dP)``.

    Incompressible (liquid) steady-state Darcy's law.  Any consistent unit set
    works; SI gives k in m2.  Sources: src2015_04/article2 (darcy_permeability),
    src2016_02/article3 (darcy_liquid_permeability).
    """
    return np.asarray(np.asarray(q, np.float64) * mu * length / (area * dp))


def darcy_rate(k: ArrayLike, *, area: float, dp: float, mu: float, length: float) -> _Float:
    """Darcy volumetric flow rate ``q = k*A*dP/(mu*L)`` (the inverse of
    :func:`darcy_permeability`).  Sources: src2018_08/article8 (darcy_rate).
    """
    return np.asarray(np.asarray(k, np.float64) * area * dp / (mu * length))


def darcy_pressure_drop(
    q: ArrayLike, *, mu: float, k: float, area: float, length: float, kr: float = 1.0
) -> _Float:
    """Pressure drop for Darcy flow ``dP = q*mu*L/(k*kr*A)``.

    ``kr`` is the relative permeability of the flowing phase (1 for
    single-phase).  Sources: src2017_02/article5 (darcy_pressure_drop).
    """
    return np.asarray(np.asarray(q, np.float64) * mu * length / (k * kr * area))


def darcy_gas_permeability(
    q: ArrayLike,
    *,
    mu: float,
    length: float,
    area: float,
    p_up: float,
    p_down: float,
    p_ref: float,
) -> _Float:
    """Compressible (gas) Darcy permeability, the pressure-squared form
    ``k = 2*q*mu*L*P_ref / (A*(P_up**2 - P_down**2))``.

    The isothermal integration of Darcy's law for an ideal gas; ``q`` is the
    volumetric rate measured at ``p_ref`` (e.g. outlet/atmospheric).  SI gives
    k in m2.  Sources: src2016_02/article3 (darcy_gas_permeability).
    """
    q_arr = np.asarray(q, np.float64)
    return np.asarray(2.0 * q_arr * mu * length * p_ref / (area * (p_up**2 - p_down**2)))


# ==========================================================================
# 2. Klinkenberg gas slippage & Knudsen regime
# ==========================================================================


def klinkenberg_apparent(
    k_inf: ArrayLike, *, b: float, p_mean: ArrayLike, c2: float = 0.0
) -> _Float:
    """Apparent gas permeability ``k_app = k_inf*(1 + b/P + c2/P**2)``.

    The first-order Klinkenberg (1941) slip correction; ``c2`` adds the
    second-order term (default 0).  ``b`` and ``P`` share units.  Sources:
    src2016_02/article3, src2018_02/article4, src2019_06/article6 (+ c2 term),
    src2020_04/article3, src2020_06/article3.
    """
    k = np.asarray(k_inf, np.float64)
    p = np.asarray(p_mean, np.float64)
    return np.asarray(k * (1.0 + b / p + c2 / p**2))


def klinkenberg_corrected(k_app: ArrayLike, *, b: float, p_mean: ArrayLike) -> _Float:
    """Liquid-equivalent permeability ``k_inf = k_app/(1 + b/P)`` — the explicit
    inverse of :func:`klinkenberg_apparent` (no c2 term).

    Distinct from the forward slip law: this *corrects* a measured apparent gas
    permeability back to the intrinsic value.  Sources: src2021_12/article08
    (overbalanced-drilling correction, pressure in bar).
    """
    k = np.asarray(k_app, np.float64)
    p = np.asarray(p_mean, np.float64)
    return np.asarray(k / (1.0 + b / p))


def fit_klinkenberg(p_mean: ArrayLike, k_app: ArrayLike) -> tuple[float, float]:
    """Fit ``(k_inf, b)`` from the Klinkenberg plot of ``k_app`` vs ``1/P``.

    ``k_app = k_inf + (k_inf*b)*(1/P)`` — the intercept is ``k_inf`` and the
    slope is ``k_inf*b``, so ``b = slope/intercept``.  Sources:
    src2016_02/article3 (klinkenberg_fit), src2019_06/article6 (fit_first_order),
    src2020_04/article3 (klinkenberg_extrapolate).
    """
    x = 1.0 / np.asarray(p_mean, np.float64)
    slope, intercept = np.polyfit(x, np.asarray(k_app, np.float64), 1)
    return float(intercept), float(slope / intercept)


def mean_free_path(
    *,
    pressure: ArrayLike,
    temperature: float,
    mu: float | None = None,
    molar_mass: float | None = None,
    d_collision: float | None = None,
) -> _Float:
    """Gas mean free path, in one of two physically distinct forms.

    Viscosity form (pass ``mu`` and ``molar_mass`` in kg/mol):
    ``lambda = (mu/P)*sqrt(pi*R*T/(2*M))``.  Kinetic-theory form (pass
    ``d_collision``, the molecular collision diameter):
    ``lambda = kB*T/(sqrt(2)*pi*d**2*P)``.  SI throughout.  Sources:
    src2016_02/article3 (viscosity), src2018_08/article8 and
    src2019_12/article10 (kinetic theory).
    """
    p = np.asarray(pressure, np.float64)
    if mu is not None and molar_mass is not None:
        return np.asarray((mu / p) * np.sqrt(np.pi * R_GAS * temperature / (2.0 * molar_mass)))
    if d_collision is not None:
        return np.asarray(KB * temperature / (np.sqrt(2.0) * np.pi * d_collision**2 * p))
    raise ValueError("mean_free_path needs either (mu, molar_mass) or d_collision")


def knudsen_number(mfp: ArrayLike, pore_size: ArrayLike) -> _Float:
    """Knudsen number ``Kn = lambda/L_pore``.

    ``pore_size`` is the characteristic length the article uses — pore radius in
    some copies, diameter in others (a factor-2 choice the caller owns).
    Sources: src2016_02/article3 (radius), src2018_08/article8 (diameter).
    """
    return np.asarray(np.asarray(mfp, np.float64) / np.asarray(pore_size, np.float64))


def flow_regime(kn: float, *, cuts: tuple[float, float, float] = (1e-3, 0.1, 10.0)) -> str:
    """Gas flow regime from the Knudsen number: ``"continuum"`` (Kn < cuts[0]),
    ``"slip"`` (< cuts[1]), ``"transition"`` (< cuts[2]), else
    ``"free-molecular"``.  Sources: src2018_08/article8, src2019_12/article10.
    """
    if kn < cuts[0]:
        return "continuum"
    if kn < cuts[1]:
        return "slip"
    if kn < cuts[2]:
        return "transition"
    return "free-molecular"


# ==========================================================================
# 3. Stress-dependent permeability
# ==========================================================================


def stress_permeability(
    k0: ArrayLike, *, gamma: float, ncs: ArrayLike, ncs0: float = 0.0
) -> _Float:
    """Exponential permeability decline ``k = k0*exp(-gamma*(NCS - NCS0))``.

    ``k0`` is the permeability at the reference net confining stress ``NCS0``
    (0 for the plain ``k0*exp(-gamma*NCS)`` form).  Sources: src2015_04/article2
    (NCS0=0), src2018_02/article4 (reference stress), src2019_04/article11,
    src2018_06/article8.
    """
    return np.asarray(
        np.asarray(k0, np.float64) * np.exp(-gamma * (np.asarray(ncs, np.float64) - ncs0))
    )


def net_confining_stress(
    total_stress: ArrayLike, *, pore_pressure: ArrayLike, biot: float = 1.0
) -> _Float:
    """Terzaghi/Biot net confining stress ``sigma_eff = sigma_total - biot*Pp``.

    Sources: src2015_04/article2, src2019_04/article11 (effective_stress);
    src2018_02/article4 uses biot=1 implicitly.
    """
    return np.asarray(
        np.asarray(total_stress, np.float64) - biot * np.asarray(pore_pressure, np.float64)
    )


def fit_stress_permeability(ncs: ArrayLike, k: ArrayLike) -> tuple[float, float]:
    """Fit ``(k0, gamma)`` from measured ``k`` vs net confining stress, a line
    in ``(NCS, ln k)``: ``k0 = exp(intercept)``, ``gamma = -slope``.  Sources:
    src2015_04/article2 (fit_stress_permeability).
    """
    slope, intercept = np.polyfit(np.asarray(ncs, np.float64), np.log(np.asarray(k, np.float64)), 1)
    return float(np.exp(intercept)), float(-slope)


# ==========================================================================
# 4. Poro-permeability transforms
# ==========================================================================


def kozeny_carman(
    phi: ArrayLike,
    *,
    specific_surface: ArrayLike,
    tau: float = 1.0,
    c: float = 5.0,
    grain_term: bool = True,
) -> _Float:
    """Kozeny-Carman permeability from porosity and specific surface area.

    ``k = phi**3 / (c*tau**2*S**2*(1-phi)**2)`` with ``grain_term=True`` (the
    classic grain-volume form), or ``k = phi**3/(c*tau**2*S**2)`` with
    ``grain_term=False``.  ``S`` is the specific surface per unit volume (1/m);
    k is returned in m2.  The Kozeny constant ``c`` (~2-5) and tortuosity
    ``tau`` are explicit.  Sources: src2019_06/article4 (c=2, tau, no grain
    term), src2022_04/article2, src2022_12/article7 (grain term, mD adapter).
    """
    p = np.asarray(phi, np.float64)
    s = np.asarray(specific_surface, np.float64)
    denom = c * tau**2 * s**2
    if grain_term:
        denom = denom * (1.0 - p) ** 2
    return np.asarray(p**3 / denom)


def kozeny_carman_ratio(
    k0: ArrayLike, phi: ArrayLike, phi0: float, *, grain_term: bool = True
) -> _Float:
    """Kozeny-Carman porosity-permeability sensitivity (ratio/update form).

    ``k = k0*(phi/phi0)**3*((1-phi0)/(1-phi))**2`` with ``grain_term=True``, or
    the simplified ``k = k0*(phi/phi0)**3`` with ``grain_term=False``.  Sources:
    src2022_04/article4, src2023_10/article_11 (grain term); src2020_04/article9
    (simplified).
    """
    p = np.asarray(phi, np.float64)
    ratio = np.asarray(k0, np.float64) * (p / phi0) ** 3
    if grain_term:
        ratio = ratio * ((1.0 - phi0) / (1.0 - p)) ** 2
    return np.asarray(ratio)


def winland_r35(k_md: ArrayLike, phi: ArrayLike) -> _Float:
    """Winland-Kolodzie r35 pore-throat radius (um) at 35% mercury saturation.

    ``log10(r35) = 0.732 + 0.588*log10(k) - 0.864*log10(phi_pct)`` — k in mD,
    phi a *fraction* (multiplied by 100 internally; the correlation is in
    percent).  Sources: src2015_02/article4, src2023_02/article2,
    src2019_10/article8 (nine copies, identical coefficients).
    """
    k = np.asarray(k_md, np.float64)
    phi_pct = np.asarray(phi, np.float64) * 100.0
    return np.asarray(10.0 ** (0.732 + 0.588 * np.log10(k) - 0.864 * np.log10(phi_pct)))


def winland_permeability(r35: ArrayLike, phi: ArrayLike) -> _Float:
    """Permeability (mD) from Winland r35 and porosity — the inverse of
    :func:`winland_r35`: ``log10(k) = (log10(r35) - 0.732 +
    0.864*log10(phi_pct))/0.588`` (phi a fraction).  Sources:
    src2015_02/article4 (winland_permeability), src2023_02/article2 (winland_k).
    """
    phi_pct = np.asarray(phi, np.float64) * 100.0
    return np.asarray(
        10.0
        ** ((np.log10(np.asarray(r35, np.float64)) - 0.732 + 0.864 * np.log10(phi_pct)) / 0.588)
    )


def swanson_permeability(sb_pc_apex: ArrayLike, *, c: float = 399.0, d: float = 1.691) -> _Float:
    """Swanson (1981) permeability from the MICP apex ``k = c*(Sb/Pc)_apex**d``.

    ``(Sb/Pc)_apex`` is the maximum bulk-mercury-saturation / injection-pressure
    ratio; k in mD.  The Wolfcamp copy uses ``c=399``, the stress-MICP copy
    ``c=339`` (pass ``c=339``).  Sources: src2015_02/article4 (c=399),
    src2018_02/article3 (c=339).
    """
    return np.asarray(c * np.asarray(sb_pc_apex, np.float64) ** d)


def micp_apex(shg: ArrayLike, pc: ArrayLike) -> tuple[_Float, int]:
    """Pittman (1992) MICP apex: the maximum ``Shg/Pc`` ratio and its index.

    Returns ``(apex_ratio, index)`` — the apex feeds
    :func:`swanson_permeability`.  Sources: src2018_02/article3 (micp_apex).
    """
    ratio = np.asarray(shg, np.float64) / np.asarray(pc, np.float64)
    i = int(np.argmax(ratio))
    return np.asarray(ratio[i]), i


def lucia_permeability(phi_g: ArrayLike, rfn: ArrayLike) -> _Float:
    """Lucia (2007) rock-fabric-number permeability transform.

    ``log10(k) = (9.7982 - 12.0838*log10(RFN)) +
    (8.6711 - 8.2965*log10(RFN))*log10(phi_g)`` — ``phi_g`` interparticle
    porosity (fraction), ``RFN`` the rock-fabric number; k in mD.  Sources:
    src2023_02/article2 (lucia_k).
    """
    r = np.asarray(rfn, np.float64)
    a = 9.7982 - 12.0838 * np.log10(r)
    b = 8.6711 - 8.2965 * np.log10(r)
    return np.asarray(10.0 ** (a + b * np.log10(np.asarray(phi_g, np.float64))))


def lucia_rfn_from_swi(phi_g: ArrayLike, swi: ArrayLike) -> _Float:
    """Lucia rock-fabric number from interparticle porosity and irreducible
    water saturation: ``log(RFN) = (3.1107 + 1.8834*log(phi_g) + log(Swi)) /
    (3.0634 + 1.4045*log(phi_g))``.  Sources: src2023_02/article2.
    """
    pg = np.asarray(phi_g, np.float64)
    num = 3.1107 + 1.8834 * np.log10(pg) + np.log10(np.asarray(swi, np.float64))
    den = 3.0634 + 1.4045 * np.log10(pg)
    return np.asarray(10.0 ** (num / den))


def poroperm_powerlaw(phi: ArrayLike, *, a: float, b: float) -> _Float:
    """Log-log poro-perm power law ``k = 10**(a + b*log10(phi))``.

    The ``a + b*log10(phi)`` (log-log) trend; k in mD, phi a fraction.  For the
    semilog ``10**(a*phi + b)`` trend see the article facades directly.  Sources:
    src2019_02/article2 (lucia_permeability A+B log-log), src2016_06/article1.
    """
    return np.asarray(10.0 ** (a + b * np.log10(np.asarray(phi, np.float64))))


def fit_poroperm(phi: ArrayLike, k_md: ArrayLike) -> tuple[float, float]:
    """Fit ``(a, b)`` for :func:`poroperm_powerlaw` by least squares of
    ``log10(k)`` on ``log10(phi)``.  Sources: log-log trend of
    src2019_02/article2, src2016_06/article1.
    """
    b, a = np.polyfit(
        np.log10(np.asarray(phi, np.float64)), np.log10(np.asarray(k_md, np.float64)), 1
    )
    return float(a), float(b)


# ==========================================================================
# 5. Rock typing & upscaling (RQI / FZI / HFU, averaging)
# ==========================================================================


def rqi(k_md: ArrayLike, phi: ArrayLike, *, c: float = 0.0314) -> _Float:
    """Amaefule Reservoir Quality Index ``RQI = c*sqrt(k_md/phi)`` (um).

    ``c = 0.0314 = sqrt(1/1014)`` converts mD to um; copies that write
    ``sqrt(k/(1014*phi))`` map by passing ``c=(1/1014)**0.5``.  Sources:
    src2023_02/article2, src2015_10/article5, src2019_10/article8.
    """
    return np.asarray(c * np.sqrt(np.asarray(k_md, np.float64) / np.asarray(phi, np.float64)))


def phi_z(phi: ArrayLike) -> _Float:
    """Normalized porosity index ``phi_z = phi/(1 - phi)``.  Sources:
    src2023_02/article2 (npi), src2015_10/article5, src2019_10/article8.
    """
    p = np.asarray(phi, np.float64)
    return np.asarray(p / (1.0 - p))


def fzi(k_md: ArrayLike, phi: ArrayLike, *, c: float = 0.0314) -> _Float:
    """Amaefule Flow Zone Indicator ``FZI = RQI/phi_z`` (um).  Sources:
    src2023_02/article2, src2015_10/article5, src2024_10/permeability_anisotropy.
    """
    return np.asarray(rqi(k_md, phi, c=c) / phi_z(phi))


def k_from_fzi(phi: ArrayLike, fzi_val: ArrayLike, *, c: float = 0.0314) -> _Float:
    """Permeability (mD) from FZI and porosity — the inverse of :func:`fzi`:
    ``k = fzi**2*phi**3 / (c**2*(1-phi)**2)``.

    Consistent with :func:`rqi`/:func:`fzi` for a shared ``c``.  Sources:
    src2015_12/article4 (permeability_from_fzi), src2023_02/article2 (1014 form,
    ~2e-4 from 1/c**2).
    """
    p = np.asarray(phi, np.float64)
    return np.asarray(np.asarray(fzi_val, np.float64) ** 2 * p**3 / (c**2 * (1.0 - p) ** 2))


def classify_hfu(fzi_values: ArrayLike, *, n_units: int = 4) -> tuple[NDArray[np.intp], _Float]:
    """Classify samples into hydraulic flow units by log-spaced FZI thresholds.

    Returns ``(labels, thresholds)`` — integer labels in ``[0, n_units-1]`` and
    the ``n_units+1`` FZI threshold values.  Sources:
    src2024_10/permeability_anisotropy (classify_hfu).
    """
    f = np.asarray(fzi_values, np.float64)
    log_f = np.log10(np.maximum(f, 1e-6))
    thresholds = np.linspace(log_f.min(), log_f.max(), n_units + 1)
    labels = np.digitize(log_f, thresholds[1:-1])
    return np.asarray(labels), np.asarray(10.0**thresholds)


def permeability_average(
    k: ArrayLike, *, method: str = "geometric", weights: ArrayLike | None = None
) -> float:
    """Average permeability by ``method`` = ``"arithmetic"`` (parallel flow),
    ``"harmonic"`` (series flow), or ``"geometric"`` (default).

    Optional ``weights`` give thickness/volume-weighted means.  Sources:
    src2016_06/article1, src2015_04/article3 (permeability_average),
    src2024_10/permeability_anisotropy (weighted).
    """
    k_arr = np.asarray(k, np.float64)
    w = None if weights is None else np.asarray(weights, np.float64)
    if method == "arithmetic":
        return float(np.mean(k_arr) if w is None else np.average(k_arr, weights=w))
    if method == "harmonic":
        if w is None:
            return float(len(k_arr) / np.sum(1.0 / k_arr))
        return float(np.sum(w) / np.sum(w / k_arr))
    if method == "geometric":
        if w is None:
            return float(np.exp(np.mean(np.log(k_arr))))
        return float(np.exp(np.average(np.log(k_arr), weights=w)))
    raise ValueError(f"method must be 'arithmetic', 'harmonic' or 'geometric', got {method!r}")


def wiener_bounds(k: ArrayLike) -> tuple[float, float]:
    """Wiener (harmonic-arithmetic) bounds on the effective permeability of a
    heterogeneous medium: ``(k_harmonic, k_arithmetic)`` — the series lower and
    parallel upper bounds.  Sources: src2015_04/article3
    (effective_permeability_bounds).
    """
    return (
        permeability_average(k, method="harmonic"),
        permeability_average(k, method="arithmetic"),
    )


# ==========================================================================
# 6. Diffusion & transport
# ==========================================================================


def diffusion_length(D: ArrayLike, t: ArrayLike, *, geometry_factor: float = 1.0) -> _Float:
    """Characteristic diffusion length ``L = sqrt(f*D*t)``.

    ``geometry_factor`` ``f`` is 1 in most copies and 2 in the
    ``sqrt(2*D*t)`` displacement form.  Sources: src2015_02/article2 (f=1),
    src2014_04/article3 (f=2).
    """
    return np.asarray(
        np.sqrt(geometry_factor * np.asarray(D, np.float64) * np.asarray(t, np.float64))
    )


def diffusion_time(L: ArrayLike, D: ArrayLike, *, geometry_factor: float = 1.0) -> _Float:
    """Diffusion / equilibration time ``t = L**2/(f*D)`` (inverse of
    :func:`diffusion_length`).  ``geometry_factor=2`` gives ``L**2/(2*D)``;
    a sealed-column mixing time uses ``geometry_factor=pi**2``.  Sources:
    src2015_08/article3, src2015_10/article2 (f=1), src2014_04/article3 (f=2).
    """
    return np.asarray(
        np.asarray(L, np.float64) ** 2 / (geometry_factor * np.asarray(D, np.float64))
    )


def stokes_einstein(T_K: float, mu: float, radius: ArrayLike) -> _Float:
    """Stokes-Einstein diffusivity ``D = kB*T/(6*pi*mu*r)`` of a sphere of
    radius ``r`` in a fluid of viscosity ``mu``.  The diameter form
    ``kB*T/(3*pi*mu*d)`` is identical with ``d=2r``.  Sources:
    src2017_02/article3 (6*pi*r), src2014_10/article5 (3*pi*d).
    """
    return np.asarray(KB * T_K / (6.0 * np.pi * mu * np.asarray(radius, np.float64)))


def fick_flux(D: float, dc: ArrayLike, dx: float) -> _Float:
    """Fick's first law steady flux ``J = -D*dc/dx``.  Sources:
    src2017_02/article3 (fick_flux).
    """
    return np.asarray(-D * np.asarray(dc, np.float64) / dx)


def erfc_profile(c0: float, x: ArrayLike, *, D: float, t: float) -> _Float:
    """Semi-infinite diffusion concentration profile
    ``C(x,t) = C0*erfc(x/(2*sqrt(D*t)))`` (fixed ``C0`` at ``x=0``, far-field 0).

    Uses ``scipy.special.erfc`` (imported lazily).  For the two-plateau step
    front ``C_far + (C0-C_far)*0.5*erfc(...)`` see the article facade.  Sources:
    src2015_08/article3 (gas_concentration_profile).
    """
    from scipy.special import erfc

    arg = np.asarray(x, np.float64) / (2.0 * np.sqrt(D * t))
    return np.asarray(c0 * erfc(arg))


def early_time_uptake(D: float, t: ArrayLike, half_length: float) -> _Float:
    """Early-time fractional uptake ``Mt/Minf = (2/L)*sqrt(D*t/pi)`` for a slab
    of half-length ``L`` (valid to ~0.6).  Sources: src2015_02/article2
    (fick_early_time_recovery).
    """
    return np.asarray((2.0 / half_length) * np.sqrt(D * np.asarray(t, np.float64) / np.pi))


def millington_quirk(D0: float, phi: ArrayLike, sw: ArrayLike = 1.0) -> _Float:
    """Millington-Quirk effective diffusivity ``D_eff = (phi*Sw)**(10/3)/phi**2 * D0``.

    Equivalent to ``D0*phi**(4/3)*Sw**(10/3)``; the ``(phi*Sw)**(10/3)/phi**2``
    structure is kept to reproduce the article float path.  Sources:
    src2014_10/article5 (millington_quirk).
    """
    p = np.asarray(phi, np.float64)
    phi_w = p * np.asarray(sw, np.float64)
    return np.asarray(phi_w ** (10.0 / 3.0) / p**2 * D0)


def pore_velocity(u_darcy: ArrayLike, phi: ArrayLike, sw: ArrayLike = 1.0) -> _Float:
    """Interstitial (pore) velocity ``v = u_darcy/(phi*Sw)``.  Sources:
    src2014_10/article5 (pore_velocity).
    """
    return np.asarray(
        np.asarray(u_darcy, np.float64) / (np.asarray(phi, np.float64) * np.asarray(sw, np.float64))
    )


def advect_disperse_1d(
    c_in: float,
    *,
    length: float,
    n_cells: int,
    t_total: float,
    v: float,
    D: float,
    k_rxn: float = 0.0,
    n_steps: int | None = None,
) -> tuple[_Float, _Float]:
    """Explicit finite-difference 1-D advection-dispersion-reaction transport.

    Solves ``dC/dt = -v*dC/dx + D*d2C/dx2 - k_rxn*C`` on ``[0, length]`` with a
    fixed inflow ``C(0)=c_in`` (upwind advection, central dispersion, explicit
    first-order reaction), returning ``(x, C)`` at ``t_total``.  The step is
    CFL/diffusion-stable (``dt = 0.4*min(dx/v, dx**2/(2*D))``) unless ``n_steps``
    is given, and ``C`` is clipped to ``[0, c_in]``.  Sources: src2014_10/article5
    (transport_1d).
    """
    dx = length / n_cells
    dt_adv = dx / max(v, _TINY)
    dt_dif = dx**2 / max(2.0 * D, _TINY)
    dt = 0.4 * min(dt_adv, dt_dif)
    steps = int(np.ceil(t_total / dt)) if n_steps is None else n_steps
    dt = t_total / steps
    c = np.zeros(n_cells)
    x = (np.arange(n_cells) + 0.5) * dx
    for _ in range(steps):
        cl = np.empty_like(c)
        cl[0] = c_in
        cl[1:] = c[:-1]
        cr = np.empty_like(c)
        cr[-1] = c[-1]
        cr[:-1] = c[1:]
        adv = -v * (c - cl) / dx
        dif = D * (cr - 2.0 * c + cl) / dx**2
        rxn = -k_rxn * c
        c = c + dt * (adv + dif + rxn)
        c = np.clip(c, 0.0, c_in)
    return np.asarray(x), np.asarray(c)
