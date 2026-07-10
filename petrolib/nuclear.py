"""Nuclear logging: capture cross-section, attenuation, density, GR, neutron.

The canonical home for the pulsed-neutron capture-cross-section (Sigma) algebra,
gamma / photon attenuation (Beer-Lambert), gamma-gamma density logging,
spectral / total gamma ray, neutron slowing-down and hydrogen-index relations,
carbon-oxygen (C/O) and spectroscopy yield conversions, and the counting /
radioactive-decay statistics that recur across the corpus.

Unit policy (LIBRARY_MERGE_PLAN.md section on nuclear): capture cross-section
``Sigma`` in capture units (1 c.u. = 1e-3 / cm); potassium ``K`` in wt%, uranium
``U`` and thorium ``Th`` in ppm; times in microseconds; densities in g/cc.
Physics parameters after ``*`` are keyword-only.  numpy-broadcastable,
scalar-in/scalar-out.

The corpus writes the volumetric Sigma balance, the GR API sum, and the C/O
endpoint interpolation in several term orders and coefficient conventions; the
canonical forms here take the recurring defaults, and the article facades map
their own constants onto the keyword parameters.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2014_06/article2_borehole_carbon_toc -- Article 2: Borehole Carbon Corrections Enable Accurate
  TOC Determination from Nuclear Spectroscopy. Jeffrey Miles and Rob Badry (2014). Petrophysics
  Vol. 55, No. 3 (June 2014), pp. 219-228. DOI: none assigned (this issue predates SPWLA DOI
  assignment).
src2014_08/article6_siliceous_ooze_petrophysics -- Article 6: Petrophysical Analysis of Siliceous-
  Ooze Sediments, More Basin, Norwegian Sea. Ahmed Awadalkarim, Morten Kanne Sorensen, Ida Lykke
  Fabricius (2014). Petrophysics Vol. 55, No. 4 (August 2014), pp. 333-348. DOI: none assigned
  (this issue predates SPWLA DOI assignment).
src2014_10/article3_nuclear_density_alternatives -- Article 3: An Assessment of Fundamentals of
  Nuclear-Based Alternatives to Conventional Chemical-Source Bulk-Density Measurement. Ahmed
  Badruzzaman (2014). Petrophysics Vol. 55, No. 5 (October 2014), pp. 415-434. DOI: none assigned
  (this issue predates SPWLA DOI assignment).
src2014_12/article5_cased_well_gas_saturation -- Article 5: Physical Basis for a Cased-Well
  Quantitative Gas-Saturation Analysis Method. F. Inanc, W.A. Gilchrist, R. Ansari, D. Chace
  (2014). Petrophysics Vol. 55, No. 6 (December 2014), pp. 598-617. DOI: none assigned (this issue
  predates SPWLA DOI assignment).
src2015_02/article5_through_the_bit_logging -- Article 5: Recharacterization and Validation of
  Through-the-Bit-Logging Tool Measurements. Slocombe, Bammi, Hunka, Reischman, Schmid (2015).
  Petrophysics Vol. 56, No. 1 (February 2015), pp. 58-71. DOI: none assigned (this issue predates
  SPWLA DOI assignment).
src2015_08/article2_condensed_vapor_gamma -- Article 2: In-Situ Evaluation of Vapor Properties
  Using Condensed Vapor Gamma. O'Sullivan (2015). Petrophysics Vol. 56, No. 4 (August 2015), pp.
  334-345. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2016_08/article4_spectral_gr_mwd -- Article 4: Spectral Gamma-Ray Measurement While Drilling.
  Xu, Huiszoon, Wang, Adolph, Yi, Cavin, Laughlin, Tollefsen, Jacobsen, Boyce (2016). Petrophysics
  Vol. 57, No. 4 (August 2016), pp. 377-389. DOI: none assigned (this issue predates SPWLA DOI
  assignment).
src2017_08/article1_bakken_nmr_relaxometry -- Article 1: High- and Low-Field NMR Relaxometry and
  Diffusometry of the Bakken Petroleum System. Kausik, Fellah, Feng, Simpson (2017). Petrophysics
  Vol. 58, No. 4 (August 2017), pp. 341-351. DOI: none assigned (this issue predates SPWLA DOI
  assignment).
src2017_08/article6_fast_neutron_gamma_density -- Article 6: A Method of Determining Formation
  Density Based on Fast-Neutron Gamma Coupled Field Theory. Zhang, Zhang, Liu, Wu, Wu, Jia, Ti, Li
  (2017). Petrophysics Vol. 58, No. 4 (August 2017), pp. 411-425. DOI: none assigned (this issue
  predates SPWLA DOI assignment).
src2018_04/article1_shaly_sand_tutorial_part2 -- Article 1 (Tutorial): What is it about Shaly
  Sands? Shaly Sand Tutorial No. 2 of 3. Thomas (2018). DOI: 10.30632/PJV59N2-2018t1. Petrophysics
  Vol. 59 No. 2 (Apr 2018).
src2018_04/article4_neutron_xray_imaging -- Article 4: Simultaneous Neutron and X-Ray Imaging of 3D
  Structure of Organic Matter and Fracture in Shales. Chiang, LaManna, Hussey, Jacobson, Liu,
  Zhang, Georgi, Kone, Chen (2018). DOI: 10.30632/PJV59N2-2018a3. Petrophysics Vol. 59 No. 2 (Apr
  2018).
src2018_10/article2_xray_sourceless_density -- Article 2: A Novel X-Ray Tool for True Sourceless
  Density Logging. Simon, Tkabladze, Beekman, Atobatele, De Looz, Grover, Hamichi, Jundt,
  McFarland, Mlcak, Reijonen, Revol, Stewart, Yeboah, Zhang (2018). DOI: 10.30632/PJV59N5-2018a1.
  Petrophysics Vol. 59 No. 5 (Oct 2018) - "Best of 2018 SPWLA Symposium" issue.
src2019_04/article7_issm_saturation_monitoring -- Article 7: In-Situ Saturation Monitoring (ISSM) -
  Recommendations for Improved Processing. Reed, Cense (2019). DOI: 10.30632/PJV60N2-2019a5.
  Petrophysics Vol. 60 No. 2 (Apr 2019).
src2019_10/article1_tmali_organic_shales -- Article 1: Thermal Maturity-Adjusted Log Interpretation
  (TMALI) in Organic Shales. Craddock, Miles, Lewis, Pomerantz (2019). DOI:
  10.30632/PJV60N5-2019a1. Petrophysics Vol. 60 No. 5 (Oct 2019).
src2019_10/article5_log_soak_log_imbibition -- Article 5: 'Log-Soak-Log' Experiment in Tengiz
  Field: Novel Technology for In-Situ Imbibition Measurements to Support an Improved Oil Recovery
  Project. Seth, Villegas, Iskakov, Playton, Lindsell, Cordova, Turmanbekova, Wang (2019). DOI:
  10.30632/PJV60N5-2019a5. Petrophysics Vol. 60 No. 5 (Oct 2019).
src2020_06/article1_casedhole_horizontal_fe -- Article 1: Lessons Learned From Casedhole Formation
  Evaluation Along Unconventional Horizontal Wells. Sullivan, Wang, Bolshakov, Song, Lazorek,
  Tohidi, Seth (2020). DOI: 10.30632/PJV61N3-2020a1. Petrophysics Vol. 61 No. 3 (Jun 2020).
src2020_06/article2_cement_quality_co_pulsed_neutron -- Article 2: Case Studies Demonstrating the
  Impact of Cement Quality on Carbon/Oxygen and Elemental Analysis From Casedhole Pulsed-Neutron
  Logging. Wang, Sullivan, Seth, Barnes, Wilson, Lazorek (2020). DOI: 10.30632/PJV61N3-2020a2.
  Petrophysics Vol. 61 No. 3 (Jun 2020).
src2020_12/article1_nuclear_spectroscopy_history -- Article 1: A History of Nuclear Spectroscopy in
  Well Logging. Pemper (2020). DOI: 10.30632/PJV61N6-2020a1. Petrophysics Vol. 61 No. 6 (Dec 2020).
src2020_12/article2_formation_chlorine_salinity -- Article 2: Formation Chlorine Measurement From
  Spectroscopy Enables Water Salinity Interpretation: Theory, Modeling, and Applications. Miles,
  Mosse, Grau (2020). DOI: 10.30632/PJV61N6-2020a2. Petrophysics Vol. 61 No. 6 (Dec 2020).
src2020_12/article3_self_compensated_spectroscopy -- Article 3: Self-Compensated Pulsed-Neutron
  Spectroscopy Measurements. Zhou, Rose, Miles, Gendur, Wang, Sullivan (2020). DOI:
  10.30632/PJV61N6-2020a3. Petrophysics Vol. 61 No. 6 (Dec 2020).
src2020_12/article4_co_sigma_saturation_casestudy -- Article 4: New Generation of Pulsed-Neutron
  Multidetector Comparison in a Challenging Multistack Clastic Reservoir - A Case Study in a Brown
  Field, Malaysia. Johare, Mohd Amin, Prasodjo, Afandi, Din (2020). DOI: 10.30632/PJV61N6-2020a4.
  Petrophysics Vol. 61 No. 6 (Dec 2020).
src2020_12/article7_sigma_gas_saturation_lowporosity -- Article 7: Multidetector Pulsed-Neutron
  Tool Application in a Low-Porosity Reservoir - A Case Study in Mutiara Field, Indonesia. Wijaya,
  Aulianagara, Guo, Naibaho, Asriwan, Amirudin (2020). DOI: 10.30632/PJV61N6-2020a7. Petrophysics
  Vol. 61 No. 6 (Dec 2020).
src2023_08/article1_nuclear_logging -- Fitz, D.E. (2023). "Evolution of Casedhole Nuclear
  Surveillance Logging Through Time", Petrophysics, Vol. 64, No. 4 (August 2023), pp. 473-501. DOI:
  10.30632/PJV64N4-2023a1.
src2023_10/article_05_laronga_pulsed_neutron_ccs -- Laronga, R., Swager, L., and Bustos, U. (2023).
  "Time-Lapse Pulsed- Neutron Logs for Carbon Capture and Sequestration: Practical Learnings and
  Key Insights." Petrophysics, 64(5), 680-699. DOI: 10.30632/PJV64N5-2023a5.
src2023_12/mcglynn_pulsed_neutron -- McGlynn et al. (2023), Petrophysics 64(6): 900-918. New
  pulsed-neutron spectroscopy instrument with LaBr3 detectors providing simultaneous C/O ratio,
  capture sigma, and ratio-based gas measurements for two- and three-phase saturation analysis.
src2024_06/article1_nuclear_logging_ccs -- Badruzzaman, A. (2024). "Nuclear Logging in Geological
  Probing for a Low-Carbon Energy Future - A New Frontier?" Petrophysics 65(3), 274-301. DOI:
  10.30632/PJV65N3-2024a1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]

# Avogadro's number (1/mol), as used across the corpus (not the CODATA digits).
NA = 6.022e23
# Sigma <-> decay-time constant (us * c.u.): tau = 4550/Sigma.
SIGMA_TAU_CONST = 4550.0
# Thermal-neutron velocity (cm/s) for the pulsed-neutron capture decay.
THERMAL_NEUTRON_VELOCITY_CM_S = 2.2e5
# Gamma-gamma electron-density -> bulk-density transform (rho_b = m*rho_e + c).
RHOE_TO_RHOB_SLOPE = 1.0704
RHOE_TO_RHOB_OFFSET = 0.1883


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


# ==========================================================================
# 1. Pulsed-neutron capture cross-section (Sigma / PNC)
# ==========================================================================


def sigma_forward(
    phi: ArrayLike,
    sw: ArrayLike,
    *,
    sigma_ma: float = 10.0,
    sigma_w: float = 55.0,
    sigma_hc: float = 21.0,
    vsh: ArrayLike = 0.0,
    sigma_sh: float = 27.0,
) -> _Float:
    """Volumetric capture cross-section ``Sigma`` (c.u.) of a rock.

    ``Sigma = (1-phi-Vsh)*Sigma_ma + Vsh*Sigma_sh + phi*(Sw*Sigma_w + (1-Sw)*Sigma_hc)``.
    With ``vsh=0`` this is the clean two-phase form.

    Sources: src2020_12/article7_sigma_gas_saturation_lowporosity,
    src2023_08/article1_nuclear_logging, src2023_10/article_05_laronga_pulsed_neutron_ccs.
    """
    phi_a = _arr(phi)
    sw_a = _arr(sw)
    vsh_a = _arr(vsh)
    matrix = (1.0 - phi_a - vsh_a) * sigma_ma + vsh_a * sigma_sh
    fluid = phi_a * (sw_a * sigma_w + (1.0 - sw_a) * sigma_hc)
    return np.asarray(matrix + fluid)


def sigma_forward_3phase(
    phi: ArrayLike,
    so: ArrayLike,
    sg: ArrayLike,
    sw: ArrayLike,
    *,
    sigma_oil: float = 22.0,
    sigma_gas: float = 8.0,
    sigma_w: float = 80.0,
    sigma_ma: float = 10.0,
) -> _Float:
    """Three-phase (oil/gas/water) capture cross-section ``Sigma`` (c.u.).

    ``Sigma = (1-phi)*Sigma_ma + phi*(So*Sigma_oil + Sg*Sigma_gas + Sw*Sigma_w)``.

    Sources: src2023_12/mcglynn_pulsed_neutron.
    """
    phi_a = _arr(phi)
    fluid = _arr(so) * sigma_oil + _arr(sg) * sigma_gas + _arr(sw) * sigma_w
    return np.asarray((1.0 - phi_a) * sigma_ma + phi_a * fluid)


def sw_from_sigma(
    sigma_t: ArrayLike,
    phi: ArrayLike,
    *,
    sigma_ma: float,
    sigma_w: float,
    sigma_hc: float,
    vsh: ArrayLike = 0.0,
    sigma_sh: float = 27.0,
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> _Float:
    """Water saturation by inverting :func:`sigma_forward` for ``Sw``.

    ``Sw = (Sigma_t - (1-phi-Vsh)*Sigma_ma - Vsh*Sigma_sh - phi*Sigma_hc)
    / (phi*(Sigma_w - Sigma_hc))``.  ``clip`` bounds the result (``None`` to skip).

    Sources: src2019_10/article1_tmali_organic_shales,
    src2019_10/article5_log_soak_log_imbibition, src2020_06/article1_casedhole_horizontal_fe,
    src2020_06/article2_cement_quality_co_pulsed_neutron,
    src2020_12/article4_co_sigma_saturation_casestudy,
    src2020_12/article7_sigma_gas_saturation_lowporosity.
    """
    sigma_ta = _arr(sigma_t)
    phi_a = _arr(phi)
    vsh_a = _arr(vsh)
    num = sigma_ta - (1.0 - phi_a - vsh_a) * sigma_ma - vsh_a * sigma_sh - phi_a * sigma_hc
    sw = num / (phi_a * (sigma_w - sigma_hc))
    if clip is not None:
        sw = np.clip(sw, clip[0], clip[1])
    return np.asarray(sw)


def delta_sw_timelapse(
    sigma_base: ArrayLike,
    sigma_mon: ArrayLike,
    phi: ArrayLike,
    *,
    sigma_w_base: float,
    sigma_w_mon: float,
    sigma_hc: float,
    sw_base: ArrayLike,
) -> _Float:
    """Monitor-survey water saturation from a time-lapse Sigma pair.

    ``Sw_mon = [(Sigma_mon - Sigma_base)/phi + Sw_base*(Sigma_w_base - Sigma_hc)]
    / (Sigma_w_mon - Sigma_hc)`` (e.g. injected brine changing ``Sigma_w``).
    """
    num = (_arr(sigma_mon) - _arr(sigma_base)) / _arr(phi) + _arr(sw_base) * (
        sigma_w_base - sigma_hc
    )
    return np.asarray(num / (sigma_w_mon - sigma_hc))


def sigma_sensitivity(phi: ArrayLike, sigma_w: float, sigma_hc: float) -> _Float:
    """Saturation sensitivity ``dSigma/dSw = phi*(Sigma_w - Sigma_hc)`` (c.u.).

    Sources: src2019_10/article5_log_soak_log_imbibition,
    src2020_12/article7_sigma_gas_saturation_lowporosity.
    """
    return np.asarray(_arr(phi) * (sigma_w - sigma_hc))


def sigma_w_from_salinity(
    ppm_nacl: ArrayLike, *, temperature_c: float = 75.0, model: str = "fitz2023"
) -> _Float:
    """Brine capture cross-section ``Sigma_w`` (c.u.) from NaCl salinity (ppm).

    ``model='fitz2023'``: ``(22 + 750*s)*(1 - 8e-4*(T-75))`` with ``s=ppm/1e6``.
    ``model='linear450k'``: ``22 + (220-22)*ppm/450000``.

    Sources: src2019_10/article5_log_soak_log_imbibition, src2023_08/article1_nuclear_logging.
    """
    ppm = _arr(ppm_nacl)
    if model == "fitz2023":
        s = ppm / 1e6
        return np.asarray((22.0 + 750.0 * s) * (1.0 - 0.0008 * (temperature_c - 75.0)))
    if model == "linear450k":
        return np.asarray(22.0 + (220.0 - 22.0) * ppm / 450000.0)
    raise ValueError(f"unknown model {model!r}; use 'fitz2023' or 'linear450k'")


def number_density(rho_g_cc: ArrayLike, wfrac: ArrayLike, atomic_mass: ArrayLike) -> _Float:
    """Atomic number density ``N = rho*NA*w/A`` (1/cm3) for a mass fraction ``w``.

    Sources: src2020_12/article1_nuclear_spectroscopy_history.
    """
    return np.asarray(_arr(rho_g_cc) * NA * _arr(wfrac) / _arr(atomic_mass))


def macroscopic_sigma(
    number_densities: ArrayLike, micro_barns: ArrayLike, *, units: str = "cu"
) -> _Float:
    """Macroscopic capture ``Sigma`` from number densities and micro cross-sections.

    ``Sigma = sum_i N_i * sigma_i`` with ``sigma`` in barns (1 barn = 1e-24 cm2);
    ``units='cu'`` returns c.u. (cm^-1 * 1e3), ``units='cm'`` returns cm^-1.

    Sources: src2020_12/article1_nuclear_spectroscopy_history.
    """
    sigma_cm = np.sum(_arr(number_densities) * _arr(micro_barns) * 1e-24)
    if units == "cu":
        return np.asarray(sigma_cm * 1e3)
    if units == "cm":
        return np.asarray(sigma_cm)
    raise ValueError(f"unknown units {units!r}; use 'cu' or 'cm'")


def sigma_from_tau(tau_us: ArrayLike) -> _Float:
    """Capture cross-section from thermal decay time ``Sigma = 4550/tau`` (c.u.).

    Sources: src2020_12/article1_nuclear_spectroscopy_history.
    """
    return np.asarray(SIGMA_TAU_CONST / _arr(tau_us))


def tau_from_sigma(sigma_cu: ArrayLike) -> _Float:
    """Thermal decay time from capture cross-section ``tau = 4550/Sigma`` (us).

    Sources: src2020_12/article1_nuclear_spectroscopy_history.
    """
    return np.asarray(SIGMA_TAU_CONST / _arr(sigma_cu))


def pnc_decay(t_us: ArrayLike, n0: float, sigma_cu: float, *, background: float = 0.0) -> _Float:
    """Pulsed-neutron capture decay ``N(t) = N0*exp(-Sigma*v*t) + background``.

    ``Sigma`` (c.u.) is converted to cm^-1 (``*1e-3``), ``v`` is the thermal-neutron
    velocity, and ``t`` (us) to seconds (``*1e-6``).

    Sources: src2024_06/article1_nuclear_logging_ccs.
    """
    sigma_cm = sigma_cu * 1e-3
    decay_rate = sigma_cm * THERMAL_NEUTRON_VELOCITY_CM_S
    return np.asarray(n0 * np.exp(-decay_rate * _arr(t_us) * 1e-6) + background)


def sigma_from_decay_fit(
    t_us: ArrayLike, counts: ArrayLike, *, fit_window: tuple[int, int] | None = None
) -> _Float:
    """Capture cross-section from a log-linear fit of a capture-decay curve.

    Fits ``ln(counts)`` vs ``t`` (seconds); ``Sigma = -slope/v * 1e3`` (c.u.).
    ``fit_window`` optionally selects a ``(start, stop)`` sample slice.
    """
    t_s = _arr(t_us) * 1e-6
    log_n = np.log(_arr(counts))
    if fit_window is not None:
        sl = slice(*fit_window)
        t_s, log_n = t_s[sl], log_n[sl]
    slope = np.polyfit(t_s, log_n, 1)[0]
    return np.asarray(-slope / THERMAL_NEUTRON_VELOCITY_CM_S * 1e3)


# ==========================================================================
# 2. Gamma / photon attenuation (Beer-Lambert)
# ==========================================================================


def beer_lambert(i0: ArrayLike, mu: ArrayLike, x: ArrayLike) -> _Float:
    """Beer-Lambert transmitted intensity ``I = I0*exp(-mu*x)``.

    Sources: src2014_10/article3_nuclear_density_alternatives,
    src2018_04/article4_neutron_xray_imaging, src2019_04/article7_issm_saturation_monitoring.
    """
    return np.asarray(_arr(i0) * np.exp(-_arr(mu) * _arr(x)))


def mu_from_intensity(i0: ArrayLike, i: ArrayLike, x: ArrayLike) -> _Float:
    """Linear attenuation coefficient ``mu = ln(I0/I)/x`` from an intensity pair.

    Sources: src2019_04/article7_issm_saturation_monitoring.
    """
    return np.asarray(np.log(_arr(i0) / _arr(i)) / _arr(x))


def attenuation_map(i: ArrayLike, i_ref: ArrayLike) -> _Float:
    """Optical-density attenuation ``-ln(I/I_ref)`` (zero-guarded ratio).

    Sources: src2018_04/article4_neutron_xray_imaging.
    """
    ratio = np.divide(_arr(i), _arr(i_ref))
    return np.asarray(-np.log(np.clip(ratio, 1e-30, None)))


def gamma_count(
    rho: ArrayLike, *, n0: float = 1e6, mu_mass: float = 0.06, spacing_cm: float = 30.0
) -> _Float:
    """Gamma count rate vs density ``N = N0*exp(-mu_mass*rho*spacing)``.

    Sources: src2017_08/article6_fast_neutron_gamma_density,
    src2018_10/article2_xray_sourceless_density.
    """
    return np.asarray(n0 * np.exp(-mu_mass * _arr(rho) * spacing_cm))


def density_from_count(
    count: ArrayLike, *, n0: float = 1e6, mu_mass: float = 0.06, spacing_cm: float = 30.0
) -> _Float:
    """Density from a gamma count ``rho = ln(N0/N)/(mu_mass*spacing)`` (inverse).

    Sources: src2017_08/article6_fast_neutron_gamma_density,
    src2018_10/article2_xray_sourceless_density.
    """
    return np.asarray(np.log(n0 / _arr(count)) / (mu_mass * spacing_cm))


# ==========================================================================
# 3. Gamma-gamma density logging
# ==========================================================================


def dual_detector_density(near: ArrayLike, far: ArrayLike, *, a: float, b: float) -> _Float:
    """Dual-detector density ``rho = a + b*ln(near/far)`` (count-ratio calibration).

    Sources: src2014_10/article3_nuclear_density_alternatives,
    src2017_08/article6_fast_neutron_gamma_density.
    """
    return np.asarray(a + b * np.log(_arr(near) / _arr(far)))


def spine_ribs(
    rho_ls: ArrayLike,
    rho_ss: ArrayLike,
    *,
    rib_coeffs: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> tuple[_Float, _Float]:
    """Spine-and-ribs compensated density -> ``(rho_b, drho)``.

    ``drho = c0 + c1*d + c2*d**2`` with ``d = rho_ls - rho_ss`` (long-short spacing);
    ``rho_b = rho_ls + drho``.

    Sources: src2018_10/article2_xray_sourceless_density.
    """
    d = _arr(rho_ls) - _arr(rho_ss)
    c0, c1, c2 = rib_coeffs
    drho = c0 + c1 * d + c2 * d**2
    return np.asarray(_arr(rho_ls) + drho), np.asarray(drho)


def electron_density_index(z: ArrayLike, a: ArrayLike, rho_m: ArrayLike) -> _Float:
    """Electron density ``rho_e = (2*Z/A)*rho_m`` of a single element/compound.

    Sources: src2015_02/article5_through_the_bit_logging,
    src2019_10/article1_tmali_organic_shales.
    """
    return np.asarray((2.0 * _arr(z) / _arr(a)) * _arr(rho_m))


def electron_density_mixture(
    z: ArrayLike, a: ArrayLike, mass_frac: ArrayLike, rho_m: ArrayLike
) -> _Float:
    """Mixture electron density ``rho_e = 2*sum(w_i*Z_i/A_i)*rho_m``.

    Sources: src2015_02/article5_through_the_bit_logging.
    """
    zam = np.sum(_arr(mass_frac) * _arr(z) / _arr(a))
    return np.asarray(2.0 * zam * _arr(rho_m))


def rhob_from_rhoe(rho_e: ArrayLike) -> _Float:
    """Bulk density from electron density ``rho_b = 1.0704*rho_e - 0.1883``.

    Sources: src2019_10/article1_tmali_organic_shales.
    """
    return np.asarray(RHOE_TO_RHOB_SLOPE * _arr(rho_e) - RHOE_TO_RHOB_OFFSET)


def rhoe_from_rhob(rho_b: ArrayLike) -> _Float:
    """Electron density from bulk density ``rho_e = (rho_b + 0.1883)/1.0704`` (inverse)."""
    return np.asarray((_arr(rho_b) + RHOE_TO_RHOB_OFFSET) / RHOE_TO_RHOB_SLOPE)


# ==========================================================================
# 4. Gamma ray
# ==========================================================================


def gr_api(
    k_pct: ArrayLike,
    u_ppm: ArrayLike,
    th_ppm: ArrayLike,
    *,
    coeff: tuple[float, float, float] = (16.0, 8.0, 4.0),
) -> _Float:
    """Total (spectral) gamma ray API ``SGR = ck*K + cu*U + cth*Th``.

    Default ``coeff=(16, 8, 4)`` (K wt%, U ppm, Th ppm); pass e.g.
    ``(16.32, 8.09, 3.93)`` for the alternative calibration.

    Sources: src2016_08/article4_spectral_gr_mwd,
    src2018_04/article1_shaly_sand_tutorial_part2, src2020_06/article1_casedhole_horizontal_fe,
    src2020_12/article1_nuclear_spectroscopy_history.
    """
    ck, cu, cth = coeff
    return np.asarray(ck * _arr(k_pct) + cu * _arr(u_ppm) + cth * _arr(th_ppm))


def cgr_api(
    k_pct: ArrayLike, th_ppm: ArrayLike, *, coeff: tuple[float, float] = (16.0, 4.0)
) -> _Float:
    """Computed (uranium-free) gamma ray API ``CGR = ck*K + cth*Th``.

    Sources: src2016_08/article4_spectral_gr_mwd.
    """
    ck, cth = coeff
    return np.asarray(ck * _arr(k_pct) + cth * _arr(th_ppm))


# ==========================================================================
# 5. Neutron / hydrogen index
# ==========================================================================


def hydrogen_index_fluid(
    fluid: str = "water", *, rho_gas: float = 0.2, hi_oil: float = 0.9
) -> float:
    """Hydrogen index of a pore fluid (``HI``, not ``HI*phi``).

    ``water`` -> 1.0, ``gas`` -> ``rho_gas``, ``oil`` -> ``hi_oil``.
    """
    table = {"water": 1.0, "gas": rho_gas, "oil": hi_oil}
    if fluid not in table:
        raise ValueError(f"unknown fluid {fluid!r}; use 'water', 'gas', or 'oil'")
    return table[fluid]


def hydrogen_index_chemical(
    rho: ArrayLike,
    n_protons: ArrayLike,
    mol_weight: ArrayLike,
    *,
    rho_ref: float = 1.0,
    n_ref: float = 2.0,
    mw_ref: float = 18.015,
) -> _Float:
    """Hydrogen index from chemistry, normalised to fresh water.

    ``HI = (rho*n_protons/M) / (rho_ref*n_ref/mw_ref)``.
    """
    sample = _arr(rho) * _arr(n_protons) / _arr(mol_weight)
    return np.asarray(sample / (rho_ref * n_ref / mw_ref))


def hi_mix(phi: ArrayLike, *, hi_fluid: float = 1.0, hi_matrix: float = 0.0) -> _Float:
    """Volumetric hydrogen-index log ``HI = phi*HI_fluid + (1-phi)*HI_matrix``.

    Sources: src2014_08/article6_siliceous_ooze_petrophysics.
    """
    phi_a = _arr(phi)
    return np.asarray(phi_a * hi_fluid + (1.0 - phi_a) * hi_matrix)


def phi_from_hi(hi_log: ArrayLike, *, hi_matrix: float, hi_fluid: float = 1.0) -> _Float:
    """Porosity from a hydrogen-index log ``phi = (HI - HI_matrix)/(HI_fluid - HI_matrix)``.

    Sources: src2014_08/article6_siliceous_ooze_petrophysics.
    """
    return np.asarray((_arr(hi_log) - hi_matrix) / (hi_fluid - hi_matrix))


def phi_hi_correction(phi_apparent: ArrayLike, hi: ArrayLike) -> _Float:
    """Hydrogen-index porosity correction ``phi = phi_apparent/HI`` (neutron & NMR).

    Sources: src2017_08/article1_bakken_nmr_relaxometry.
    """
    return np.asarray(_arr(phi_apparent) / _arr(hi))


def collision_parameter(a_mass: ArrayLike) -> _Float:
    """Neutron collision parameter ``alpha = ((A-1)/(A+1))**2``.

    Sources: src2014_12/article5_cased_well_gas_saturation.
    """
    a = _arr(a_mass)
    return np.asarray(((a - 1.0) / (a + 1.0)) ** 2)


def average_lethargy_gain(a_mass: ArrayLike) -> _Float:
    """Mean logarithmic energy decrement ``xi = 1 + alpha/(1-alpha)*ln(alpha)``.

    The hydrogen limit ``A=1`` (``alpha=0``) returns ``xi=1``.

    Sources: src2014_12/article5_cased_well_gas_saturation.
    """
    alpha = collision_parameter(a_mass)
    safe = np.where(alpha > 0.0, alpha, 1.0)
    xi = 1.0 + alpha / (1.0 - alpha) * np.log(safe)
    return np.asarray(np.where(alpha > 0.0, xi, 1.0))


def moderating_power(a_mass: ArrayLike, sigma_s: ArrayLike) -> _Float:
    """Moderating power ``xi*Sigma_s`` (slowing-down power).

    Sources: src2014_12/article5_cased_well_gas_saturation.
    """
    return np.asarray(average_lethargy_gain(a_mass) * _arr(sigma_s))


def slowing_down_length_empirical(
    phi: ArrayLike, e_mev: ArrayLike, *, ls0: float = 20.0, e_ref: float = 4.5
) -> _Float:
    """Empirical slowing-down length ``Ls = Ls0*(1 - 0.6*phi)*sqrt(E/E_ref)`` (cm)."""
    return np.asarray(ls0 * (1.0 - 0.6 * _arr(phi)) * np.sqrt(_arr(e_mev) / e_ref))


def phi_from_ls(
    ls: ArrayLike, e_mev: ArrayLike, *, ls0: float = 20.0, e_ref: float = 4.5
) -> _Float:
    """Porosity from slowing-down length (inverse of :func:`slowing_down_length_empirical`)."""
    return np.asarray((1.0 - _arr(ls) / (ls0 * np.sqrt(_arr(e_mev) / e_ref))) / 0.6)


def transport_length_mix(vol_fracs: ArrayLike, lengths: ArrayLike) -> _Float:
    """Harmonic transport/migration-length mix ``L = 1/sum(v_i/L_i)``."""
    return np.asarray(1.0 / np.sum(_arr(vol_fracs) / _arr(lengths)))


# Migration-length neutron-porosity endpoints (matrix, fluid) in cm.
_LM_ENDPOINTS = {"limestone": (15.5, 6.0), "sandstone": (16.5, 6.0), "dolomite": (14.5, 6.0)}


def phi_n_from_lm(lm_star: ArrayLike, *, lithology: str = "limestone") -> _Float:
    """Neutron porosity from migration length ``phi = (Lm_ma - Lm)/(Lm_ma - Lm_fl)``."""
    if lithology not in _LM_ENDPOINTS:
        raise ValueError(f"unknown lithology {lithology!r}; use {sorted(_LM_ENDPOINTS)}")
    lm_ma, lm_fl = _LM_ENDPOINTS[lithology]
    return np.asarray((lm_ma - _arr(lm_star)) / (lm_ma - lm_fl))


def compensated_neutron_porosity(
    near: ArrayLike, far: ArrayLike, *, a: float = 0.0, b: float = -30.0, c: float = 45.0
) -> _Float:
    """Compensated neutron porosity ``phi = a + b*lnR + c*lnR**2`` (p.u.), ``R=near/far``."""
    ln_r = np.log(_arr(near) / _arr(far))
    return np.asarray(a + b * ln_r + c * ln_r**2)


# ==========================================================================
# 6. Carbon/oxygen (C/O) and spectroscopy
# ==========================================================================


def co_ratio(c_yield: ArrayLike, o_yield: ArrayLike) -> _Float:
    """Carbon-oxygen ratio ``COR = Y_C/Y_O``.

    Sources: src2020_06/article2_cement_quality_co_pulsed_neutron,
    src2020_12/article1_nuclear_spectroscopy_history.
    """
    return np.asarray(_arr(c_yield) / _arr(o_yield))


def so_from_co(
    cor: ArrayLike,
    cor_water: ArrayLike,
    cor_oil: ArrayLike,
    *,
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> _Float:
    """Oil saturation from C/O endpoint interpolation ``So = (COR-COR_w)/(COR_o-COR_w)``.

    Sources: src2020_06/article2_cement_quality_co_pulsed_neutron,
    src2020_12/article4_co_sigma_saturation_casestudy.
    """
    so = (_arr(cor) - _arr(cor_water)) / (_arr(cor_oil) - _arr(cor_water))
    if clip is not None:
        so = np.clip(so, clip[0], clip[1])
    return np.asarray(so)


def co_forward_3phase(
    phi: ArrayLike,
    so: ArrayLike,
    sg: ArrayLike,
    sw: ArrayLike,
    *,
    c_oil: float = 0.85,
    c_gas: float = 0.75,
    c_mat: float = 0.1,
    o_w: float = 1.0,
    o_mat: float = 0.55,
) -> _Float:
    """Inelastic C/O forward model ``COR = C/(O+1e-9)`` from a 3-phase saturation.

    ``C = phi*(So*c_oil + Sg*c_gas) + (1-phi)*c_mat``;
    ``O = phi*Sw*o_w + (1-phi)*o_mat``.

    Sources: src2023_12/mcglynn_pulsed_neutron.
    """
    phi_a = _arr(phi)
    carbon = phi_a * (_arr(so) * c_oil + _arr(sg) * c_gas) + (1.0 - phi_a) * c_mat
    oxygen = phi_a * _arr(sw) * o_w + (1.0 - phi_a) * o_mat
    return np.asarray(carbon / (oxygen + 1e-9))


def yields_to_weights(fy2w: ArrayLike, s: ArrayLike, y: ArrayLike) -> _Float:
    """Elemental weight fraction from a spectroscopy yield ``W = FY2W*S*Y``.

    Sources: src2020_12/article2_formation_chlorine_salinity,
    src2020_12/article3_self_compensated_spectroscopy.
    """
    return np.asarray(_arr(fy2w) * _arr(s) * _arr(y))


def weights_to_yields(fy2w: ArrayLike, s: ArrayLike, w: ArrayLike) -> _Float:
    """Spectroscopy yield from a weight fraction ``Y = W/(FY2W*S)`` (inverse).

    Sources: src2020_12/article2_formation_chlorine_salinity.
    """
    return np.asarray(_arr(w) / (_arr(fy2w) * _arr(s)))


def toc_from_yield(y_toc: ArrayLike, calib: ArrayLike) -> _Float:
    """Total organic carbon from a carbon yield ``TOC = calib*Y_TOC``.

    Sources: src2014_06/article2_borehole_carbon_toc.
    """
    return np.asarray(_arr(calib) * _arr(y_toc))


# ==========================================================================
# 7. Counting statistics and radioactive decay
# ==========================================================================


def counting_precision(counts: ArrayLike) -> _Float:
    """Relative counting precision ``1/sqrt(N)`` (Poisson)."""
    return np.asarray(1.0 / np.sqrt(_arr(counts)))


def counting_sigma(value: ArrayLike, counts: ArrayLike) -> _Float:
    """Absolute counting uncertainty ``value/sqrt(N)`` (Poisson).

    Sources: src2014_10/article3_nuclear_density_alternatives.
    """
    return np.asarray(_arr(value) / np.sqrt(_arr(counts)))


def decay_constant(half_life: ArrayLike) -> _Float:
    """Radioactive decay constant ``lambda = ln(2)/T_half``.

    Sources: src2015_08/article2_condensed_vapor_gamma.
    """
    return np.asarray(np.log(2.0) / _arr(half_life))


def radioactive_decay(n0: ArrayLike, t: ArrayLike, half_life: ArrayLike) -> _Float:
    """Radioactive decay ``N = N0*exp(-lambda*t)`` with ``lambda = ln(2)/T_half``.

    Sources: src2015_08/article2_condensed_vapor_gamma.
    """
    lam = decay_constant(half_life)
    return np.asarray(_arr(n0) * np.exp(-lam * _arr(t)))


def activity(n: ArrayLike, half_life: ArrayLike) -> _Float:
    """Radioactive activity ``A = lambda*N`` with ``lambda = ln(2)/T_half``.

    Sources: src2015_08/article2_condensed_vapor_gamma.
    """
    return np.asarray(decay_constant(half_life) * _arr(n))
