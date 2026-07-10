"""Electromagnetic / dielectric petrophysics: permittivity, dispersion, mixing.

The canonical home for the complex-permittivity algebra, relaxation models
(Debye / Cole-Cole / Havriliak-Negami and the Pelton IP resistivity form),
volumetric mixing laws (Lichtenecker-Rother / CRIM, Maxwell-Garnett, symmetric
and asymmetric Bruggeman), spheroid depolarization factors, and the
induction / propagation-logging relations (skin depth, complex wavenumber,
phase / attenuation) that recur across the corpus.

Conventions (LIBRARY_MERGE_PLAN.md section on em_dielectric):

* Frequency is ``freq_hz`` in Hz; angular frequency ``omega = 2*pi*freq_hz``.
* Complex permittivity uses the engineering ``-j`` loss convention:
  ``eps* = eps' - j*(eps'' + sigma/(omega*eps0))``.  A positive conductivity
  therefore appears as a *negative* imaginary part.
* Physics parameters after ``*`` are keyword-only.
* Functions are numpy-broadcastable and scalar-in/scalar-out.  The mixing and
  effective-medium helpers are complex-capable: real inputs return real arrays,
  complex inputs return complex arrays.

Constants: ``EPS0`` (vacuum permittivity, F/m) and ``MU0`` (vacuum
permeability, H/m).

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2014_04/article5_deepwater_formation_evaluation -- Article 5: Formation-Evaluation Challenges
  and Opportunities in Deepwater Roland Chemali, Wade Samec, Ron Balliet, Paul Cooper, David
  Torres, Chris Jones (2014). Petrophysics Vol. 55, No. 2 (April 2014), pp. 124-135. DOI: none
  assigned (this issue predates SPWLA DOI assignment).
src2014_08/article5_dielectric_permittivity_shales -- Article 5: Dielectric Permittivity: A
  Petrophysical Parameter for Shales. Matthew Josh (2014). Petrophysics Vol. 55, No. 4 (August
  2014), pp. 319-332. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2014_10/article1_lwd_inversion_anisotropy -- Article 1: Inversion-Based Interpretation of
  Logging-While-Drilling Resistivity and Nuclear Measurements: Field Examples in High-Angle and
  Horizontal Wells Olabode Ijasan, Carlos Torres-Verdin, William E. Preeg, John Rasmus, Edward J.
  Stockhausen (2014). Petrophysics Vol. 55, No. 5 (October 2014), pp. 374-391. DOI: none assigned
  (this issue predates SPWLA DOI assignment).
src2014_10/article6_dielectric_oil_movability -- Article 6: Application of an Oil-Movability
  Quicklook Technique Using Dielectric Measurements at Four Depths of Investigation. S.T. Grayson
  and J.L. Hemingway (2014). Petrophysics Vol. 55, No. 5 (October 2014), pp. 461-469. DOI: none
  assigned (this issue predates SPWLA DOI assignment).
src2014_12/article3_nmr_dielectric_limestone -- Article 3: Experimental Study of the Effects of
  Wettability and Fluid Saturation on NMR and Dielectric Measurements in Limestone Lalitha
  Venkataramanan, Martin D. Hurlimann, Jeffrey A. Tarvin, Kamilla Fellah, Diana Acero-Allard,
  Nikita V. Seleznev (2014). Petrophysics Vol. 55, No. 6 (December 2014), pp. 572-586. DOI: none
  assigned (this issue predates SPWLA DOI assignment).
src2014_12/article4_porescale_dielectric -- Article 4: Pore-Scale Evaluation of Dielectric
  Measurements in Formations with Complex Pore and Grain Structures. Huangye Chen and Zoya Heidari
  (2014). Petrophysics Vol. 55, No. 6 (December 2014), pp. 587-597. DOI: none assigned (this issue
  predates SPWLA DOI assignment).
src2014_12/article6_array_induction_geometric_factors -- Article 6: Reminiscences on the
  Development of the First Commercial Array-Induction Measurement. Peter Elkington (2014).
  Petrophysics Vol. 55, No. 6 (December 2014), pp. 618-623. DOI: none assigned (this issue predates
  SPWLA DOI assignment).
src2016_02/article5_shaly_sand_graphical_solutions -- Article 5: Graphical Solutions for Laminated
  and Dispersed Shaly Sands. Bootle (2016). Petrophysics Vol. 57, No. 1 (February 2016), pp. 51-59.
  DOI: none assigned (this issue predates SPWLA DOI assignment).
src2016_10/article1_emla_lookahead_resistivity -- Article 1: Looking Ahead of the Bit While
  Drilling: From Vision to Reality. Constable, Antonsen, Stalheim, Olsen, Fjell, Dray, Eikenes,
  Aarflot, Haldorsen, Digranes, Seydoux, Omeragic, Thiel, Davydychev, Denichou, Salim, Frey, Homan,
  Tan (2016). Petrophysics Vol. 57, No. 5 (October 2016), pp. 426-446. DOI: none assigned (this
  issue predates SPWLA DOI assignment).
src2016_10/article4_dielectric_archie_greenriver -- Article 4: Advanced Log Interpretation in Field
  Development. Merkel, Lessenger (2016). Petrophysics Vol. 57, No. 5 (October 2016), pp. 479-491.
  DOI: none assigned (this issue predates SPWLA DOI assignment).
src2017_06/article5_complex_resistivity_dispersion -- Article 5: Borehole Measurements of the
  Complex-Resistivity Dispersion Spectrum: A New Logging Method to Identify Low-Resistivity
  Reservoirs. Jiang, Ke, Kang, Sun, Yin (2017). Petrophysics Vol. 58, No. 3 (June 2017), pp.
  281-288. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2018_04/article6_dielectric_matrix_crim_cda -- Article 6: Improving Dielectric Interpretation by
  Calibrating Matrix Permittivity and Solving Dielectric Mixing Laws With a New Graphical Method.
  Wang, Wang, Toumelin, Brown, Crousse (2018). DOI: 10.30632/PJV59N2-2018a5. Petrophysics Vol. 59
  No. 2 (Apr 2018).
src2018_04/article7_bakken_dielectric_dispersion -- Article 7: Bakken Petroleum System
  Characterization Using Dielectric-Dispersion Logs. Han, Misra (2018). DOI:
  10.30632/PJV59N2-2018a6. Petrophysics Vol. 59 No. 2 (Apr 2018).
src2018_06/article4_clay_network_resistivity_saturation -- Article 4: A New Resistivity-Based Model
  for Improved Hydrocarbon Saturation Assessment in Clay-Rich Formations Using Quantitative
  Geometry of the Clay Network. Garcia, Jagadisan, Rostami, Heidari (2018). DOI:
  10.30632/PJV59N3-2018a3. Petrophysics Vol. 59 No. 3 (Jun 2018).
src2018_06/article5_wideband_em_dem_permittivity -- Article 5: Coherent Interpretation of Wideband
  Electromagnetic Measurements in the Millihertz to Gigahertz Frequency Range. Seleznev, Hou,
  Freed, Habashy, Feng, Fellah, Xu, Nadeev (2018). DOI: 10.30632/PJV59N3-2018a4. Petrophysics Vol.
  59 No. 3 (Jun 2018).
src2019_12/article3_lwd_permittivity -- Article 3: Deducing Electrical Permittivity of Formations
  From LWD Resistivity Measurements. Stalheim (2019). DOI: 10.30632/PJV60N6-2019a3. Petrophysics
  Vol. 60 No. 6 (Dec 2019).
src2020_02/article2_russian_western_resistivity -- Article 2: Comparison of the Russian and Western
  Resistivity Logs in Typical Western Siberian Reservoir Environments - A Numerical Study. Epov,
  Sukhorukova, Nechaev, Petrov, Rabinovich, Weston, Tyurin, Wang, Abubakar, Claverie (2020). DOI:
  10.30632/PJV61N1-2020a1. Petrophysics Vol. 61 No. 1 (Feb 2020).
src2020_02/article3_hdil_array_induction_anisotropic -- Article 3: Response Characteristics of an
  Array Induction Tool (HDIL) in Heterogeneous Anisotropic Formations. Liu, Zhang, Zhang, Xu, Kang,
  Xiao (2020). DOI: 10.30632/PJV61N1-2020a2. Petrophysics Vol. 61 No. 1 (Feb 2020).
src2021_02/article4_obm_dielectric_fracture_fill -- Article 4: Identifying Fracture-Filling
  Material in Oil-Based Mud With Dielectric Borehole Imaging. Schlicht, Zhang, Luling, Graham,
  Cournot, Sadownyk (2021). DOI: 10.30632/PJV62N1-2021a3. Petrophysics Vol. 62 No. 1 (Feb 2021).
src2021_02/article8_injectite_em_3d_inversion -- Article 8: Mapping Complex Injectite Bodies With
  Multiwell Electromagnetic 3D Inversion Data. Clegg, Eriksen, Best, Tollefsen, Kowicki, Marchant
  (2021). DOI: 10.30632/PJV62N1-2021a7. Petrophysics Vol. 62 No. 1 (Feb 2021).
src2021_06/article5_lwd_colocated_antenna -- Article 5: First LWD Co-Located Antenna Sensors for
  Real-Time Anisotropy and Dip Angle Determination, Yielding Better Look-Ahead Detection. Bittar,
  Wu, Ma, Pan, Fan, Griffing, Lozinsky (2021). DOI: 10.30632/PJV62N3-2021a4. Petrophysics Vol. 62
  No. 3 (Jun 2021).
src2021_08/article1_thinly_bedded_petrophysics -- Article 1: Tutorial - Petrophysics of Thinly
  Bedded Formations. Aldred (2021). DOI: 10.30632/PJV62N4-2021t1.
src2023_06/article7_dielectric_nmr -- Funk, J., Myers, M., Hathon, L. "NMR-Mapped Distributions of
  Dielectric Dispersion" Petrophysics, Vol. 64, No. 3 (June 2023), pp. 421-437 DOI:
  10.30632/PJV64N3-2023a7.
src2023_08/article4_obm_imager_inversion -- Implements a simplified version of the high-definition
  OBM borehole-imager forward / inverse model from: Chen, Y.-H., Zhang, T., Bloemenkamp, R., Liang,
  L. (2023). "Fracture Imaging and Response Characterization of the High-Definition Oil-Based Mud
  Borehole Imagers Through Modeling and Inversion", Petrophysics, Vol. 64, No. 4, pp. 544-554. DOI:
  10.30632/PJV64N4-2023a4.
src2025_04/udar_look_ahead -- UDAR Look-Ahead-While-Drilling Technology Assessment. Based on:
  Cuadros et al., "Look-Ahead-While-Drilling Technology Assessment for Early Hazards Identification
  in Presalt Offshore Brazil", Petrophysics, Vol. 66, No. 2, April 2025, pp. 190–211.
src2026_06/a01_carbonate_pore_type_dielectric -- AlZoukani, A. M., Al-Hamad, M., and Abdallah, W.
  (2026). Unlocking the Effect of Pore Types on Dielectric Permittivity in Carbonates.
  Petrophysics, 67(3), 470-481. DOI: 10.30632/PJV67N3-2026a1 (Best Petrophysics Papers from MEOS
  GEO 2025; orig. SPE-227798-MS).
src2026_06/a02_mf_dielectric_fracture_sensitivity -- Al-Qouzi, W., Hassan, A., Attia, H., El-
  Husseiny, A., and Mahmoud, M. (2026). Experimental Study of Sensitivity of Multifrequency (MF)
  Dielectric Measurements to Hydraulic Fractures in Sandstone and Carbonate Formations.
  Petrophysics, 67(3), 482-508. DOI: 10.30632/PJV67N3-2026a2 (orig. SPE-227827-MS).
src2026_06/a05_udar_anisotropy_sensitivity -- Bower, M., Xie, H., Wang, G. L., Leveque, S., and
  Dolan, J. (2026). Anisotropy Sensitivity in Ultradeep Azimuthal Resistivity Technologies.
  Petrophysics, 67(3), 544-559. DOI: 10.30632/PJV67N3-2026a5.
src2026_06/a09_mf_dielectric_emulsion -- Albenayyan, N., Hassan, A., El-Husseiny, A., and Mahmoud,
  M. (2026). A Method to Characterize the Emulsions in Oil-Water Production Wells Using
  Multifrequency Dielectric Technique. Petrophysics, 67(3), 594-618. DOI: 10.30632/PJV67N3-2026a9
  (cover paper).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]
_Complex = NDArray[np.complex128]
_Num = NDArray[Any]  # float64 for real inputs, complex128 for complex inputs

# Physical constants (SI).
EPS0 = 8.8541878128e-12  # vacuum permittivity, F/m
MU0 = 4.0e-7 * np.pi  # vacuum permeability, H/m

# Nepers -> decibels for field (amplitude) attenuation, per the plan's
# induction-limit definition ``attenuation_db = 8.686 * L/delta``.
_NEPER_TO_DB = 8.686


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


def _num(x: ArrayLike) -> _Num:
    """Convert to a float64 array, or complex128 if the input is complex."""
    a = np.asarray(x)
    if np.iscomplexobj(a):
        return a.astype(np.complex128)
    return a.astype(np.float64)


# ==========================================================================
# 1. Complex permittivity algebra (-j loss convention)
# ==========================================================================


def complex_permittivity(
    eps_real: ArrayLike,
    *,
    sigma: ArrayLike = 0.0,
    freq_hz: ArrayLike | None = None,
    eps_imag: ArrayLike = 0.0,
) -> _Complex:
    """Complex permittivity ``eps* = eps' - j*(eps'' + sigma/(omega*eps0))``.

    ``eps_real`` is the real part ``eps'``; ``eps_imag`` is any dielectric loss
    ``eps''`` already in permittivity units; ``sigma`` (S/m) adds the ohmic loss
    ``sigma/(omega*eps0)``.  ``freq_hz`` is required only when ``sigma`` is
    nonzero.  Returns a complex array in the ``-j`` engineering convention.

    Sources: src2018_04/article7_bakken_dielectric_dispersion,
    src2026_06/a01_carbonate_pore_type_dielectric.
    """
    er = _arr(eps_real)
    loss = _arr(eps_imag)
    sig = _arr(sigma)
    if np.any(sig != 0.0):
        if freq_hz is None:
            raise ValueError("freq_hz is required when sigma is nonzero")
        w = 2.0 * np.pi * _arr(freq_hz)
        loss = loss + sig / (w * EPS0)
    return np.asarray(er - 1j * loss, np.complex128)


def imag_permittivity_from_sigma(sigma: ArrayLike, freq_hz: ArrayLike) -> _Float:
    """Loss permittivity from conductivity: ``eps'' = sigma/(2*pi*f*eps0)``.

    Sources: src2026_06/a01_carbonate_pore_type_dielectric.
    """
    w = 2.0 * np.pi * _arr(freq_hz)
    return np.asarray(_arr(sigma) / (w * EPS0))


def sigma_from_imag_permittivity(eps_imag: ArrayLike, freq_hz: ArrayLike) -> _Float:
    """Conductivity from loss permittivity: ``sigma = eps''*2*pi*f*eps0``.

    Sources: src2026_06/a02_mf_dielectric_fracture_sensitivity.
    """
    w = 2.0 * np.pi * _arr(freq_hz)
    return np.asarray(_arr(eps_imag) * w * EPS0)


def loss_tangent(eps_star: ArrayLike) -> _Float:
    """Loss tangent ``tan(delta) = -Im(eps*)/Re(eps*)`` (-j convention).

    Sources: src2026_06/a01_carbonate_pore_type_dielectric.
    """
    z = np.asarray(eps_star, np.complex128)
    return np.asarray(-z.imag / z.real)


def impedivity(eps_r: ArrayLike, sigma: ArrayLike, freq_hz: ArrayLike) -> _Complex:
    """Complex impedivity ``1/(j*omega*eps0*eps_r + sigma)`` (inverse admittivity).

    Sources: src2023_08/article4_obm_imager_inversion.
    """
    w = 2.0 * np.pi * _arr(freq_hz)
    admittivity = 1j * w * EPS0 * _arr(eps_r) + _arr(sigma)
    return np.asarray(1.0 / admittivity, np.complex128)


def water_permittivity(
    rw: ArrayLike, freq_hz: ArrayLike, *, eps_real: float = 55.0, eps_dl: float = 15.0
) -> _Complex:
    """Brine complex permittivity from ``rw`` (ohm.m) at GHz-band frequencies.

    ``eps* = eps_real - j*(eps_dl + sigma_w/(omega*eps0))`` with the brine
    conductivity ``sigma_w = 1/rw``.  ``eps_real`` is the real permittivity of
    water and ``eps_dl`` an added dielectric-loss (double-layer) term.
    """
    w = 2.0 * np.pi * _arr(freq_hz)
    sigma_w = 1.0 / _arr(rw)
    loss = eps_dl + sigma_w / (w * EPS0)
    return np.asarray(eps_real - 1j * loss, np.complex128)


# ==========================================================================
# 2. Dielectric relaxation / dispersion models
# ==========================================================================


def debye(freq_hz: ArrayLike, *, eps_inf: float, eps_s: float, tau: float) -> _Complex:
    """Single-pole Debye relaxation ``eps_inf + (eps_s-eps_inf)/(1 + j*omega*tau)``.

    Sources: src2014_08/article5_dielectric_permittivity_shales,
    src2023_06/article7_dielectric_nmr.
    """
    w = 2.0 * np.pi * _arr(freq_hz)
    return np.asarray(eps_inf + (eps_s - eps_inf) / (1.0 + 1j * w * tau), np.complex128)


def cole_cole(
    freq_hz: ArrayLike, *, eps_inf: float, eps_s: float, tau: float, alpha: float = 0.0
) -> _Complex:
    """Cole-Cole relaxation with distribution exponent ``alpha`` (0 -> Debye).

    ``eps_inf + (eps_s-eps_inf)/(1 + (j*omega*tau)**(1-alpha))``.

    Sources: src2014_08/article5_dielectric_permittivity_shales,
    src2026_06/a02_mf_dielectric_fracture_sensitivity.
    """
    w = 2.0 * np.pi * _arr(freq_hz)
    jwt = 1j * w * tau
    return np.asarray(eps_inf + (eps_s - eps_inf) / (1.0 + jwt ** (1.0 - alpha)), np.complex128)


def havriliak_negami(
    freq_hz: ArrayLike,
    *,
    eps_inf: float,
    eps_s: float,
    tau: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> _Complex:
    """Havriliak-Negami relaxation (``alpha=beta=1`` -> Debye).

    ``eps_inf + (eps_s-eps_inf)/(1 + (j*omega*tau)**alpha)**beta``.

    Sources: src2023_06/article7_dielectric_nmr.
    """
    w = 2.0 * np.pi * _arr(freq_hz)
    jwt = 1j * w * tau
    return np.asarray(eps_inf + (eps_s - eps_inf) / (1.0 + jwt**alpha) ** beta, np.complex128)


def cole_cole_resistivity(
    freq_hz: ArrayLike, *, rho0: float, chargeability: float, tau: float, c: float
) -> _Complex:
    """Pelton (1978) Cole-Cole complex resistivity (induced-polarization form).

    ``rho(omega) = rho0*(1 - m*(1 - 1/(1 + (j*omega*tau)**c)))`` with DC
    resistivity ``rho0``, chargeability ``m``, time constant ``tau`` and
    frequency exponent ``c``.

    Sources: src2017_06/article5_complex_resistivity_dispersion.
    """
    w = 2.0 * np.pi * _arr(freq_hz)
    jwtc = (1j * w * tau) ** c
    return np.asarray(rho0 * (1.0 - chargeability * (1.0 - 1.0 / (1.0 + jwtc))), np.complex128)


# ==========================================================================
# 3. Volumetric mixing laws (CRIM family)
# ==========================================================================


def mix_power_law(fractions: ArrayLike, eps_components: ArrayLike, *, alpha: float = 0.5) -> _Num:
    """Lichtenecker-Rother power-law mixing over the leading (component) axis.

    ``eps_mix = (sum_i f_i * eps_i**alpha)**(1/alpha)``.  ``alpha=0.5`` is the
    Complex Refractive Index Method (CRIM / time-average of sqrt).  Complex-safe:
    ``eps_components`` may carry loss.  ``fractions`` and ``eps_components``
    broadcast with the component index along ``axis=0``.

    Sources: src2014_12/article4_porescale_dielectric,
    src2018_04/article6_dielectric_matrix_crim_cda,
    src2021_02/article4_obm_dielectric_fracture_fill.
    """
    f = _num(fractions)
    e = _num(eps_components)
    mixed = np.sum(f * e**alpha, axis=0)
    return np.asarray(mixed ** (1.0 / alpha))


def crim(
    phi: ArrayLike,
    sw: ArrayLike = 1.0,
    *,
    eps_w: float = 78.0,
    eps_hc: float = 2.2,
    eps_matrix: float = 5.0,
    alpha: float = 0.5,
) -> _Float:
    """CRIM 3-component rock forward model (water / hydrocarbon / matrix).

    ``eps_mix = (phi*Sw*eps_w**a + phi*(1-Sw)*eps_hc**a + (1-phi)*eps_m**a)**(1/a)``
    with ``a = alpha`` (0.5 = classic CRIM square-root mixing).

    Sources: src2014_10/article6_dielectric_oil_movability,
    src2014_12/article3_nmr_dielectric_limestone, src2014_12/article4_porescale_dielectric,
    src2016_10/article4_dielectric_archie_greenriver,
    src2018_04/article6_dielectric_matrix_crim_cda,
    src2018_04/article7_bakken_dielectric_dispersion, src2019_12/article3_lwd_permittivity.
    """
    phi_a = _arr(phi)
    sw_a = _arr(sw)
    mixed = (
        phi_a * sw_a * eps_w**alpha
        + phi_a * (1.0 - sw_a) * eps_hc**alpha
        + (1.0 - phi_a) * eps_matrix**alpha
    )
    return np.asarray(mixed ** (1.0 / alpha))


def sw_from_permittivity(
    eps_meas: ArrayLike,
    phi: ArrayLike,
    *,
    eps_w: float,
    eps_hc: float,
    eps_matrix: float,
    alpha: float = 0.5,
    clip: bool = True,
) -> _Float:
    """Water saturation by inverting :func:`crim` for ``Sw``.

    ``Sw = (eps_meas**a - (1-phi)*eps_m**a - phi*eps_hc**a) / (phi*(eps_w**a - eps_hc**a))``.
    ``clip=True`` bounds the result to ``[0, 1]``.

    Sources: src2014_12/article3_nmr_dielectric_limestone,
    src2014_12/article4_porescale_dielectric, src2018_04/article7_bakken_dielectric_dispersion,
    src2019_12/article3_lwd_permittivity.
    """
    em = _arr(eps_meas)
    phi_a = _arr(phi)
    num = em**alpha - (1.0 - phi_a) * eps_matrix**alpha - phi_a * eps_hc**alpha
    sw = num / (phi_a * (eps_w**alpha - eps_hc**alpha))
    if clip:
        sw = np.clip(sw, 0.0, 1.0)
    return np.asarray(sw)


def bvw_from_permittivity(
    eps_meas: ArrayLike,
    phi: ArrayLike,
    *,
    eps_w: float,
    eps_hc: float,
    eps_matrix: float,
    alpha: float = 0.5,
) -> _Float:
    """Bulk volume water ``phi*Sw`` from :func:`crim`, unclipped (salinity-robust).

    ``BVW = (eps_meas**a - (1-phi)*eps_m**a - phi*eps_hc**a) / (eps_w**a - eps_hc**a)``.

    Sources: src2016_10/article4_dielectric_archie_greenriver.
    """
    em = _arr(eps_meas)
    phi_a = _arr(phi)
    num = em**alpha - (1.0 - phi_a) * eps_matrix**alpha - phi_a * eps_hc**alpha
    return np.asarray(num / (eps_w**alpha - eps_hc**alpha))


def water_filled_porosity(
    eps_meas: ArrayLike,
    *,
    eps_matrix: float,
    eps_w: float,
    alpha: float = 0.5,
    clip: bool = True,
) -> _Float:
    """Water-filled porosity from a two-component (water/matrix) CRIM inversion.

    ``phi_w = (eps_meas**a - eps_m**a) / (eps_w**a - eps_m**a)``; ``clip=True``
    bounds the result to ``[0, 1]``.

    Sources: src2014_10/article6_dielectric_oil_movability,
    src2018_04/article6_dielectric_matrix_crim_cda.
    """
    em = _arr(eps_meas)
    phi_w = (em**alpha - eps_matrix**alpha) / (eps_w**alpha - eps_matrix**alpha)
    if clip:
        phi_w = np.clip(phi_w, 0.0, 1.0)
    return np.asarray(phi_w)


# ==========================================================================
# 4. Effective-medium theories
# ==========================================================================


def maxwell_garnett(
    eps_host: ArrayLike, eps_incl: ArrayLike, f_incl: ArrayLike, *, depol: float = 1.0 / 3.0
) -> _Num:
    """Maxwell-Garnett effective permittivity with depolarization ``depol``.

    ``eps_eff = eps_h*(1 + f*ds/(eps_h + N*(1-f)*ds))`` where ``ds = eps_i-eps_h``
    and ``N = depol`` (``1/3`` for spheres).  Complex-safe -- also valid for
    conductivity mixing.

    Sources: src2018_06/article4_clay_network_resistivity_saturation,
    src2026_06/a09_mf_dielectric_emulsion.
    """
    eh = _num(eps_host)
    ei = _num(eps_incl)
    f = _num(f_incl)
    ds = ei - eh
    return np.asarray(eh * (1.0 + f * ds / (eh + depol * (1.0 - f) * ds)))


def bruggeman_symmetric(
    fractions: ArrayLike, eps_components: ArrayLike, *, iterations: int = 60
) -> _Num:
    """Symmetric Bruggeman effective-medium permittivity (N phases, spheres).

    Solves ``sum_i f_i*(eps_i - eps_eff)/(eps_i + 2*eps_eff) = 0`` by Newton
    iteration from the arithmetic-mean seed.  Complex-capable.  ``fractions`` and
    ``eps_components`` index the phases along ``axis=0``.
    """
    f = _num(fractions)
    e = _num(eps_components)
    eff = np.sum(f * e, axis=0)  # arithmetic-mean seed
    for _ in range(iterations):
        denom = e + 2.0 * eff
        resid = np.sum(f * (e - eff) / denom, axis=0)
        deriv = -3.0 * np.sum(f * e / denom**2, axis=0)
        eff = eff - resid / deriv
    return np.asarray(eff)


def hanai_bruggeman(
    eps_host: ArrayLike, eps_incl: ArrayLike, f_incl: ArrayLike, *, iterations: int = 80
) -> _Num:
    """Asymmetric (Hanai) Bruggeman effective permittivity by robust Newton solve.

    Solves ``((eps_eff-eps_i)/(eps_h-eps_i))*(eps_h/eps_eff)**(1/3) = 1 - f`` for
    the effective permittivity ``eps_eff`` (inclusion fraction ``f``).
    Complex-capable; no absolute-value damping.
    """
    eh = _num(eps_host)
    ei = _num(eps_incl)
    f = _num(f_incl)
    target = 1.0 - f
    eff = (1.0 - f) * eh + f * ei  # arithmetic seed
    c = eh ** (1.0 / 3.0) / (eh - ei)
    for _ in range(iterations):
        g = c * (eff - ei) * eff ** (-1.0 / 3.0) - target
        dg = c * eff ** (-4.0 / 3.0) * ((2.0 / 3.0) * eff + (1.0 / 3.0) * ei)
        eff = eff - g / dg
    return np.asarray(eff)


def depolarization_spheroid(aspect: ArrayLike) -> tuple[_Float, _Float, _Float]:
    """Osborn depolarization factors ``(Lx, Ly, Lz)`` for a spheroid.

    ``aspect = c/a`` is the polar/equatorial semi-axis ratio: ``1`` -> sphere
    ``(1/3, 1/3, 1/3)``; ``<1`` -> oblate (disk); ``>1`` -> prolate (needle).
    ``Lz`` is along the symmetry (polar) axis and ``Lx = Ly = (1-Lz)/2``.  The
    three factors sum to 1.

    Sources: src2018_06/article5_wideband_em_dem_permittivity.
    """
    r = np.asarray(aspect, np.float64)
    lz = np.full(r.shape, 1.0 / 3.0)

    obl = r < 1.0
    pro = r > 1.0

    # Oblate: e = sqrt(1 - r^2); Lz = (1/e^2)*(1 - sqrt(1-e^2)/e * arcsin(e)).
    ro = np.where(obl, r, 0.5)  # safe dummy off-branch
    eo = np.sqrt(1.0 - ro**2)
    eo_s = np.where(eo == 0.0, 1.0, eo)
    lz_obl = (1.0 / eo_s**2) * (1.0 - np.sqrt(1.0 - eo**2) / eo_s * np.arcsin(eo))

    # Prolate: e = sqrt(1 - 1/r^2); Lz = ((1-e^2)/e^2)*(-1 + (1/(2e))*ln((1+e)/(1-e))).
    rp = np.where(pro, r, 2.0)  # safe dummy off-branch
    ep = np.sqrt(1.0 - 1.0 / rp**2)
    ep_s = np.where(ep == 0.0, 1.0, ep)
    lz_pro = ((1.0 - ep**2) / ep_s**2) * (
        -1.0 + (1.0 / (2.0 * ep_s)) * np.log((1.0 + ep) / (1.0 - ep))
    )

    lz = np.where(pro, lz_pro, np.where(obl, lz_obl, lz))
    lx = (1.0 - lz) / 2.0
    return np.asarray(lx), np.asarray(lx), np.asarray(lz)


# ==========================================================================
# 5. Induction / propagation-logging relations
# ==========================================================================


def skin_depth(rho: ArrayLike, freq_hz: ArrayLike, *, mu_r: float = 1.0) -> _Float:
    """EM skin depth ``delta = sqrt(2*rho/(omega*mu))`` (``mu = mu0*mu_r``).

    Sources: src2016_10/article1_emla_lookahead_resistivity,
    src2016_10/article4_dielectric_archie_greenriver,
    src2020_02/article2_russian_western_resistivity,
    src2021_02/article8_injectite_em_3d_inversion, src2021_06/article5_lwd_colocated_antenna,
    src2025_04/udar_look_ahead, src2026_06/a05_udar_anisotropy_sensitivity.
    """
    w = 2.0 * np.pi * _arr(freq_hz)
    return np.asarray(np.sqrt(2.0 * _arr(rho) / (w * MU0 * mu_r)))


def induction_number(spacing_m: ArrayLike, rho: ArrayLike, freq_hz: ArrayLike) -> _Float:
    """Induction number ``L/delta`` (near-field vs far-field discriminator).

    Sources: src2026_06/a05_udar_anisotropy_sensitivity.
    """
    return np.asarray(_arr(spacing_m) / skin_depth(rho, freq_hz))


def complex_wavenumber(
    freq_hz: ArrayLike, sigma: ArrayLike, eps_r: ArrayLike, *, mu_r: float = 1.0
) -> tuple[_Float, _Float]:
    """Exact lossy-medium wavenumber ``(k_r, k_i) = (beta, alpha)``.

    Phase constant ``beta`` and attenuation constant ``alpha`` of ``k = beta -
    j*alpha`` from the loss tangent ``p = sigma/(omega*eps0*eps_r)``:
    ``k_r = base*sqrt((sqrt(1+p^2)+1)/2)`` and
    ``k_i = base*sqrt((sqrt(1+p^2)-1)/2)`` with ``base = omega*sqrt(mu*eps0*eps_r)``.
    """
    w = 2.0 * np.pi * _arr(freq_hz)
    eps = EPS0 * _arr(eps_r)
    base = w * np.sqrt(MU0 * mu_r * eps)
    p = _arr(sigma) / (w * eps)
    root = np.sqrt(1.0 + p**2)
    k_r = base * np.sqrt((root + 1.0) / 2.0)
    k_i = base * np.sqrt((root - 1.0) / 2.0)
    return np.asarray(k_r), np.asarray(k_i)


def sigma_eps_from_wavenumber(
    k_r: ArrayLike, k_i: ArrayLike, freq_hz: ArrayLike, *, mu_r: float = 1.0
) -> tuple[_Float, _Float]:
    """Closed-form inverse of :func:`complex_wavenumber` -> ``(sigma, eps_r)``.

    From ``k^2 = omega^2*mu*eps*``: ``eps_r = (k_r^2 - k_i^2)/(omega^2*mu*eps0)``
    and ``sigma = 2*k_r*k_i/(omega*mu)``.
    """
    w = 2.0 * np.pi * _arr(freq_hz)
    mu = MU0 * mu_r
    kr = _arr(k_r)
    ki = _arr(k_i)
    eps_r = (kr**2 - ki**2) / (w**2 * mu * EPS0)
    sigma = 2.0 * kr * ki / (w * mu)
    return np.asarray(sigma), np.asarray(eps_r)


def phase_shift_deg(rho: ArrayLike, freq_hz: ArrayLike, spacing_m: ArrayLike) -> _Float:
    """Induction-limit phase shift over ``spacing_m``: ``degrees(L/delta)``.

    Sources: src2021_06/article5_lwd_colocated_antenna.
    """
    return np.asarray(np.degrees(induction_number(spacing_m, rho, freq_hz)))


def attenuation_db(rho: ArrayLike, freq_hz: ArrayLike, spacing_m: ArrayLike) -> _Float:
    """Induction-limit amplitude attenuation over ``spacing_m``: ``8.686*L/delta`` dB.

    Sources: src2021_06/article5_lwd_colocated_antenna.
    """
    return np.asarray(_NEPER_TO_DB * induction_number(spacing_m, rho, freq_hz))


def resistivity_from_phase(
    phase_deg: ArrayLike, freq_hz: ArrayLike, spacing_m: ArrayLike
) -> _Float:
    """Induction-limit inverse of :func:`phase_shift_deg` -> apparent ``rho``.

    ``rho = L^2*omega*mu0/(2*phase_rad^2)`` with ``phase_rad = radians(phase_deg)``.

    Sources: src2021_06/article5_lwd_colocated_antenna.
    """
    w = 2.0 * np.pi * _arr(freq_hz)
    phase_rad = np.radians(_arr(phase_deg))
    spacing = _arr(spacing_m)
    return np.asarray(spacing**2 * w * MU0 / (2.0 * phase_rad**2))


def attenuation_phase_from_voltages(v_near: ArrayLike, v_far: ArrayLike) -> tuple[_Float, _Float]:
    """Attenuation (dB) and phase shift (deg) from near/far complex voltages.

    ``ratio = v_near/v_far``; ``atten_db = 20*log10|ratio|`` and
    ``phase_deg = degrees(angle(ratio))``.

    Sources: src2016_10/article1_emla_lookahead_resistivity.
    """
    ratio = np.asarray(v_near, np.complex128) / np.asarray(v_far, np.complex128)
    atten_db = 20.0 * np.log10(np.abs(ratio))
    phase_deg = np.degrees(np.angle(ratio))
    return np.asarray(atten_db), np.asarray(phase_deg)


# ==========================================================================
# 6. Resistivity anisotropy and radial (Doll) geometry
# ==========================================================================


def anisotropy_coefficient(rh: ArrayLike, rv: ArrayLike) -> _Float:
    """Resistivity anisotropy coefficient ``lambda = sqrt(Rv/Rh)`` (canonical).

    Sources: src2014_04/article5_deepwater_formation_evaluation,
    src2014_10/article1_lwd_inversion_anisotropy,
    src2016_02/article5_shaly_sand_graphical_solutions,
    src2020_02/article2_russian_western_resistivity,
    src2020_02/article3_hdil_array_induction_anisotropic,
    src2021_08/article1_thinly_bedded_petrophysics, src2026_06/a05_udar_anisotropy_sensitivity.
    """
    return np.asarray(np.sqrt(_arr(rv) / _arr(rh)))


def apparent_resistivity_dip(rh: ArrayLike, rv: ArrayLike, dip_deg: ArrayLike) -> _Float:
    """Apparent resistivity at relative dip ``theta``: ``Rh*sqrt(cos^2 + (Rv/Rh)*sin^2)``.

    Sources: src2020_02/article3_hdil_array_induction_anisotropic,
    src2021_08/article1_thinly_bedded_petrophysics.
    """
    rh_a = _arr(rh)
    rv_a = _arr(rv)
    theta = np.radians(_arr(dip_deg))
    return np.asarray(rh_a * np.sqrt(np.cos(theta) ** 2 + (rv_a / rh_a) * np.sin(theta) ** 2))


def doll_radial_geometric_factor(r: ArrayLike, spacing_m: ArrayLike) -> _Float:
    """Doll radial geometric factor ``G = r^2/(r^2 + (L/2)^2)`` (integrated to radius r).

    Sources: src2014_12/article6_array_induction_geometric_factors.
    """
    r_a = _arr(r)
    half = _arr(spacing_m) / 2.0
    return np.asarray(r_a**2 / (r_a**2 + half**2))


def apparent_conductivity_two_zone(
    sigma_xo: ArrayLike, sigma_t: ArrayLike, r_invasion: ArrayLike, spacing_m: ArrayLike
) -> _Float:
    """Doll two-zone (invaded + virgin) apparent conductivity.

    ``sigma_a = G_xo*sigma_xo + (1-G_xo)*sigma_t`` with the radial geometric
    factor ``G_xo`` of the invasion radius from :func:`doll_radial_geometric_factor`.

    Sources: src2014_12/article6_array_induction_geometric_factors.
    """
    g = doll_radial_geometric_factor(r_invasion, spacing_m)
    return np.asarray(g * _arr(sigma_xo) + (1.0 - g) * _arr(sigma_t))
