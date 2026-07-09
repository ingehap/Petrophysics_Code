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
    """Saturation sensitivity ``dSigma/dSw = phi*(Sigma_w - Sigma_hc)`` (c.u.)."""
    return np.asarray(_arr(phi) * (sigma_w - sigma_hc))


def sigma_w_from_salinity(
    ppm_nacl: ArrayLike, *, temperature_c: float = 75.0, model: str = "fitz2023"
) -> _Float:
    """Brine capture cross-section ``Sigma_w`` (c.u.) from NaCl salinity (ppm).

    ``model='fitz2023'``: ``(22 + 750*s)*(1 - 8e-4*(T-75))`` with ``s=ppm/1e6``.
    ``model='linear450k'``: ``22 + (220-22)*ppm/450000``.
    """
    ppm = _arr(ppm_nacl)
    if model == "fitz2023":
        s = ppm / 1e6
        return np.asarray((22.0 + 750.0 * s) * (1.0 - 0.0008 * (temperature_c - 75.0)))
    if model == "linear450k":
        return np.asarray(22.0 + (220.0 - 22.0) * ppm / 450000.0)
    raise ValueError(f"unknown model {model!r}; use 'fitz2023' or 'linear450k'")


def number_density(rho_g_cc: ArrayLike, wfrac: ArrayLike, atomic_mass: ArrayLike) -> _Float:
    """Atomic number density ``N = rho*NA*w/A`` (1/cm3) for a mass fraction ``w``."""
    return np.asarray(_arr(rho_g_cc) * NA * _arr(wfrac) / _arr(atomic_mass))


def macroscopic_sigma(
    number_densities: ArrayLike, micro_barns: ArrayLike, *, units: str = "cu"
) -> _Float:
    """Macroscopic capture ``Sigma`` from number densities and micro cross-sections.

    ``Sigma = sum_i N_i * sigma_i`` with ``sigma`` in barns (1 barn = 1e-24 cm2);
    ``units='cu'`` returns c.u. (cm^-1 * 1e3), ``units='cm'`` returns cm^-1.
    """
    sigma_cm = np.sum(_arr(number_densities) * _arr(micro_barns) * 1e-24)
    if units == "cu":
        return np.asarray(sigma_cm * 1e3)
    if units == "cm":
        return np.asarray(sigma_cm)
    raise ValueError(f"unknown units {units!r}; use 'cu' or 'cm'")


def sigma_from_tau(tau_us: ArrayLike) -> _Float:
    """Capture cross-section from thermal decay time ``Sigma = 4550/tau`` (c.u.)."""
    return np.asarray(SIGMA_TAU_CONST / _arr(tau_us))


def tau_from_sigma(sigma_cu: ArrayLike) -> _Float:
    """Thermal decay time from capture cross-section ``tau = 4550/Sigma`` (us)."""
    return np.asarray(SIGMA_TAU_CONST / _arr(sigma_cu))


def pnc_decay(t_us: ArrayLike, n0: float, sigma_cu: float, *, background: float = 0.0) -> _Float:
    """Pulsed-neutron capture decay ``N(t) = N0*exp(-Sigma*v*t) + background``.

    ``Sigma`` (c.u.) is converted to cm^-1 (``*1e-3``), ``v`` is the thermal-neutron
    velocity, and ``t`` (us) to seconds (``*1e-6``).
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
    """Beer-Lambert transmitted intensity ``I = I0*exp(-mu*x)``."""
    return np.asarray(_arr(i0) * np.exp(-_arr(mu) * _arr(x)))


def mu_from_intensity(i0: ArrayLike, i: ArrayLike, x: ArrayLike) -> _Float:
    """Linear attenuation coefficient ``mu = ln(I0/I)/x`` from an intensity pair."""
    return np.asarray(np.log(_arr(i0) / _arr(i)) / _arr(x))


def attenuation_map(i: ArrayLike, i_ref: ArrayLike) -> _Float:
    """Optical-density attenuation ``-ln(I/I_ref)`` (zero-guarded ratio)."""
    ratio = np.divide(_arr(i), _arr(i_ref))
    return np.asarray(-np.log(np.clip(ratio, 1e-30, None)))


def gamma_count(
    rho: ArrayLike, *, n0: float = 1e6, mu_mass: float = 0.06, spacing_cm: float = 30.0
) -> _Float:
    """Gamma count rate vs density ``N = N0*exp(-mu_mass*rho*spacing)``."""
    return np.asarray(n0 * np.exp(-mu_mass * _arr(rho) * spacing_cm))


def density_from_count(
    count: ArrayLike, *, n0: float = 1e6, mu_mass: float = 0.06, spacing_cm: float = 30.0
) -> _Float:
    """Density from a gamma count ``rho = ln(N0/N)/(mu_mass*spacing)`` (inverse)."""
    return np.asarray(np.log(n0 / _arr(count)) / (mu_mass * spacing_cm))


# ==========================================================================
# 3. Gamma-gamma density logging
# ==========================================================================


def dual_detector_density(near: ArrayLike, far: ArrayLike, *, a: float, b: float) -> _Float:
    """Dual-detector density ``rho = a + b*ln(near/far)`` (count-ratio calibration)."""
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
    """
    d = _arr(rho_ls) - _arr(rho_ss)
    c0, c1, c2 = rib_coeffs
    drho = c0 + c1 * d + c2 * d**2
    return np.asarray(_arr(rho_ls) + drho), np.asarray(drho)


def electron_density_index(z: ArrayLike, a: ArrayLike, rho_m: ArrayLike) -> _Float:
    """Electron density ``rho_e = (2*Z/A)*rho_m`` of a single element/compound."""
    return np.asarray((2.0 * _arr(z) / _arr(a)) * _arr(rho_m))


def electron_density_mixture(
    z: ArrayLike, a: ArrayLike, mass_frac: ArrayLike, rho_m: ArrayLike
) -> _Float:
    """Mixture electron density ``rho_e = 2*sum(w_i*Z_i/A_i)*rho_m``."""
    zam = np.sum(_arr(mass_frac) * _arr(z) / _arr(a))
    return np.asarray(2.0 * zam * _arr(rho_m))


def rhob_from_rhoe(rho_e: ArrayLike) -> _Float:
    """Bulk density from electron density ``rho_b = 1.0704*rho_e - 0.1883``."""
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
    """
    ck, cu, cth = coeff
    return np.asarray(ck * _arr(k_pct) + cu * _arr(u_ppm) + cth * _arr(th_ppm))


def cgr_api(
    k_pct: ArrayLike, th_ppm: ArrayLike, *, coeff: tuple[float, float] = (16.0, 4.0)
) -> _Float:
    """Computed (uranium-free) gamma ray API ``CGR = ck*K + cth*Th``."""
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
    """Volumetric hydrogen-index log ``HI = phi*HI_fluid + (1-phi)*HI_matrix``."""
    phi_a = _arr(phi)
    return np.asarray(phi_a * hi_fluid + (1.0 - phi_a) * hi_matrix)


def phi_from_hi(hi_log: ArrayLike, *, hi_matrix: float, hi_fluid: float = 1.0) -> _Float:
    """Porosity from a hydrogen-index log ``phi = (HI - HI_matrix)/(HI_fluid - HI_matrix)``."""
    return np.asarray((_arr(hi_log) - hi_matrix) / (hi_fluid - hi_matrix))


def phi_hi_correction(phi_apparent: ArrayLike, hi: ArrayLike) -> _Float:
    """Hydrogen-index porosity correction ``phi = phi_apparent/HI`` (neutron & NMR)."""
    return np.asarray(_arr(phi_apparent) / _arr(hi))


def collision_parameter(a_mass: ArrayLike) -> _Float:
    """Neutron collision parameter ``alpha = ((A-1)/(A+1))**2``."""
    a = _arr(a_mass)
    return np.asarray(((a - 1.0) / (a + 1.0)) ** 2)


def average_lethargy_gain(a_mass: ArrayLike) -> _Float:
    """Mean logarithmic energy decrement ``xi = 1 + alpha/(1-alpha)*ln(alpha)``.

    The hydrogen limit ``A=1`` (``alpha=0``) returns ``xi=1``.
    """
    alpha = collision_parameter(a_mass)
    safe = np.where(alpha > 0.0, alpha, 1.0)
    xi = 1.0 + alpha / (1.0 - alpha) * np.log(safe)
    return np.asarray(np.where(alpha > 0.0, xi, 1.0))


def moderating_power(a_mass: ArrayLike, sigma_s: ArrayLike) -> _Float:
    """Moderating power ``xi*Sigma_s`` (slowing-down power)."""
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
    """Carbon-oxygen ratio ``COR = Y_C/Y_O``."""
    return np.asarray(_arr(c_yield) / _arr(o_yield))


def so_from_co(
    cor: ArrayLike,
    cor_water: ArrayLike,
    cor_oil: ArrayLike,
    *,
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> _Float:
    """Oil saturation from C/O endpoint interpolation ``So = (COR-COR_w)/(COR_o-COR_w)``."""
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
    """
    phi_a = _arr(phi)
    carbon = phi_a * (_arr(so) * c_oil + _arr(sg) * c_gas) + (1.0 - phi_a) * c_mat
    oxygen = phi_a * _arr(sw) * o_w + (1.0 - phi_a) * o_mat
    return np.asarray(carbon / (oxygen + 1e-9))


def yields_to_weights(fy2w: ArrayLike, s: ArrayLike, y: ArrayLike) -> _Float:
    """Elemental weight fraction from a spectroscopy yield ``W = FY2W*S*Y``."""
    return np.asarray(_arr(fy2w) * _arr(s) * _arr(y))


def weights_to_yields(fy2w: ArrayLike, s: ArrayLike, w: ArrayLike) -> _Float:
    """Spectroscopy yield from a weight fraction ``Y = W/(FY2W*S)`` (inverse)."""
    return np.asarray(_arr(w) / (_arr(fy2w) * _arr(s)))


def toc_from_yield(y_toc: ArrayLike, calib: ArrayLike) -> _Float:
    """Total organic carbon from a carbon yield ``TOC = calib*Y_TOC``."""
    return np.asarray(_arr(calib) * _arr(y_toc))


# ==========================================================================
# 7. Counting statistics and radioactive decay
# ==========================================================================


def counting_precision(counts: ArrayLike) -> _Float:
    """Relative counting precision ``1/sqrt(N)`` (Poisson)."""
    return np.asarray(1.0 / np.sqrt(_arr(counts)))


def counting_sigma(value: ArrayLike, counts: ArrayLike) -> _Float:
    """Absolute counting uncertainty ``value/sqrt(N)`` (Poisson)."""
    return np.asarray(_arr(value) / np.sqrt(_arr(counts)))


def decay_constant(half_life: ArrayLike) -> _Float:
    """Radioactive decay constant ``lambda = ln(2)/T_half``."""
    return np.asarray(np.log(2.0) / _arr(half_life))


def radioactive_decay(n0: ArrayLike, t: ArrayLike, half_life: ArrayLike) -> _Float:
    """Radioactive decay ``N = N0*exp(-lambda*t)`` with ``lambda = ln(2)/T_half``."""
    lam = decay_constant(half_life)
    return np.asarray(_arr(n0) * np.exp(-lam * _arr(t)))


def activity(n: ArrayLike, half_life: ArrayLike) -> _Float:
    """Radioactive activity ``A = lambda*N`` with ``lambda = ln(2)/T_half``."""
    return np.asarray(decay_constant(half_life) * _arr(n))
