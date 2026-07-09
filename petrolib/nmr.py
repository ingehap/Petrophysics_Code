"""Nuclear magnetic resonance: relaxation physics, T2 statistics, forward
models, permeability transforms, and fluid typing.

The Brownstein-Tarr surface-relaxation law, the T2 log-mean, the BVI/FFI
cutoff split, and the Timur-Coates / SDR / Timur permeability transforms are
each re-implemented across the article corpus (the June 2022 NMR special issue
alone has eleven articles).  This module is the one canonical home; facades
keep their article's unit adapters and cutoff defaults.

Hazards this module's API is designed around (LIBRARY_MERGE_PLAN.md
section 9):

- **Gyromagnetic ratio drift.**  Articles use 2.675e8, 2*pi*42.58e6 =
  2.6753e8, and 26752 rad/(ms*mT) interchangeably.  ``gamma`` defaults to the
  CODATA :data:`petrolib.constants.GAMMA_H`; call sites whose published
  numbers depend on their local value pass it explicitly.
- **Pore-shape factor.**  ``r = shape_factor*rho*T2`` uses shape_factor 3 for a
  sphere radius, 2 for a cylinder radius, 6 for a sphere diameter.  It is an
  explicit parameter, never baked in.
- **Timur-Coates form.**  ``(phi/C)**m*(FFI/BVI)**n`` (the classic form) and
  ``C*phi**m*(FFI/BVI)**n`` (the prefactor form) are both in the corpus under
  one name; ``form=`` selects, and C/m/n are explicit.
- **T2 cutoff.**  BVI/FFI split at 33 ms (sandstone) or 90-100 ms (carbonate);
  the cutoff is an explicit parameter, and :func:`t2_partition` generalizes to
  the CBW/BVI/FFI three-way split.

Units follow the article convention: T1/T2 in ms, surface relaxivity rho in
um/s, S/V in 1/um, diffusivity D in um2/ms, gradient G in T/m, echo spacing TE
in ms, permeability in mD, porosity a fraction.  numpy-only at import; any
fit/inversion routine imports scipy lazily.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import GAMMA_H

_Float = NDArray[np.float64]

_TINY = 1e-30  # zero-denominator guard


# ==========================================================================
# 1. Relaxation physics (Brownstein-Tarr, diffusion, pore size)
# ==========================================================================


def diffusion_relaxation_rate(
    D: ArrayLike, *, G: ArrayLike, TE: ArrayLike, gamma: float = GAMMA_H
) -> _Float:
    """Diffusion relaxation rate ``1/T2_D = (gamma*G*TE)**2 * D / 12``.

    Restricted/unrestricted diffusion in a field gradient (the CPMG diffusion
    term).  ``gamma`` is the proton gyromagnetic ratio in rad/(s*T); pass G, TE,
    D in a consistent (SI) unit set.  Sources: src2018_04/article3,
    src2019_12/article5 (gamma=2pi*42.58e6), src2021_04/article1.
    """
    d = np.asarray(D, np.float64)
    g = np.asarray(G, np.float64)
    te = np.asarray(TE, np.float64)
    return np.asarray((gamma * g * te) ** 2 * d / 12.0)


def relaxation_rate(
    *,
    t2_bulk: ArrayLike = np.inf,
    rho: ArrayLike = 0.0,
    s_over_v: ArrayLike = 0.0,
    D: ArrayLike = 0.0,
    G: ArrayLike = 0.0,
    TE: ArrayLike = 0.0,
    gamma: float = GAMMA_H,
) -> _Float:
    """Brownstein-Tarr relaxation rate (bulk + surface + diffusion).

    ``1/T2 = 1/T2_bulk + rho*(S/V) + (gamma*G*TE)**2*D/12``.  The default
    ``t2_bulk=inf`` and zero surface/diffusion terms let a caller build any
    subset (surface-only, bulk+surface, or the full three-mechanism rate).
    Use a consistent unit set.  Sources: src2019_12/article6 (t2_total),
    src2021_04/article5, src2020_04/article2, src2014_06/article3.
    """
    inv_bulk = 1.0 / np.asarray(t2_bulk, np.float64)
    surface = np.asarray(rho, np.float64) * np.asarray(s_over_v, np.float64)
    diffusion = diffusion_relaxation_rate(D, G=G, TE=TE, gamma=gamma)
    return np.asarray(inv_bulk + surface + diffusion)


def t2_apparent(
    *,
    t2_bulk: ArrayLike = np.inf,
    rho: ArrayLike = 0.0,
    s_over_v: ArrayLike = 0.0,
    D: ArrayLike = 0.0,
    G: ArrayLike = 0.0,
    TE: ArrayLike = 0.0,
    gamma: float = GAMMA_H,
) -> _Float:
    """Apparent relaxation time ``T2 = 1/relaxation_rate(...)`` — the inverse of
    :func:`relaxation_rate` with the same keywords.  Sources:
    src2019_12/article5 (t2_apparent), src2021_04/article5 (t2_relaxation).
    """
    return np.asarray(
        1.0
        / relaxation_rate(t2_bulk=t2_bulk, rho=rho, s_over_v=s_over_v, D=D, G=G, TE=TE, gamma=gamma)
    )


def combine_relaxation_times(*times_ms: ArrayLike) -> _Float:
    """Combine relaxation times as parallel rates ``1/T = sum(1/Ti)``.

    Variadic over the bulk/surface/diffusion contributions (or any set of
    component times).  Sources: src2016_08/article1 (total_relaxation_rate),
    src2017_10/article2 (combined_relaxation_time).
    """
    inv = sum(1.0 / np.asarray(t, np.float64) for t in times_ms)
    return np.asarray(1.0 / inv)


def surface_to_volume(t2_ms: ArrayLike, *, rho: float, t2_bulk: ArrayLike = np.inf) -> _Float:
    """Surface-to-volume ratio from a relaxation time, inverting Brownstein-Tarr.

    ``S/V = (1/T2 - 1/T2_bulk)/rho`` (``t2_bulk=inf`` gives the surface-only
    ``1/(rho*T2)``).  Sources: src2018_04/article3, src2015_10/article3,
    src2016_08/article2.
    """
    t2 = np.asarray(t2_ms, np.float64)
    return np.asarray((1.0 / t2 - 1.0 / np.asarray(t2_bulk, np.float64)) / rho)


def pore_radius_from_t2(
    t2_ms: ArrayLike,
    *,
    rho: float,
    shape_factor: float = 3.0,
    t2_bulk: ArrayLike = np.inf,
) -> _Float:
    """Pore radius from a relaxation time ``r = shape_factor/(S/V)``.

    With ``t2_bulk=inf`` this is ``r = shape_factor*rho*T2``.  ``shape_factor``
    is 3 for a sphere radius, 2 for a cylinder radius, 6 for a sphere diameter
    (explicit; the corpus mixes all three).  Sources: src2017_04/article3
    (generic alpha), src2021_08/article3 (3.0), src2014_12/article3 (bulk form).
    """
    return np.asarray(shape_factor / surface_to_volume(t2_ms, rho=rho, t2_bulk=t2_bulk))


def surface_relaxivity_from_pore(
    t2_ms: ArrayLike, radius_um: ArrayLike, *, shape_factor: float = 3.0
) -> _Float:
    """Surface relaxivity from a relaxation time and known pore size.

    ``rho = radius/(shape_factor*T2)`` — the inverse of :func:`pore_radius_from_t2`
    in the surface-only limit (``shape_factor=1`` gives the raw ``r/T2``).
    Sources: src2022_06/article5 (apparent_surface_relaxivity, shape_factor 1).
    """
    return np.asarray(
        np.asarray(radius_um, np.float64) / (shape_factor * np.asarray(t2_ms, np.float64))
    )


def larmor_frequency(b0_T: ArrayLike, *, gamma: float = GAMMA_H) -> _Float:
    """Larmor frequency ``f = gamma*B0/(2*pi)`` (Hz).  ``gamma`` in rad/(s*T).
    Sources: src2014_06/article3 (larmor_frequency).
    """
    return np.asarray(gamma * np.asarray(b0_T, np.float64) / (2.0 * np.pi))


# ==========================================================================
# 2. T2 distribution statistics & partitions
# ==========================================================================


def t2_logmean(t2_ms: ArrayLike, amplitude: ArrayLike) -> float:
    """Amplitude-weighted geometric mean ``T2LM = exp(sum(A*ln T2)/sum(A))``.

    Returns ``nan`` on zero (or negative) total amplitude.  Sources:
    src2014_06/article4, src2016_08/article2, src2026_06/a03 (nan guard).
    """
    t2 = np.asarray(t2_ms, np.float64)
    a = np.asarray(amplitude, np.float64)
    total = a.sum()
    if total <= 0:
        return float("nan")
    return float(np.exp(np.sum(a * np.log(t2)) / total))


def total_porosity(amplitude: ArrayLike) -> float:
    """Total NMR porosity as the summed T2 amplitude ``phi = sum(A)``.  Sources:
    src2021_08/article3, src2015_04/article5.
    """
    return float(np.sum(np.asarray(amplitude, np.float64)))


def t2_partition(
    t2_ms: ArrayLike,
    amplitude: ArrayLike,
    *,
    cutoffs_ms: tuple[float, ...] = (33.0,),
    fractions: bool = False,
) -> tuple[float, ...]:
    """Partition T2 amplitude into ``len(cutoffs)+1`` bands by T2 cutoffs.

    Band ``i`` sums amplitude where ``cutoffs[i-1] <= T2 < cutoffs[i]`` (the
    first band is ``T2 < cutoffs[0]``, the last ``T2 >= cutoffs[-1]``).  One
    cutoff gives BVI/FFI (33 ms sandstone, 100 ms carbonate); two give the
    CBW/capillary/free split (3/33 ms).  ``fractions=True`` normalizes by the
    total.  Sources: src2015_12/article1, src2021_04/article5 (3-way),
    src2026_02/nmr_discrete_inversion.
    """
    t2 = np.asarray(t2_ms, np.float64)
    a = np.asarray(amplitude, np.float64)
    edges = (-np.inf, *cutoffs_ms, np.inf)
    bands = [
        float(np.sum(a[(t2 >= lo) & (t2 < hi)]))
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    if fractions:
        total = sum(bands)
        if total > 0:
            bands = [b / total for b in bands]
    return tuple(bands)


def bvi_ffi(
    t2_ms: ArrayLike, amplitude: ArrayLike, *, cutoff_ms: float = 33.0
) -> tuple[float, float]:
    """Bound (BVI) and free (FFI) fluid split at a single T2 cutoff.

    ``BVI = sum(A[T2 < cutoff])``, ``FFI = sum(A[T2 >= cutoff])`` — the two-band
    case of :func:`t2_partition`.  Sources: src2021_08/article3 (33 ms),
    src2014_06/article4 (100 ms).
    """
    bvi, ffi = t2_partition(t2_ms, amplitude, cutoffs_ms=(cutoff_ms,))
    return bvi, ffi


# ==========================================================================
# 3. Forward models (CPMG, T1 recovery, kernels)
# ==========================================================================


def cpmg_kernel(t_ms: ArrayLike, t2_grid_ms: ArrayLike) -> _Float:
    """CPMG forward kernel ``K[i,j] = exp(-t_i/T2_j)``.  Sources:
    src2022_06/article3, src2026_02/nmr_discrete_inversion, src2015_04/article5.
    """
    t = np.asarray(t_ms, np.float64)
    t2 = np.asarray(t2_grid_ms, np.float64)
    return np.asarray(np.exp(-t[:, None] / t2[None, :]))


def multiexp_decay(
    t_ms: ArrayLike,
    amplitudes: ArrayLike,
    t2_ms: ArrayLike,
    *,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> _Float:
    """Multi-exponential CPMG decay ``M(t) = sum_i A_i*exp(-t/T2_i)``.

    Scalars give a mono-exponential decay.  ``noise>0`` adds zero-mean Gaussian
    noise and requires an explicit ``rng`` for reproducibility.  Sources:
    src2018_04/article3, src2015_08/article1 (multiexponential_decay).
    """
    t = np.atleast_1d(np.asarray(t_ms, np.float64))
    a = np.atleast_1d(np.asarray(amplitudes, np.float64))
    t2 = np.atleast_1d(np.asarray(t2_ms, np.float64))
    signal = (a * np.exp(-t[:, None] / t2[None, :])).sum(axis=1)
    if noise:
        if rng is None:
            raise ValueError("multiexp_decay needs an rng when noise > 0")
        signal = signal + rng.normal(0.0, noise, size=signal.shape)
    return np.asarray(signal)


def t1_saturation_recovery(t_ms: ArrayLike, m0: float, t1_ms: float) -> _Float:
    """Saturation-recovery magnetization ``M(t) = M0*(1 - exp(-t/T1))``.
    Sources: src2019_06/article5 (saturation_recovery), src2020_04/article2.
    """
    t = np.asarray(t_ms, np.float64)
    return np.asarray(m0 * (1.0 - np.exp(-t / t1_ms)))


def t1_inversion_recovery(t_ms: ArrayLike, m0: float, t1_ms: float) -> _Float:
    """Inversion-recovery magnetization ``M(t) = M0*(1 - 2*exp(-t/T1))``.
    Sources: src2019_06/article5 (inversion_recovery), src2024_06/article6.
    """
    t = np.asarray(t_ms, np.float64)
    return np.asarray(m0 * (1.0 - 2.0 * np.exp(-t / t1_ms)))


def t1t2_kernel(
    t_echo_ms: ArrayLike,
    t_wait_ms: ArrayLike,
    t1_grid_ms: ArrayLike,
    t2_grid_ms: ArrayLike,
    *,
    mode: str = "saturation",
) -> _Float:
    """2D T1-T2 Kronecker kernel ``kron(K_T1, K_T2)``.

    ``K_T2 = exp(-t_echo/T2)``; ``K_T1 = 1 - exp(-tw/T1)`` for ``mode="saturation"``
    or ``1 - 2*exp(-tw/T1)`` for ``mode="inversion"`` (the corpus disagrees on
    the T1-axis recovery form, so it is explicit).  Sources:
    src2022_06/article3 (saturation), src2024_06/article6 (inversion).
    """
    k_t2 = cpmg_kernel(t_echo_ms, t2_grid_ms)
    tw = np.asarray(t_wait_ms, np.float64)
    t1 = np.asarray(t1_grid_ms, np.float64)
    exp_t1 = np.exp(-tw[:, None] / t1[None, :])
    if mode == "saturation":
        k_t1 = 1.0 - exp_t1
    elif mode == "inversion":
        k_t1 = 1.0 - 2.0 * exp_t1
    else:
        raise ValueError(f"mode must be 'saturation' or 'inversion', got {mode!r}")
    return np.asarray(np.kron(k_t1, k_t2))


def fit_t1(t_ms: ArrayLike, signal: ArrayLike, *, model: str = "saturation") -> tuple[float, float]:
    """Fit ``(M0, T1)`` from a T1 recovery curve (scipy ``curve_fit``, lazy).

    ``model="saturation"`` fits ``M0*(1-exp(-t/T1))``; ``"inversion"`` fits
    ``M0*(1-2*exp(-t/T1))``.  Sources: src2019_06/article5 (fit_t1_saturation).
    """
    from scipy.optimize import curve_fit

    t = np.asarray(t_ms, np.float64)
    s = np.asarray(signal, np.float64)

    def _sat(tt: _Float, m0: float, t1: float) -> _Float:
        return np.asarray(m0 * (1.0 - np.exp(-tt / t1)))

    def _inv(tt: _Float, m0: float, t1: float) -> _Float:
        return np.asarray(m0 * (1.0 - 2.0 * np.exp(-tt / t1)))

    if model == "saturation":
        func = _sat
    elif model == "inversion":
        func = _inv
    else:
        raise ValueError(f"model must be 'saturation' or 'inversion', got {model!r}")
    p0 = [float(np.max(np.abs(s))), 1.0]
    popt, _ = curve_fit(func, t, s, p0=p0, maxfev=10000)
    return float(popt[0]), float(popt[1])


# ==========================================================================
# 4. Permeability transforms (Timur-Coates, SDR, Timur)
# ==========================================================================


def timur_coates(
    phi: ArrayLike,
    ffi: ArrayLike,
    bvi: ArrayLike,
    *,
    C: float = 10.0,
    m: float = 4.0,
    n: float = 2.0,
    form: str = "classic",
) -> _Float:
    """Timur-Coates NMR permeability (mD).

    ``form="classic"``: ``k = (phi/C)**m*(FFI/BVI)**n`` (the quartic-ratio form,
    m=4/n=2).  ``form="prefactor"``: ``k = C*phi**m*(FFI/BVI)**n``.  These are
    algebraically distinct families sharing the name; copies that report in
    different units apply their scale in the facade.  Sources:
    src2014_06/article4 (classic), src2022_06/article3 (prefactor).
    """
    phi_a = np.asarray(phi, np.float64)
    ratio = np.asarray(ffi, np.float64) / np.asarray(bvi, np.float64)
    if form == "classic":
        return np.asarray((phi_a / C) ** m * ratio**n)
    if form == "prefactor":
        return np.asarray(C * phi_a**m * ratio**n)
    raise ValueError(f"form must be 'classic' or 'prefactor', got {form!r}")


def sdr(
    phi: ArrayLike,
    t2lm_ms: ArrayLike,
    *,
    a: float = 4.0,
    m: float = 4.0,
    n: float = 2.0,
    rho_um_s: float | None = None,
) -> _Float:
    """Schlumberger-Doll-Research NMR permeability (mD).

    ``k = a*phi**m*T2LM**n``; if ``rho_um_s`` is given, the KSDR form
    ``a*phi**m*(rho*T2LM)**n``.  The prefactor ``a`` is calibration-specific
    (4.0, 4.6, ...) and explicit.  Sources: src2021_04/article1 (a=4.0),
    src2022_06/article3 (a=4.6), src2015_10/article3 (KSDR).
    """
    phi_a = np.asarray(phi, np.float64)
    t2lm = np.asarray(t2lm_ms, np.float64)
    if rho_um_s is not None:
        t2lm = rho_um_s * t2lm
    return np.asarray(a * phi_a**m * t2lm**n)


def timur(
    phi: ArrayLike, swirr: ArrayLike, *, a: float = 4800.0, b: float = 4.4, c: float = 2.0
) -> _Float:
    """Timur (1968) NMR permeability ``k = a*phi**b/Swirr**c`` (mD).

    The prefactor ``a`` is a regional calibration (4800, 8581, 1360, ...) and
    explicit.  Sources: src2025_06/cross_calibrated_permeability (a=4800),
    src2026_04/a07 (a=8581), src2025_10/a6.
    """
    phi_a = np.asarray(phi, np.float64)
    sw = np.asarray(swirr, np.float64)
    return np.asarray(a * phi_a**b / sw**c)


# ==========================================================================
# 5. Fluid typing & hydrogen index
# ==========================================================================


def t1_t2_ratio(t1: ArrayLike, t2: ArrayLike) -> _Float:
    """T1/T2 ratio for fluid typing.  Sources: src2019_02/article3,
    src2017_04/article4, src2017_08/article1 (all plain t1/t2).
    """
    return np.asarray(np.asarray(t1, np.float64) / np.asarray(t2, np.float64))


def classify_t1t2(t1t2: ArrayLike, *, cutoff: float = 2.0) -> NDArray[np.bool_]:
    """Boolean hydrocarbon flag ``T1/T2 >= cutoff`` (True = hydrocarbon/bound).

    The cutoff varies across the corpus (2.0 sandstone/shale, 4.0 kerogen);
    it is explicit.  Sources: src2016_02/article2 (2.0), src2016_12/article4
    (2.0), src2022_06/article1 (4.0).
    """
    return np.asarray(t1t2, np.float64) >= cutoff


def partition_t1t2_map(
    t1t2: ArrayLike, amplitudes: ArrayLike, *, cutoff: float = 2.0
) -> tuple[float, float]:
    """Split a T1-T2 amplitude map into ``(v_hc, v_water)`` at a T1/T2 cutoff.

    ``v_hc`` sums the amplitude where ``T1/T2 >= cutoff``; ``v_water`` is the
    rest.  Sources: src2016_02/article2 (partition_2d_map).
    """
    amp = np.asarray(amplitudes, np.float64)
    hc_mask = classify_t1t2(t1t2, cutoff=cutoff)
    v_hc = float(np.sum(amp[hc_mask]))
    return v_hc, float(np.sum(amp) - v_hc)


def nmr_saturation(v_fluid: ArrayLike, v_total: ArrayLike) -> _Float:
    """NMR fluid saturation ``S = V_fluid/V_total``.  Sources:
    src2016_02/article2 (nmr_oil_saturation), src2025_02/mr_bulk_saturation.
    """
    return np.asarray(np.asarray(v_fluid, np.float64) / np.asarray(v_total, np.float64))


def hydrogen_index(
    rho: ArrayLike,
    n_protons: ArrayLike,
    mol_weight: float,
    *,
    rho_w: float = 1.0,
    n_w: float = 2.0,
    m_w: float = 18.02,
) -> _Float:
    """Hydrogen index vs water ``HI = (rho*n/M)/(rho_w*n_w/M_w)``.

    The water reference is 2 protons / 18.02 g/mol at 1 g/cc, so pure water
    has HI = 1.  Sources: src2017_06/article2 (hydrogen_index).
    """
    rho_arr = np.asarray(rho, np.float64)
    n_arr = np.asarray(n_protons, np.float64)
    return np.asarray((rho_arr * n_arr / mol_weight) / (rho_w * n_w / m_w))


def porosity_hi_correction(phi_apparent: ArrayLike, hi: ArrayLike) -> _Float:
    """HI-corrected porosity ``phi_true = phi_apparent/HI``.

    The apparent NMR porosity under-reads when the fluid HI < 1 (gas, light
    oil).  Sources: src2017_08/article1 (hi_corrected_porosity),
    src2025_04/unconventional_porosity.
    """
    return np.asarray(np.asarray(phi_apparent, np.float64) / np.asarray(hi, np.float64))


# ==========================================================================
# 6. Relaxation theory (BPP, Mitra, tortuosity)
# ==========================================================================


def bpp_spectral_density(omega: ArrayLike, tau_c: float) -> _Float:
    """Bloembergen-Purcell-Pound spectral density ``J = tau_c/(1+(omega*tau_c)**2)``.
    Sources: src2016_08/article1, src2017_08/article1 (bpp_spectral_density).
    """
    w = np.asarray(omega, np.float64)
    return np.asarray(tau_c / (1.0 + (w * tau_c) ** 2))


def bpp_t1_t2(
    omega0: float, tau_c: float, *, dipolar_constant: float = 1.0
) -> tuple[_Float, _Float]:
    """BPP relaxation times ``(T1, T2)`` from the correlation time.

    ``1/T1 = (3/10)*C*(J(w0)+4*J(2w0))``,
    ``1/T2 = (3/20)*C*(3*J(0)+5*J(w0)+2*J(2w0))`` — the full dipolar prefactors
    (``C`` the dipolar constant).  Sources: src2016_08/article1 (bpp_rates).
    """
    j0 = bpp_spectral_density(0.0, tau_c)
    j1 = bpp_spectral_density(omega0, tau_c)
    j2 = bpp_spectral_density(2.0 * omega0, tau_c)
    inv_t1 = (3.0 / 10.0) * dipolar_constant * (j1 + 4.0 * j2)
    inv_t2 = (3.0 / 20.0) * dipolar_constant * (3.0 * j0 + 5.0 * j1 + 2.0 * j2)
    return np.asarray(1.0 / inv_t1), np.asarray(1.0 / inv_t2)


def mitra_short_time(
    d0: float, t: ArrayLike, s_over_v: float, *, normalized: bool = True
) -> _Float:
    """Mitra short-time restricted diffusion ``D(t)/D0 = 1 - (4/(9*sqrt(pi)))*(S/V)*sqrt(D0*t)``.

    ``normalized=True`` returns the ratio ``D(t)/D0``; ``False`` returns the
    apparent diffusivity ``D(t)``.  Sources: src2015_02/article3
    (mitra_restricted_diffusion, absolute), src2019_12/article5 (pade_short_time,
    ratio).
    """
    tt = np.asarray(t, np.float64)
    ratio = 1.0 - (4.0 / (9.0 * np.sqrt(np.pi))) * s_over_v * np.sqrt(d0 * tt)
    return np.asarray(ratio if normalized else d0 * ratio)


def tortuosity(d0: ArrayLike, d_inf: ArrayLike) -> _Float:
    """Diffusive tortuosity ``tau = D0/D_inf`` (free / restricted-plateau
    diffusivity).  Sources: src2021_06/article2 (diffusive_tortuosity),
    src2019_12/article5 (tortuosity, arg order reversed — facades map).
    """
    return np.asarray(np.asarray(d0, np.float64) / np.asarray(d_inf, np.float64))
