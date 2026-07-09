"""Acoustic / elastic conversions, rock-physics mixing, and geomechanics.

The canonical home for the moduli<->velocity algebra, mineral/fluid mixing
bounds (Voigt/Reuss/Hill, Wood, Brie), Gassmann fluid substitution, acoustic
impedance / interface / attenuation, Thomsen anisotropy, and the poroelastic
stress / fracture-pressure relations that recur across the corpus.

Unit policy (LIBRARY_MERGE_PLAN.md section on acoustic_geomech): this module is
**strict SI** -- moduli in Pa, densities in kg/m3, velocities in m/s, stresses
and pressures in Pa, angles in degrees at the API boundary.  The corpus writes
the same physics in GPa+g/cc (1e9/1e3 baked in), GPa+kg/m3, and unit-neutral
forms; those conversions live in the article facades or ``petrolib.units``, not
here.  Impedance is returned in Rayl by default with an opt-in ``out="mrayl"``.

numpy-broadcastable, scalar-in/scalar-out.  Functions that return a pair (K,G),
(Vp,Vs), (lambda,mu), (E,nu) return them as a tuple in the documented order --
an order-collision hazard the corpus hits both ways, so the order is fixed and
documented here and facades map to it explicitly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]

# Standard gravity (m/s2), used by the SI stress/pressure integrations.
G_STANDARD = 9.80665


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


# ==========================================================================
# 1. Elastic moduli <-> velocity conversions (strict SI: Pa, kg/m3, m/s)
# ==========================================================================


def moduli_from_velocity(vp: ArrayLike, vs: ArrayLike, rho: ArrayLike) -> tuple[_Float, _Float]:
    """Bulk and shear modulus from P/S velocities and density -> ``(K, G)``.

    ``G = rho*Vs**2``; ``K = rho*Vp**2 - 4/3*G``.  Returns ``(K, G)`` in Pa
    (with SI inputs).  The corpus writes the pair both ``(K, G)`` and ``(G, K)``
    -- the order here is fixed as ``(K, G)``.
    """
    vp_a = _arr(vp)
    vs_a = _arr(vs)
    rho_a = _arr(rho)
    g = rho_a * vs_a**2
    k = rho_a * vp_a**2 - 4.0 / 3.0 * g
    return np.asarray(k), np.asarray(g)


def velocity_from_moduli(k: ArrayLike, g: ArrayLike, rho: ArrayLike) -> tuple[_Float, _Float]:
    """P/S velocities from bulk/shear modulus and density -> ``(Vp, Vs)``.

    ``Vp = sqrt((K + 4/3*G)/rho)``; ``Vs = sqrt(G/rho)`` (m/s with SI inputs).
    """
    k_a = _arr(k)
    g_a = _arr(g)
    rho_a = _arr(rho)
    vp = np.sqrt((k_a + 4.0 / 3.0 * g_a) / rho_a)
    vs = np.sqrt(g_a / rho_a)
    return np.asarray(vp), np.asarray(vs)


def stiffness_from_velocity(rho: ArrayLike, v: ArrayLike) -> _Float:
    """Stiffness coefficient ``C = rho*V**2`` (the P-wave modulus ``M`` for Vp)."""
    return np.asarray(_arr(rho) * _arr(v) ** 2)


def lame_from_velocity(vp: ArrayLike, vs: ArrayLike, rho: ArrayLike) -> tuple[_Float, _Float]:
    """Lame parameters from velocities and density -> ``(lambda, mu)``.

    ``mu = rho*Vs**2``; ``lambda = rho*Vp**2 - 2*mu`` (Pa with SI inputs).
    """
    vp_a = _arr(vp)
    vs_a = _arr(vs)
    rho_a = _arr(rho)
    mu = rho_a * vs_a**2
    lam = rho_a * vp_a**2 - 2.0 * mu
    return np.asarray(lam), np.asarray(mu)


def velocity_from_slowness(dt: ArrayLike, *, dt_unit: str = "us/ft") -> _Float:
    """Velocity (m/s) from acoustic slowness.

    ``dt_unit="us/ft"`` -> ``0.3048 * 1e6 / dt``; ``"us/m"`` -> ``1e6 / dt``.
    """
    dt_a = _arr(dt)
    if dt_unit == "us/ft":
        return np.asarray(0.3048 * 1.0e6 / dt_a)
    if dt_unit == "us/m":
        return np.asarray(1.0e6 / dt_a)
    raise ValueError(f"dt_unit must be 'us/ft' or 'us/m', got {dt_unit!r}")


def slowness_from_velocity(v: ArrayLike, *, dt_unit: str = "us/ft") -> _Float:
    """Acoustic slowness from velocity (m/s) -- inverse of ``velocity_from_slowness``.

    ``dt_unit="us/ft"`` -> ``0.3048 * 1e6 / v`` (us/ft); ``"us/m"`` -> ``1e6 / v``.
    """
    v_a = _arr(v)
    if dt_unit == "us/ft":
        return np.asarray(0.3048 * 1.0e6 / v_a)
    if dt_unit == "us/m":
        return np.asarray(1.0e6 / v_a)
    raise ValueError(f"dt_unit must be 'us/ft' or 'us/m', got {dt_unit!r}")


def youngs_poisson_dynamic(vp: ArrayLike, vs: ArrayLike, rho: ArrayLike) -> tuple[_Float, _Float]:
    """Dynamic Young's modulus and Poisson's ratio from velocities -> ``(E, nu)``.

    ``nu = (Vp**2 - 2*Vs**2)/(2*(Vp**2 - Vs**2))``;
    ``E = rho*Vs**2*(3*Vp**2 - 4*Vs**2)/(Vp**2 - Vs**2)`` (E in Pa with SI inputs).
    """
    vp2 = _arr(vp) ** 2
    vs2 = _arr(vs) ** 2
    rho_a = _arr(rho)
    nu = (vp2 - 2.0 * vs2) / (2.0 * (vp2 - vs2))
    e = rho_a * vs2 * (3.0 * vp2 - 4.0 * vs2) / (vp2 - vs2)
    return np.asarray(e), np.asarray(nu)


def youngs_from_kg(k: ArrayLike, g: ArrayLike) -> _Float:
    """Young's modulus from bulk and shear modulus ``E = 9*K*G/(3*K + G)``."""
    k_a = _arr(k)
    g_a = _arr(g)
    return np.asarray(9.0 * k_a * g_a / (3.0 * k_a + g_a))


def poisson_from_kg(k: ArrayLike, g: ArrayLike) -> _Float:
    """Poisson's ratio from bulk and shear modulus ``nu = (3K - 2G)/(2*(3K + G))``."""
    k_a = _arr(k)
    g_a = _arr(g)
    return np.asarray((3.0 * k_a - 2.0 * g_a) / (2.0 * (3.0 * k_a + g_a)))


def poisson_from_velocity(vp: ArrayLike, vs: ArrayLike) -> _Float:
    """Poisson's ratio from P/S velocities ``(Vp**2 - 2*Vs**2)/(2*(Vp**2 - Vs**2))``."""
    vp2 = _arr(vp) ** 2
    vs2 = _arr(vs) ** 2
    return np.asarray((vp2 - 2.0 * vs2) / (2.0 * (vp2 - vs2)))


def pwave_modulus(k: ArrayLike, g: ArrayLike) -> _Float:
    """P-wave (constrained) modulus ``M = K + 4/3*G``."""
    return np.asarray(_arr(k) + 4.0 / 3.0 * _arr(g))


# ==========================================================================
# 2. Mineral / fluid mixing and Gassmann fluid substitution
# ==========================================================================


def voigt(f: ArrayLike, m: ArrayLike, *, axis: int = -1) -> _Float:
    """Voigt (arithmetic) bound ``sum(f_i*m_i)`` over the mixing axis.

    ``f`` volume fractions and ``m`` component moduli along ``axis``.
    """
    return np.asarray(np.sum(_arr(f) * _arr(m), axis=axis))


def reuss(f: ArrayLike, m: ArrayLike, *, axis: int = -1) -> _Float:
    """Reuss (harmonic) bound ``1/sum(f_i/m_i)`` over the mixing axis."""
    return np.asarray(1.0 / np.sum(_arr(f) / _arr(m), axis=axis))


def voigt_reuss_hill(f: ArrayLike, m: ArrayLike, *, axis: int = -1) -> _Float:
    """Voigt-Reuss-Hill average ``0.5*(Voigt + Reuss)``."""
    return np.asarray(0.5 * (voigt(f, m, axis=axis) + reuss(f, m, axis=axis)))


def wood_fluid_modulus(saturations: ArrayLike, moduli: ArrayLike, *, axis: int = -1) -> _Float:
    """Wood's law effective fluid modulus (Reuss average over fluid phases).

    ``1/K_fl = sum(S_i/K_i)`` with saturations summing to one along ``axis``.
    """
    return reuss(saturations, moduli, axis=axis)


def brie_fluid_modulus(
    sw: ArrayLike, k_liquid: ArrayLike, k_gas: ArrayLike, *, e: float = 3.0
) -> _Float:
    """Brie et al. (1995) fluid-mixing modulus ``(K_liq - K_gas)*Sw**e + K_gas``.

    ``e`` is the Brie exponent (default 3.0); ``Sw`` is the liquid saturation.
    """
    sw_a = _arr(sw)
    return np.asarray((_arr(k_liquid) - _arr(k_gas)) * sw_a**e + _arr(k_gas))


def gassmann_ksat(
    *, k_dry: ArrayLike, k_mineral: ArrayLike, k_fluid: ArrayLike, phi: ArrayLike
) -> _Float:
    """Gassmann saturated bulk modulus from the dry-frame modulus.

    ``K_sat = K_dry + (1 - K_dry/K_min)**2 /
    (phi/K_fl + (1 - phi)/K_min - K_dry/K_min**2)``.  Keyword-only to defeat the
    ``k_dry``/``k_sat`` argument collision.
    """
    kd = _arr(k_dry)
    km = _arr(k_mineral)
    kf = _arr(k_fluid)
    p = _arr(phi)
    num = (1.0 - kd / km) ** 2
    den = p / kf + (1.0 - p) / km - kd / km**2
    return np.asarray(kd + num / den)


def gassmann_kdry(
    *, k_sat: ArrayLike, k_mineral: ArrayLike, k_fluid: ArrayLike, phi: ArrayLike
) -> _Float:
    """Gassmann dry-frame bulk modulus from the saturated modulus (inverse).

    ``K_dry = (K_sat*(phi*K_min/K_fl + 1 - phi) - K_min) /
    (phi*K_min/K_fl + K_sat/K_min - 1 - phi)``.  Keyword-only.
    """
    ks = _arr(k_sat)
    km = _arr(k_mineral)
    kf = _arr(k_fluid)
    p = _arr(phi)
    ratio = p * km / kf
    num = ks * (ratio + 1.0 - p) - km
    den = ratio + ks / km - 1.0 - p
    return np.asarray(num / den)


# ==========================================================================
# 3. Impedance, interface reflection / transmission, attenuation
# ==========================================================================


def acoustic_impedance(rho: ArrayLike, v: ArrayLike, *, out: str = "rayl") -> _Float:
    """Acoustic impedance ``Z = rho*v``.

    ``out="rayl"`` returns SI Rayl (kg/m2/s); ``"mrayl"`` returns MRayl (Z/1e6).
    """
    z = _arr(rho) * _arr(v)
    if out == "rayl":
        return np.asarray(z)
    if out == "mrayl":
        return np.asarray(z / 1.0e6)
    raise ValueError(f"out must be 'rayl' or 'mrayl', got {out!r}")


def reflection_coefficient(z1: ArrayLike, z2: ArrayLike) -> _Float:
    """Normal-incidence reflection coefficient ``(Z2 - Z1)/(Z2 + Z1)``.

    ``z1`` is the incident-side impedance, ``z2`` the transmitted side.
    """
    z1_a = _arr(z1)
    z2_a = _arr(z2)
    return np.asarray((z2_a - z1_a) / (z2_a + z1_a))


def transmission_energy(z1: ArrayLike, z2: ArrayLike) -> _Float:
    """Transmitted energy fraction ``1 - R**2`` at a normal-incidence interface."""
    r = reflection_coefficient(z1, z2)
    return np.asarray(1.0 - r**2)


def attenuation_db(a1: ArrayLike, a2: ArrayLike) -> _Float:
    """Attenuation in decibels ``20*log10(A1/A2)`` between two amplitudes."""
    return np.asarray(20.0 * np.log10(_arr(a1) / _arr(a2)))


def attenuation_coefficient(a0: ArrayLike, ax: ArrayLike, x: ArrayLike) -> _Float:
    """Attenuation coefficient ``alpha`` in ``A = A0*exp(-alpha*x)``.

    ``alpha = ln(A0/Ax)/x`` (1/length in the units of ``x``).
    """
    return np.asarray(np.log(_arr(a0) / _arr(ax)) / _arr(x))


def snell_angle(v1: ArrayLike, v2: ArrayLike, *, degrees: bool = True) -> _Float:
    """Snell critical/refraction angle ``arcsin(V1/V2)`` from two velocities.

    With ``V1`` the slower medium this is the critical angle.  Returned in
    degrees by default (``degrees=False`` for radians).
    """
    theta = np.arcsin(_arr(v1) / _arr(v2))
    return np.asarray(np.degrees(theta) if degrees else theta)


# ==========================================================================
# 4. Thomsen anisotropy parameters (from VTI stiffnesses)
# ==========================================================================


def thomsen_epsilon(c11: ArrayLike, c33: ArrayLike) -> _Float:
    """Thomsen ``epsilon = (C11 - C33)/(2*C33)`` (P-wave anisotropy)."""
    c33_a = _arr(c33)
    return np.asarray((_arr(c11) - c33_a) / (2.0 * c33_a))


def thomsen_gamma(c66: ArrayLike, c44: ArrayLike) -> _Float:
    """Thomsen ``gamma = (C66 - C44)/(2*C44)`` (S-wave anisotropy)."""
    c44_a = _arr(c44)
    return np.asarray((_arr(c66) - c44_a) / (2.0 * c44_a))


def thomsen_delta(c13: ArrayLike, c33: ArrayLike, c44: ArrayLike) -> _Float:
    """Thomsen ``delta = ((C13 + C44)**2 - (C33 - C44)**2)/(2*C33*(C33 - C44))``."""
    c33_a = _arr(c33)
    c44_a = _arr(c44)
    return np.asarray(
        ((_arr(c13) + c44_a) ** 2 - (c33_a - c44_a) ** 2) / (2.0 * c33_a * (c33_a - c44_a))
    )


def annie_c13(c11: ArrayLike, c66: ArrayLike) -> _Float:
    """ANNIE approximation ``C13 = C11 - 2*C66`` (assumes ``C13 = C12``)."""
    return np.asarray(_arr(c11) - 2.0 * _arr(c66))


def annie_c11(c33: ArrayLike, c44: ArrayLike, c66: ArrayLike) -> _Float:
    """ANNIE approximation ``C11 = 2*(C66 - C44) + C33``."""
    return np.asarray(2.0 * (_arr(c66) - _arr(c44)) + _arr(c33))


def mannie_c13(c11: ArrayLike, c66: ArrayLike, *, k: float = 1.0) -> _Float:
    """Modified-ANNIE ``C13 = k*(C11 - 2*C66)`` (``k=1`` recovers ANNIE)."""
    return np.asarray(k * (_arr(c11) - 2.0 * _arr(c66)))


def mannie_c11(c33: ArrayLike, c44: ArrayLike, c66: ArrayLike, *, kp: float = 1.0) -> _Float:
    """Modified-ANNIE ``C11 = kp*2*(C66 - C44) + C33`` (``kp=1`` recovers ANNIE)."""
    return np.asarray(kp * (2.0 * (_arr(c66) - _arr(c44))) + _arr(c33))


def mannie2_c66(c11: ArrayLike, c33: ArrayLike, c44: ArrayLike, *, k: float = 0.93) -> _Float:
    """Modified-ANNIE-2 shear closure ``C66 = C44*(1 + k*(C11 - C33)/C33)``.

    ``k`` is the gamma/epsilon anisotropy ratio (corpus default 0.93).
    """
    c33_a = _arr(c33)
    return np.asarray(_arr(c44) * (1.0 + k * (_arr(c11) - c33_a) / c33_a))


def shear_wave_splitting(v_fast: ArrayLike, v_slow: ArrayLike) -> _Float:
    """Shear-wave splitting (anisotropy) ``(V_fast - V_slow)/V_fast``."""
    vf = _arr(v_fast)
    return np.asarray((vf - _arr(v_slow)) / vf)


def vti_engineering_moduli(
    c11: ArrayLike,
    c33: ArrayLike,
    c44: ArrayLike,
    c66: ArrayLike,
    c13: ArrayLike,
    *,
    c12: ArrayLike | None = None,
) -> dict[str, _Float]:
    """VTI engineering moduli from the stiffness constants.

    Returns ``{"Ev","Eh","nu_v","nu_h","Gvh","Ghh"}`` -- vertical/horizontal
    Young's moduli, Poisson's ratios, and shear moduli.  ``c12`` defaults to
    ``c11 - 2*c66``.  Standard VTI relations::

        Ev   = c33 - 2*c13**2/(c11 + c12)
        Eh   = (c11 - c12)*(c11*c33 - 2*c13**2 + c12*c33)/(c11*c33 - c13**2)
        nu_v = c13/(c11 + c12)
        nu_h = (c12*c33 - c13**2)/(c11*c33 - c13**2)
        Gvh  = c44,  Ghh = c66
    """
    c11a = _arr(c11)
    c33a = _arr(c33)
    c44a = _arr(c44)
    c66a = _arr(c66)
    c13a = _arr(c13)
    c12a = c11a - 2.0 * c66a if c12 is None else _arr(c12)
    ev = c33a - 2.0 * c13a**2 / (c11a + c12a)
    eh = (c11a - c12a) * (c11a * c33a - 2.0 * c13a**2 + c12a * c33a) / (c11a * c33a - c13a**2)
    nu_v = c13a / (c11a + c12a)
    nu_h = (c12a * c33a - c13a**2) / (c11a * c33a - c13a**2)
    return {
        "Ev": np.asarray(ev),
        "Eh": np.asarray(eh),
        "nu_v": np.asarray(nu_v),
        "nu_h": np.asarray(nu_h),
        "Gvh": np.asarray(c44a),
        "Ghh": np.asarray(c66a),
    }


# ==========================================================================
# 5. Geomechanics -- poroelastic stress and fracture pressures (SI: Pa)
# ==========================================================================


def biot_coefficient(k_dry: ArrayLike, k_mineral: ArrayLike) -> _Float:
    """Biot-Willis effective-stress coefficient ``alpha = 1 - K_dry/K_mineral``."""
    return np.asarray(1.0 - _arr(k_dry) / _arr(k_mineral))


def effective_stress(sigma_total: ArrayLike, pp: ArrayLike, *, biot: ArrayLike = 1.0) -> _Float:
    """Terzaghi/Biot effective stress ``sigma_eff = sigma_total - biot*Pp``."""
    return np.asarray(_arr(sigma_total) - _arr(biot) * _arr(pp))


def eaton_pore_pressure(
    overburden: ArrayLike,
    hydrostatic: ArrayLike,
    observed: ArrayLike,
    normal: ArrayLike,
    *,
    exponent: float = 3.0,
    log_type: str = "sonic",
) -> _Float:
    """Eaton pore pressure ``Pp = S_ov - (S_ov - P_hydro)*ratio**exponent``.

    ``log_type="sonic"`` -> ``ratio = normal/observed`` (slowness or velocity);
    ``"resistivity"`` -> ``ratio = observed/normal``.  The ratio-direction flip
    between the two log types is a documented corpus hazard; the default
    exponent 3.0 is the sonic value (resistivity typically uses ~1.2).
    """
    ov = _arr(overburden)
    hy = _arr(hydrostatic)
    obs = _arr(observed)
    norm = _arr(normal)
    if log_type == "sonic":
        ratio = norm / obs
    elif log_type == "resistivity":
        ratio = obs / norm
    else:
        raise ValueError(f"log_type must be 'sonic' or 'resistivity', got {log_type!r}")
    return np.asarray(ov - (ov - hy) * ratio**exponent)


def min_horizontal_stress(
    sv: ArrayLike,
    pp: ArrayLike,
    nu: ArrayLike,
    *,
    biot: ArrayLike = 1.0,
    e: ArrayLike = 0.0,
    eps_h: ArrayLike = 0.0,
    eps_H: ArrayLike = 0.0,
    tectonic: ArrayLike = 0.0,
) -> _Float:
    """Minimum horizontal stress -- poroelastic uniaxial-strain superset.

    ``sh = nu/(1-nu)*(Sv - biot*Pp) + biot*Pp
    + E/(1-nu**2)*(eps_h + nu*eps_H) + tectonic``.  With ``e=eps=0`` this reduces
    to the uniaxial-strain-plus-constant-``tectonic`` form; with ``tectonic=0``
    and the strain terms it is the Thiercelin-Plumb poroelastic form (the corpus
    writes both under this name and as ``closure_stress``).  ``e`` is Young's
    modulus in the same pressure unit as the stresses.
    """
    nu_a = _arr(nu)
    pp_a = _arr(pp)
    sv_a = _arr(sv)
    biot_a = _arr(biot)
    poro = nu_a / (1.0 - nu_a) * (sv_a - biot_a * pp_a) + biot_a * pp_a
    strain = _arr(e) / (1.0 - nu_a**2) * (_arr(eps_h) + nu_a * _arr(eps_H))
    return np.asarray(poro + strain + _arr(tectonic))


def breakdown_pressure(
    sh: ArrayLike, sH: ArrayLike, pp: ArrayLike, *, tensile_strength: ArrayLike = 0.0
) -> _Float:
    """Hubbert-Willis breakdown pressure ``3*sh - sH - Pp + T0``.

    ``sh`` minimum and ``sH`` maximum horizontal stress; ``T0`` tensile strength.
    """
    return np.asarray(3.0 * _arr(sh) - _arr(sH) - _arr(pp) + _arr(tensile_strength))


def reopening_pressure(sh: ArrayLike, sH: ArrayLike, pp: ArrayLike) -> _Float:
    """Fracture reopening pressure ``3*sh - sH - Pp`` (breakdown with T0=0)."""
    return np.asarray(3.0 * _arr(sh) - _arr(sH) - _arr(pp))


def shmax_from_reopening(pr: ArrayLike, sh: ArrayLike, pp: ArrayLike) -> _Float:
    """Maximum horizontal stress from the reopening pressure ``sH = 3*sh - Pr - Pp``."""
    return np.asarray(3.0 * _arr(sh) - _arr(pr) - _arr(pp))


def kirsch_hoop_stress(sH: ArrayLike, sh: ArrayLike, pw: ArrayLike, theta_deg: ArrayLike) -> _Float:
    """Kirsch tangential (hoop) stress at the borehole wall.

    ``sigma_theta = (sH + sh) - 2*(sH - sh)*cos(2*theta) - Pw`` with ``theta``
    measured from the ``sH`` azimuth (degrees).
    """
    th = np.radians(_arr(theta_deg))
    sH_a = _arr(sH)
    sh_a = _arr(sh)
    return np.asarray((sH_a + sh_a) - 2.0 * (sH_a - sh_a) * np.cos(2.0 * th) - _arr(pw))


def brittleness_rickman(
    e: ArrayLike,
    nu: ArrayLike,
    *,
    e_range: tuple[float, float] = (10.0, 80.0),
    nu_range: tuple[float, float] = (0.15, 0.40),
    percent: bool = False,
) -> _Float:
    """Rickman (2008) brittleness index from Young's modulus and Poisson's ratio.

    Averages the min-max normalised ``E`` (high E is brittle) and ``nu`` (low nu
    is brittle): ``0.5*((E-Emin)/(Emax-Emin) + (nu-nu_max)/(nu_min-nu_max))``.
    ``e_range``/``nu_range`` are the normalisation endpoints (defaults are
    Rickman's GPa / dimensionless brackets); ``percent`` scales to 0-100.
    ``E`` must be given in the same units as ``e_range`` (GPa by default).
    """
    e_a = _arr(e)
    nu_a = _arr(nu)
    be = (e_a - e_range[0]) / (e_range[1] - e_range[0])
    bn = (nu_a - nu_range[1]) / (nu_range[0] - nu_range[1])
    b = 0.5 * (be + bn)
    return np.asarray(b * 100.0 if percent else b)


# ==========================================================================
# 6. Empirical Vs-from-Vp and pressure / stress-depth profiles
# ==========================================================================


def vs_from_vp(vp: ArrayLike, *, method: str = "castagna_ls") -> _Float:
    """Empirical shear velocity from compressional velocity (Vp in km/s).

    ``method``: ``"castagna_ls"`` limestone
    ``-0.05050*Vp**2 + 1.10168*Vp - 1.0305``; ``"pickett_ls"`` ``Vp/1.9``;
    ``"pickett_dol"`` ``Vp/1.8``; ``"carroll"`` ``0.756090*Vp**0.81846``;
    ``"brocher"`` the Brocher (2005) 4th-order polynomial.  Vs in km/s.
    Sources: src2021_06/article3 (all five forms).
    """
    v = _arr(vp)
    if method == "castagna_ls":
        return np.asarray(-0.05050 * v**2 + 1.10168 * v - 1.0305)
    if method == "pickett_ls":
        return np.asarray(v / 1.9)
    if method == "pickett_dol":
        return np.asarray(v / 1.8)
    if method == "carroll":
        return np.asarray(0.756090 * v**0.81846)
    if method == "brocher":
        return np.asarray(0.7858 - 1.2344 * v + 0.7949 * v**2 - 0.1238 * v**3 + 0.0064 * v**4)
    raise ValueError(
        f"method must be castagna_ls/pickett_ls/pickett_dol/carroll/brocher, got {method!r}"
    )


def overburden_stress(depth: ArrayLike, rho_bulk: ArrayLike, *, g: float = G_STANDARD) -> _Float:
    """Overburden (vertical) stress ``sigma_v = rho_bulk*g*depth`` (SI, Pa).

    ``rho_bulk`` is the (average or per-sample) overburden bulk density and ``g``
    defaults to standard gravity; the cumulative depth-column integration with a
    seawater layer stays in the caller (a field-unit corpus form).
    """
    return np.asarray(_arr(rho_bulk) * g * _arr(depth))


def hydrostatic_pressure(
    depth: ArrayLike, *, rho: ArrayLike = 1000.0, g: float = G_STANDARD
) -> _Float:
    """Hydrostatic pressure ``P = rho*g*depth`` (SI, Pa).

    ``rho`` is the pore-fluid density (default fresh water 1000 kg/m3).
    """
    return np.asarray(_arr(rho) * g * _arr(depth))


def bowers_pore_pressure(
    velocity: ArrayLike,
    overburden: ArrayLike,
    *,
    a: float = 10.0,
    b: float = 0.7,
    v0: float = 5000.0,
    unloading: bool = False,
    u: float = 3.0,
    sigma_max: float = 5000.0,
) -> _Float:
    """Bowers (1995) pore pressure from velocity and overburden (field units).

    Loading effective stress ``sigma = ((V - V0)/A)**(1/B)``; the unloading
    branch adds the Bowers hysteresis ``sigma = sigma_max*(sigma_load/sigma_max)
    **(1/U)``.  Pore pressure ``Pp = overburden - sigma``.  ``V``/``V0`` in ft/s,
    stresses in psi (the corpus convention); ``V0`` is the mudline velocity.
    """
    v = _arr(velocity)
    ov = _arr(overburden)
    sigma_load = ((v - v0) / a) ** (1.0 / b)
    if unloading:
        sigma_eff = sigma_max * (sigma_load / sigma_max) ** (1.0 / u)
    else:
        sigma_eff = sigma_load
    return np.asarray(ov - sigma_eff)
