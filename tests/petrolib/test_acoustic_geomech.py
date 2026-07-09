"""Tests for petrolib.acoustic_geomech: golden values, round trips, method
dispatch, and error paths for the elastic / rock-physics / geomechanics forms."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import acoustic_geomech as ag  # noqa: E402

# --- elastic moduli <-> velocity ---------------------------------------------


def test_moduli_from_velocity_golden() -> None:
    # SI sandstone: Vp=4000, Vs=2400, rho=2500
    k, g = ag.moduli_from_velocity(4000.0, 2400.0, 2500.0)
    np.testing.assert_allclose(g, 2500.0 * 2400.0**2, rtol=1e-12)
    np.testing.assert_allclose(k, 2500.0 * 4000.0**2 - 4.0 / 3.0 * g, rtol=1e-12)


def test_moduli_velocity_round_trip() -> None:
    vp, vs, rho = 3800.0, 2200.0, 2450.0
    k, g = ag.moduli_from_velocity(vp, vs, rho)
    vp2, vs2 = ag.velocity_from_moduli(k, g, rho)
    np.testing.assert_allclose([vp2, vs2], [vp, vs], rtol=1e-12)


def test_stiffness_and_lame() -> None:
    np.testing.assert_allclose(ag.stiffness_from_velocity(2500.0, 4000.0), 2500.0 * 4000.0**2)
    lam, mu = ag.lame_from_velocity(4000.0, 2400.0, 2500.0)
    np.testing.assert_allclose(mu, 2500.0 * 2400.0**2, rtol=1e-12)
    np.testing.assert_allclose(lam, 2500.0 * 4000.0**2 - 2.0 * mu, rtol=1e-12)
    # pwave modulus = K + 4/3 G = lambda + 2 mu = rho*Vp^2
    np.testing.assert_allclose(
        ag.pwave_modulus(*ag.moduli_from_velocity(4000.0, 2400.0, 2500.0)),
        2500.0 * 4000.0**2,
        rtol=1e-12,
    )


def test_slowness_round_trip_and_units() -> None:
    # 100 us/ft -> velocity -> slowness
    v = ag.velocity_from_slowness(100.0, dt_unit="us/ft")
    np.testing.assert_allclose(v, 0.3048 * 1e6 / 100.0, rtol=1e-12)
    np.testing.assert_allclose(ag.slowness_from_velocity(v, dt_unit="us/ft"), 100.0, rtol=1e-12)
    np.testing.assert_allclose(ag.velocity_from_slowness(200.0, dt_unit="us/m"), 5000.0, rtol=1e-12)


def test_slowness_bad_unit_raises() -> None:
    with pytest.raises(ValueError, match="dt_unit"):
        ag.velocity_from_slowness(100.0, dt_unit="s/m")
    with pytest.raises(ValueError, match="dt_unit"):
        ag.slowness_from_velocity(4000.0, dt_unit="furlongs")


def test_youngs_poisson_consistency() -> None:
    vp, vs, rho = 4000.0, 2400.0, 2500.0
    e, nu = ag.youngs_poisson_dynamic(vp, vs, rho)
    k, g = ag.moduli_from_velocity(vp, vs, rho)
    np.testing.assert_allclose(e, ag.youngs_from_kg(k, g), rtol=1e-10)
    np.testing.assert_allclose(nu, ag.poisson_from_kg(k, g), rtol=1e-10)
    np.testing.assert_allclose(nu, ag.poisson_from_velocity(vp, vs), rtol=1e-12)


# --- mixing / fluid substitution ---------------------------------------------


def test_voigt_reuss_hill_bounds() -> None:
    f = np.array([0.6, 0.4])
    m = np.array([37.0, 21.0])
    v = ag.voigt(f, m)
    r = ag.reuss(f, m)
    h = ag.voigt_reuss_hill(f, m)
    np.testing.assert_allclose(v, 0.6 * 37.0 + 0.4 * 21.0, rtol=1e-12)
    np.testing.assert_allclose(r, 1.0 / (0.6 / 37.0 + 0.4 / 21.0), rtol=1e-12)
    np.testing.assert_allclose(h, 0.5 * (v + r), rtol=1e-12)
    assert float(v) >= float(h) >= float(r)


def test_wood_and_brie() -> None:
    s = np.array([0.7, 0.3])
    kfl = np.array([2.2e9, 1.0e5])
    np.testing.assert_allclose(ag.wood_fluid_modulus(s, kfl), 1.0 / (0.7 / 2.2e9 + 0.3 / 1.0e5))
    # Brie at Sw=1 recovers the liquid modulus; Sw=0 the gas modulus
    np.testing.assert_allclose(ag.brie_fluid_modulus(1.0, 2.2e9, 1.0e5), 2.2e9, rtol=1e-12)
    np.testing.assert_allclose(ag.brie_fluid_modulus(0.0, 2.2e9, 1.0e5), 1.0e5, rtol=1e-12)
    np.testing.assert_allclose(
        ag.brie_fluid_modulus(0.5, 2.2e9, 1.0e5, e=8.0),
        (2.2e9 - 1.0e5) * 0.5**8 + 1.0e5,
        rtol=1e-12,
    )


def test_gassmann_round_trip() -> None:
    kd = ag.gassmann_kdry(k_sat=30e9, k_mineral=37e9, k_fluid=2.2e9, phi=0.2)
    ks = ag.gassmann_ksat(k_dry=kd, k_mineral=37e9, k_fluid=2.2e9, phi=0.2)
    np.testing.assert_allclose(ks, 30e9, rtol=1e-10)


def test_biot_coefficient() -> None:
    np.testing.assert_allclose(ag.biot_coefficient(20e9, 37e9), 1.0 - 20e9 / 37e9, rtol=1e-12)


# --- impedance / interface / attenuation -------------------------------------


def test_acoustic_impedance_units_and_error() -> None:
    np.testing.assert_allclose(ag.acoustic_impedance(2500.0, 4000.0), 1.0e7, rtol=1e-12)
    np.testing.assert_allclose(ag.acoustic_impedance(2500.0, 4000.0, out="mrayl"), 10.0, rtol=1e-12)
    with pytest.raises(ValueError, match="out"):
        ag.acoustic_impedance(2500.0, 4000.0, out="rayls")


def test_reflection_transmission_energy_conserved() -> None:
    z1, z2 = 1.5e6, 1.0e7
    r = ag.reflection_coefficient(z1, z2)
    np.testing.assert_allclose(r, (z2 - z1) / (z2 + z1), rtol=1e-12)
    np.testing.assert_allclose(r**2 + ag.transmission_energy(z1, z2), 1.0, rtol=1e-12)
    # zero contrast -> no reflection
    np.testing.assert_allclose(ag.reflection_coefficient(z1, z1), 0.0, atol=1e-15)


def test_attenuation_and_snell() -> None:
    np.testing.assert_allclose(ag.attenuation_db(1.0, 0.1), 20.0, rtol=1e-12)
    np.testing.assert_allclose(ag.attenuation_coefficient(1.0, np.exp(-2.0), 4.0), 0.5, rtol=1e-12)
    np.testing.assert_allclose(ag.snell_angle(1500.0, 3000.0), 30.0, rtol=1e-12)
    np.testing.assert_allclose(
        ag.snell_angle(1500.0, 3000.0, degrees=False), np.arcsin(0.5), rtol=1e-12
    )


# --- anisotropy --------------------------------------------------------------


def test_thomsen_parameters() -> None:
    np.testing.assert_allclose(ag.thomsen_epsilon(34.0, 30.0), (34.0 - 30.0) / (2.0 * 30.0))
    np.testing.assert_allclose(ag.thomsen_gamma(9.0, 5.0), (9.0 - 5.0) / (2.0 * 5.0))
    np.testing.assert_allclose(
        ag.thomsen_delta(8.0, 30.0, 5.0),
        ((8.0 + 5.0) ** 2 - (30.0 - 5.0) ** 2) / (2.0 * 30.0 * (30.0 - 5.0)),
    )
    # isotropic stiffness -> zero anisotropy
    np.testing.assert_allclose(ag.thomsen_epsilon(30.0, 30.0), 0.0, atol=1e-15)


def test_annie_family() -> None:
    np.testing.assert_allclose(ag.annie_c13(34.0, 9.0), 34.0 - 2.0 * 9.0)
    np.testing.assert_allclose(ag.annie_c11(30.0, 5.0, 9.0), 2.0 * (9.0 - 5.0) + 30.0)
    # k=1 / kp=1 recover ANNIE
    np.testing.assert_allclose(ag.mannie_c13(34.0, 9.0, k=1.0), ag.annie_c13(34.0, 9.0))
    np.testing.assert_allclose(ag.mannie_c11(30.0, 5.0, 9.0, kp=1.0), ag.annie_c11(30.0, 5.0, 9.0))
    np.testing.assert_allclose(
        ag.mannie2_c66(34.0, 30.0, 5.0, k=0.93), 5.0 * (1.0 + 0.93 * (34.0 - 30.0) / 30.0)
    )


def test_shear_wave_splitting_and_vti() -> None:
    np.testing.assert_allclose(ag.shear_wave_splitting(2600.0, 2500.0), 100.0 / 2600.0, rtol=1e-12)
    vti = ag.vti_engineering_moduli(34.0, 22.0, 5.0, 9.0, 8.0)
    assert set(vti) == {"Ev", "Eh", "nu_v", "nu_h", "Gvh", "Ghh"}
    c12 = 34.0 - 2.0 * 9.0
    np.testing.assert_allclose(vti["Ev"], 22.0 - 2.0 * 8.0**2 / (34.0 + c12), rtol=1e-12)
    np.testing.assert_allclose(vti["nu_v"], 8.0 / (34.0 + c12), rtol=1e-12)
    np.testing.assert_allclose(vti["Gvh"], 5.0)
    np.testing.assert_allclose(vti["Ghh"], 9.0)
    # explicit c12 override is honoured
    vti2 = ag.vti_engineering_moduli(34.0, 22.0, 5.0, 9.0, 8.0, c12=15.0)
    np.testing.assert_allclose(vti2["nu_v"], 8.0 / (34.0 + 15.0), rtol=1e-12)


# --- geomechanics ------------------------------------------------------------


def test_effective_stress_and_biot() -> None:
    np.testing.assert_allclose(ag.effective_stress(50e6, 20e6), 30e6, rtol=1e-12)
    np.testing.assert_allclose(ag.effective_stress(50e6, 20e6, biot=0.8), 50e6 - 0.8 * 20e6)


def test_eaton_direction_and_error() -> None:
    ov, hy, obs, norm = 100.0, 46.0, 80.0, 100.0
    son = ag.eaton_pore_pressure(ov, hy, obs, norm, exponent=3.0, log_type="sonic")
    res = ag.eaton_pore_pressure(ov, hy, obs, norm, exponent=3.0, log_type="resistivity")
    np.testing.assert_allclose(son, ov - (ov - hy) * (norm / obs) ** 3.0, rtol=1e-12)
    np.testing.assert_allclose(res, ov - (ov - hy) * (obs / norm) ** 3.0, rtol=1e-12)
    assert not np.isclose(son, res)
    with pytest.raises(ValueError, match="log_type"):
        ag.eaton_pore_pressure(ov, hy, obs, norm, log_type="gamma")


def test_min_horizontal_stress_superset() -> None:
    sv, pp, nu = 100.0, 40.0, 0.25
    base = nu / (1.0 - nu) * (sv - pp) + pp
    # uniaxial + tectonic
    np.testing.assert_allclose(
        ag.min_horizontal_stress(sv, pp, nu, tectonic=5.0), base + 5.0, rtol=1e-12
    )
    # Thiercelin-Plumb strain term
    np.testing.assert_allclose(
        ag.min_horizontal_stress(sv, pp, nu, e=30.0, eps_h=1e-3, eps_H=5e-4),
        base + 30.0 / (1.0 - nu**2) * (1e-3 + nu * 5e-4),
        rtol=1e-12,
    )


def test_fracture_pressures_round_trip() -> None:
    sh, sH, pp = 40.0, 55.0, 20.0
    np.testing.assert_allclose(ag.breakdown_pressure(sh, sH, pp), 3 * sh - sH - pp)
    np.testing.assert_allclose(
        ag.breakdown_pressure(sh, sH, pp, tensile_strength=6.0), 3 * sh - sH - pp + 6.0
    )
    pr = ag.reopening_pressure(sh, sH, pp)
    np.testing.assert_allclose(ag.shmax_from_reopening(pr, sh, pp), sH, rtol=1e-12)


def test_kirsch_hoop_stress() -> None:
    sH, sh, pw = 60.0, 40.0, 30.0
    # at theta=0 (sH azimuth): sigma_theta = 3*sh - sH - pw
    np.testing.assert_allclose(ag.kirsch_hoop_stress(sH, sh, pw, 0.0), 3 * sh - sH - pw, rtol=1e-12)
    # at theta=90: sigma_theta = 3*sH - sh - pw
    np.testing.assert_allclose(
        ag.kirsch_hoop_stress(sH, sh, pw, 90.0), 3 * sH - sh - pw, rtol=1e-12
    )


def test_brittleness_rickman_endpoints() -> None:
    np.testing.assert_allclose(ag.brittleness_rickman(80.0, 0.15), 1.0, rtol=1e-12)
    np.testing.assert_allclose(ag.brittleness_rickman(10.0, 0.40), 0.0, atol=1e-15)
    np.testing.assert_allclose(ag.brittleness_rickman(80.0, 0.15, percent=True), 100.0, rtol=1e-12)


def test_vs_from_vp_methods_and_error() -> None:
    vp = 5.0
    np.testing.assert_allclose(
        ag.vs_from_vp(vp, method="castagna_ls"),
        -0.05050 * 25.0 + 1.10168 * 5.0 - 1.0305,
        rtol=1e-12,
    )
    np.testing.assert_allclose(ag.vs_from_vp(vp, method="pickett_ls"), 5.0 / 1.9, rtol=1e-12)
    np.testing.assert_allclose(ag.vs_from_vp(vp, method="pickett_dol"), 5.0 / 1.8, rtol=1e-12)
    np.testing.assert_allclose(ag.vs_from_vp(vp, method="carroll"), 0.756090 * 5.0**0.81846)
    np.testing.assert_allclose(
        ag.vs_from_vp(vp, method="brocher"),
        0.7858 - 1.2344 * 5 + 0.7949 * 25 - 0.1238 * 125 + 0.0064 * 625,
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="method"):
        ag.vs_from_vp(vp, method="mudrock")


def test_pressure_profiles_and_bowers() -> None:
    np.testing.assert_allclose(ag.overburden_stress(3000.0, 2400.0, g=9.81), 2400.0 * 9.81 * 3000.0)
    np.testing.assert_allclose(
        ag.hydrostatic_pressure(3000.0, rho=1030.0), 1030.0 * 9.80665 * 3000.0
    )
    # Bowers loading: Pp = OB - ((V-V0)/A)^(1/B)
    ob, vel = 6000.0, 8000.0
    np.testing.assert_allclose(
        ag.bowers_pore_pressure(vel, ob, a=10.0, b=0.7, v0=5000.0),
        ob - ((vel - 5000.0) / 10.0) ** (1.0 / 0.7),
        rtol=1e-12,
    )
    # unloading branch differs from loading
    loading = ag.bowers_pore_pressure(vel, ob)
    unloading = ag.bowers_pore_pressure(vel, ob, unloading=True)
    assert not np.isclose(loading, unloading)


def test_vectorized_broadcast() -> None:
    vp = np.array([3500.0, 4000.0, 4500.0])
    vs = np.array([2000.0, 2400.0, 2700.0])
    k, g = ag.moduli_from_velocity(vp, vs, 2500.0)
    assert k.shape == (3,) and g.shape == (3,)
    e, nu = ag.youngs_poisson_dynamic(vp, vs, 2500.0)
    assert np.all(nu > 0) and np.all(e > 0)
