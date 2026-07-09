"""Tests for petrolib.em_dielectric: golden values, model limits, round trips,
and error paths across complex permittivity, dispersion, CRIM mixing,
effective-medium theories, and the induction / propagation relations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import em_dielectric as em  # noqa: E402

# --- complex permittivity algebra ---------------------------------------------


def test_complex_permittivity_and_sigma_inverse() -> None:
    w = 2.0 * np.pi * 1e9
    z = em.complex_permittivity(20.0, sigma=0.5, freq_hz=1e9, eps_imag=2.0)
    np.testing.assert_allclose(z.real, 20.0)
    np.testing.assert_allclose(-z.imag, 2.0 + 0.5 / (w * em.EPS0), rtol=1e-12)
    # no ohmic term when sigma is zero (freq_hz not required)
    z0 = em.complex_permittivity(20.0, eps_imag=2.0)
    np.testing.assert_allclose([z0.real, -z0.imag], [20.0, 2.0])
    with pytest.raises(ValueError, match="freq_hz"):
        em.complex_permittivity(20.0, sigma=0.5)


def test_imag_sigma_round_trip_and_loss_tangent() -> None:
    ei = em.imag_permittivity_from_sigma(0.5, 1e9)
    np.testing.assert_allclose(em.sigma_from_imag_permittivity(ei, 1e9), 0.5, rtol=1e-12)
    np.testing.assert_allclose(em.loss_tangent(20.0 - 5j), 5.0 / 20.0, rtol=1e-12)


def test_impedivity_and_water_permittivity() -> None:
    # impedivity is the reciprocal of the complex admittivity
    zt = em.impedivity(20.0, 0.5, 1e9)
    adm = 1j * 2.0 * np.pi * 1e9 * em.EPS0 * 20.0 + 0.5
    np.testing.assert_allclose(zt, 1.0 / adm, rtol=1e-12)
    # brine permittivity: real part fixed, loss grows as Rw drops
    zw_fresh = em.water_permittivity(10.0, 1e9)
    zw_saline = em.water_permittivity(0.1, 1e9)
    np.testing.assert_allclose(zw_fresh.real, 55.0)
    assert -zw_saline.imag > -zw_fresh.imag


# --- dispersion models --------------------------------------------------------


def test_cole_cole_and_hn_reduce_to_debye() -> None:
    f = np.array([1e6, 1e8, 1e10])
    kw = {"eps_inf": 5.0, "eps_s": 80.0, "tau": 1e-9}
    d = em.debye(f, **kw)
    np.testing.assert_allclose(em.cole_cole(f, alpha=0.0, **kw), d, rtol=1e-12)
    np.testing.assert_allclose(em.havriliak_negami(f, alpha=1.0, beta=1.0, **kw), d, rtol=1e-12)
    # Debye endpoints: eps_s at DC, eps_inf at high frequency
    np.testing.assert_allclose(em.debye(1.0, **kw).real, 80.0, atol=1e-6)
    np.testing.assert_allclose(em.debye(1e15, **kw).real, 5.0, atol=1e-6)


def test_pelton_cole_cole_resistivity_limits() -> None:
    kw = {"rho0": 10.0, "chargeability": 0.4, "tau": 1.0, "c": 0.5}
    # DC -> rho0 ; high frequency -> rho0*(1 - m)
    np.testing.assert_allclose(em.cole_cole_resistivity(1e-12, **kw).real, 10.0, atol=1e-4)
    np.testing.assert_allclose(em.cole_cole_resistivity(1e12, **kw).real, 6.0, atol=1e-4)


# --- CRIM mixing --------------------------------------------------------------


def test_crim_forward_inverse_round_trip() -> None:
    phi, sw = 0.25, 0.6
    props = {"eps_w": 78.0, "eps_hc": 2.2, "eps_matrix": 5.0}
    eps = em.crim(phi, sw, **props)
    np.testing.assert_allclose(em.sw_from_permittivity(eps, phi, **props), sw, rtol=1e-9)
    np.testing.assert_allclose(em.bvw_from_permittivity(eps, phi, **props), phi * sw, rtol=1e-9)
    # mix_power_law with the same volumes reproduces the CRIM result
    mpl = em.mix_power_law([phi * sw, phi * (1 - sw), 1 - phi], [78.0, 2.2, 5.0], alpha=0.5)
    np.testing.assert_allclose(mpl, eps, rtol=1e-12)


def test_sw_clip_and_water_filled_porosity() -> None:
    props = {"eps_w": 78.0, "eps_hc": 2.2, "eps_matrix": 5.0}
    # an over-high measured permittivity clips Sw to 1
    high = em.crim(0.25, 1.0, **props) + 5.0
    np.testing.assert_allclose(em.sw_from_permittivity(high, 0.25, **props), 1.0)
    assert em.sw_from_permittivity(high, 0.25, clip=False, **props) > 1.0
    # 2-component inversion recovers a purely water-filled porosity
    eps = em.crim(0.3, 1.0, eps_w=78.0, eps_hc=2.2, eps_matrix=5.0)
    np.testing.assert_allclose(
        em.water_filled_porosity(eps, eps_matrix=5.0, eps_w=78.0), 0.3, rtol=1e-9
    )


# --- effective-medium theories ------------------------------------------------


def test_maxwell_garnett_endpoints_and_complex() -> None:
    np.testing.assert_allclose(em.maxwell_garnett(5.0, 80.0, 0.0), 5.0, rtol=1e-12)
    mg = em.maxwell_garnett(5.0, 80.0, 0.2)
    assert 5.0 < float(mg) < 80.0
    mgc = em.maxwell_garnett(5.0 - 0.1j, 80.0 - 30.0j, 0.2)
    assert np.iscomplexobj(mgc)


def test_bruggeman_symmetric_residual_and_degenerate() -> None:
    np.testing.assert_allclose(em.bruggeman_symmetric([0.5, 0.5], [10.0, 10.0]), 10.0, rtol=1e-9)
    e, f = np.array([80.0, 5.0]), np.array([0.3, 0.7])
    eff = em.bruggeman_symmetric(f, e)
    assert abs(float(np.sum(f * (e - eff) / (e + 2.0 * eff)))) < 1e-10


def test_hanai_bruggeman_satisfies_implicit_equation() -> None:
    eh, ei, fi = 5.0, 80.0, 0.3
    eff = em.hanai_bruggeman(eh, ei, fi)
    lhs = ((eff - ei) / (eh - ei)) * (eh / eff) ** (1.0 / 3.0)
    np.testing.assert_allclose(lhs, 1.0 - fi, atol=1e-8)
    np.testing.assert_allclose(em.hanai_bruggeman(eh, ei, 0.0), eh, atol=1e-6)


def test_depolarization_spheroid_limits_and_sum() -> None:
    np.testing.assert_allclose(em.depolarization_spheroid(1.0), (1 / 3, 1 / 3, 1 / 3))
    lx, ly, lz = em.depolarization_spheroid([0.5, 1.0, 3.0])
    np.testing.assert_allclose(lx + ly + lz, 1.0)
    # oblate concentrates depolarization on the short polar axis; prolate the reverse
    assert float(lz[0]) > 1 / 3 and float(lz[2]) < 1 / 3


# --- induction / propagation --------------------------------------------------


def test_skin_depth_induction_and_phase_atten() -> None:
    delta = em.skin_depth(10.0, 2e4)
    np.testing.assert_allclose(
        delta, np.sqrt(2.0 * 10.0 / (2.0 * np.pi * 2e4 * em.MU0)), rtol=1e-12
    )
    n = em.induction_number(1.0, 10.0, 2e4)
    np.testing.assert_allclose(n, 1.0 / delta, rtol=1e-12)
    np.testing.assert_allclose(em.phase_shift_deg(10.0, 2e4, 1.0), np.degrees(n), rtol=1e-12)
    np.testing.assert_allclose(em.attenuation_db(10.0, 2e4, 1.0), 8.686 * n, rtol=1e-12)
    # phase -> resistivity inverse
    ph = em.phase_shift_deg(10.0, 2e4, 1.0)
    np.testing.assert_allclose(em.resistivity_from_phase(ph, 2e4, 1.0), 10.0, rtol=1e-9)


def test_complex_wavenumber_round_trip() -> None:
    kr, ki = em.complex_wavenumber(1e8, 0.5, 20.0)
    sigma, eps_r = em.sigma_eps_from_wavenumber(kr, ki, 1e8)
    np.testing.assert_allclose(sigma, 0.5, rtol=1e-9)
    np.testing.assert_allclose(eps_r, 20.0, rtol=1e-9)


def test_attenuation_phase_from_voltages() -> None:
    atten, phase = em.attenuation_phase_from_voltages(1.0, 0.5 * np.exp(-0.5j))
    np.testing.assert_allclose(atten, 20.0 * np.log10(2.0), rtol=1e-12)
    np.testing.assert_allclose(phase, np.degrees(0.5), rtol=1e-12)


# --- anisotropy / Doll geometry -----------------------------------------------


def test_anisotropy_and_doll() -> None:
    np.testing.assert_allclose(em.anisotropy_coefficient(2.0, 8.0), 2.0, rtol=1e-12)
    # isotropic medium: apparent resistivity is dip-independent
    np.testing.assert_allclose(em.apparent_resistivity_dip(4.0, 4.0, 30.0), 4.0, rtol=1e-12)
    # r == L/2 gives G = 1/2; two-zone conductivity is the G-weighted average
    np.testing.assert_allclose(em.doll_radial_geometric_factor(1.0, 2.0), 0.5, rtol=1e-12)
    np.testing.assert_allclose(
        em.apparent_conductivity_two_zone(0.1, 0.02, 1.0, 2.0), 0.5 * 0.1 + 0.5 * 0.02, rtol=1e-12
    )
