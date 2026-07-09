"""Tests for petrolib.flow_transport: golden values, round-trip identities,
hazard-trap pins, and shadow equivalence against verbatim article bodies."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import flow_transport as ft  # noqa: E402
from petrolib.testing import assert_matches_original  # noqa: E402

PHI = np.array([0.06, 0.10, 0.18, 0.25, 0.33])
K_MD = np.array([0.05, 1.0, 10.0, 100.0, 800.0])


# --- Darcy single-phase & gas -------------------------------------------------


def _original_darcy_permeability_2015_04(flow_rate, viscosity, length, area, dp):
    return flow_rate * viscosity * length / (area * dp)


def _original_darcy_rate_2018_08(k, area, dp, mu, length):
    return k * area * dp / (mu * length)


def _original_darcy_gas_permeability_2016_02(q, viscosity, length, area, p0, p1, p_ref):
    return 2.0 * q * viscosity * length * p_ref / (area * (p0**2 - p1**2))


def test_darcy_equivalence_and_roundtrip() -> None:
    assert_matches_original(
        _original_darcy_permeability_2015_04,
        lambda q, mu, L, A, dp: ft.darcy_permeability(q, mu=mu, length=L, area=A, dp=dp),
        [(1e-6, 1e-3, 0.05, 1e-4, 5e5), (np.array([1e-6, 2e-6]), 1e-3, 0.05, 1e-4, 5e5)],
    )
    assert_matches_original(
        _original_darcy_rate_2018_08,
        lambda k, A, dp, mu, L: ft.darcy_rate(k, area=A, dp=dp, mu=mu, length=L),
        [(2e-13, 1e-4, 5e5, 1e-3, 0.05)],
    )
    assert_matches_original(
        _original_darcy_gas_permeability_2016_02,
        lambda q, mu, L, A, p0, p1, pr: ft.darcy_gas_permeability(
            q, mu=mu, length=L, area=A, p_up=p0, p_down=p1, p_ref=pr
        ),
        [(1e-6, 1e-3, 0.05, 1e-4, 3e5, 1e5, 1e5)],
    )
    # k -> q -> k round trip
    mu, length, area, dp = 1e-3, 0.05, 1e-4, 5e5
    k = np.array([1e-14, 1e-13, 1e-12])
    q = ft.darcy_rate(k, area=area, dp=dp, mu=mu, length=length)
    np.testing.assert_allclose(
        ft.darcy_permeability(q, mu=mu, length=length, area=area, dp=dp), k, rtol=1e-12
    )
    # pressure_drop carries kr and inverts darcy_rate at kr=1
    dp_back = ft.darcy_pressure_drop(q, mu=mu, k=k, area=area, length=length)
    np.testing.assert_allclose(dp_back, dp, rtol=1e-12)


# --- Klinkenberg & Knudsen ----------------------------------------------------


def _original_klinkenberg_apparent_2016_02(kl, b, pm):
    return kl * (1.0 + b / pm)


def _original_klinkenberg_second_order_2019_06(k_l, b, c, p_mean):
    return k_l * (1.0 + b / p_mean + c / p_mean**2)


def _original_klinkenberg_corrected_2021_12(k_gas, b_slip, p_mean_bar):
    return k_gas / (1.0 + b_slip / p_mean_bar)


def _original_mfp_viscosity_2016_02(viscosity, pressure, temperature, molar_mass):
    return (viscosity / pressure) * np.sqrt(
        np.pi * 8.31446261815324 * temperature / (2.0 * molar_mass)
    )


def _original_mfp_kinetic_2018_08(temperature, pressure, collision_diameter):
    return 1.380649e-23 * temperature / (np.sqrt(2.0) * np.pi * collision_diameter**2 * pressure)


def test_klinkenberg_forms_and_hazard() -> None:
    pm = np.array([1.0, 2.0, 5.0, 10.0])
    assert_matches_original(
        _original_klinkenberg_apparent_2016_02,
        lambda kl, b, p: ft.klinkenberg_apparent(kl, b=b, p_mean=p),
        [(5.0, 0.3, pm)],
    )
    assert_matches_original(
        _original_klinkenberg_second_order_2019_06,
        lambda kl, b, c, p: ft.klinkenberg_apparent(kl, b=b, p_mean=p, c2=c),
        [(5.0, 0.3, 0.1, pm)],
    )
    assert_matches_original(
        _original_klinkenberg_corrected_2021_12,
        lambda kg, b, p: ft.klinkenberg_corrected(kg, b=b, p_mean=p),
        [(6.0, 0.3, pm)],
    )
    # HAZARD PIN: apparent and corrected are inverses, never the same value
    k_app = ft.klinkenberg_apparent(5.0, b=0.3, p_mean=pm)
    assert not np.allclose(k_app, ft.klinkenberg_corrected(5.0, b=0.3, p_mean=pm))
    np.testing.assert_allclose(ft.klinkenberg_corrected(k_app, b=0.3, p_mean=pm), 5.0, rtol=1e-12)
    # fit recovers (k_inf, b)
    k_inf, b = ft.fit_klinkenberg(pm, ft.klinkenberg_apparent(4.2, b=0.5, p_mean=pm))
    assert (k_inf, b) == pytest.approx((4.2, 0.5), rel=1e-9)


def test_mean_free_path_branches_and_regime() -> None:
    # the two forms are physically distinct: pin both against their bodies
    assert_matches_original(
        _original_mfp_viscosity_2016_02,
        lambda mu, p, T, M: ft.mean_free_path(pressure=p, temperature=T, mu=mu, molar_mass=M),
        [(1.8e-5, 1e6, 300.0, 0.028)],
    )
    assert_matches_original(
        _original_mfp_kinetic_2018_08,
        lambda T, p, d: ft.mean_free_path(pressure=p, temperature=T, d_collision=d),
        [(300.0, 1e6, 3.8e-10)],
    )
    with pytest.raises(ValueError):
        ft.mean_free_path(pressure=1e6, temperature=300.0)
    assert ft.knudsen_number(1e-7, 1e-6) == pytest.approx(0.1)
    assert ft.flow_regime(1e-4) == "continuum"
    assert ft.flow_regime(0.05) == "slip"
    assert ft.flow_regime(1.0) == "transition"
    assert ft.flow_regime(50.0) == "free-molecular"


# --- Stress-dependent permeability --------------------------------------------


def _original_stress_perm_reference_2018_02(ki, gamma, ncs, ncs_i):
    return ki * np.exp(-gamma * (ncs - ncs_i))


def test_stress_permeability() -> None:
    ncs = np.array([5.0, 10.0, 20.0, 40.0])
    # plain form (ncs0=0)
    np.testing.assert_allclose(
        ft.stress_permeability(100.0, gamma=0.02, ncs=ncs), 100.0 * np.exp(-0.02 * ncs), rtol=1e-12
    )
    # reference-stress form
    assert_matches_original(
        _original_stress_perm_reference_2018_02,
        lambda ki, g, n, ni: ft.stress_permeability(ki, gamma=g, ncs=n, ncs0=ni),
        [(80.0, 0.03, ncs, 5.0)],
    )
    assert ft.net_confining_stress(30.0, pore_pressure=10.0, biot=0.9) == pytest.approx(21.0)
    # fit recovers (k0, gamma)
    k0, gamma = ft.fit_stress_permeability(ncs, 100.0 * np.exp(-0.02 * ncs))
    assert (k0, gamma) == pytest.approx((100.0, 0.02), rel=1e-9)


# --- Kozeny-Carman ------------------------------------------------------------


def _original_kozeny_carman_fvm_2019_06(porosity, tau, surface_area, c=2.0):
    return porosity**3 / (c * tau**2 * surface_area**2)


def _original_kozeny_carman_grain_2022_12(phi, Sv, c=5.0):
    return phi**3 / (c * Sv**2 * (1.0 - phi) ** 2 + 1e-30)


def _original_kozeny_carman_update_2023_10(phi, phi_new, k):
    return k * (phi_new / phi) ** 3 * ((1.0 - phi) / (1.0 - phi_new)) ** 2


def _original_kozeny_carman_simplified_2020_04(k0, phi, phi0):
    return k0 * (phi / phi0) ** 3


def test_kozeny_carman() -> None:
    s = np.array([1e6, 5e5, 2e5])
    phi = np.array([0.10, 0.18, 0.25])
    # surface-area, no grain term (fvm c=2, tau)
    assert_matches_original(
        _original_kozeny_carman_fvm_2019_06,
        lambda p, tau, sa, c=2.0: ft.kozeny_carman(
            p, specific_surface=sa, tau=tau, c=c, grain_term=False
        ),
        [(phi, 1.5, s)],
    )
    # surface-area with grain term; the 2022_12 body adds a 1e-30 guard the
    # library omits, so compare where the guard is negligible (S large)
    np.testing.assert_allclose(
        ft.kozeny_carman(phi, specific_surface=s, c=5.0, grain_term=True),
        _original_kozeny_carman_grain_2022_12(phi, s, 5.0),
        rtol=1e-12,
    )
    # ratio/update forms
    assert_matches_original(
        _original_kozeny_carman_update_2023_10,
        lambda phi, phi_new, k: ft.kozeny_carman_ratio(k, phi_new, phi, grain_term=True),
        [(0.20, np.array([0.12, 0.18, 0.24]), 100.0)],
    )
    assert_matches_original(
        _original_kozeny_carman_simplified_2020_04,
        lambda k0, phi, phi0: ft.kozeny_carman_ratio(k0, phi, phi0, grain_term=False),
        [(100.0, phi, 0.20)],
    )


# --- Winland / Swanson / Lucia / poro-perm ------------------------------------


def _original_winland_r35_2015_02(k, phi):
    return 10.0 ** (0.732 + 0.588 * np.log10(k) - 0.864 * np.log10(np.asarray(phi, float) * 100.0))


def _original_swanson_2015_02(sb_pc_max, c=399.0, d=1.691):
    return c * np.asarray(sb_pc_max, float) ** d


def _original_lucia_k_2023_02(phi_g, rfn):
    a = 9.7982 - 12.0838 * np.log10(rfn)
    b = 8.6711 - 8.2965 * np.log10(rfn)
    return 10.0 ** (a + b * np.log10(phi_g))


def test_winland_swanson_lucia() -> None:
    assert_matches_original(
        _original_winland_r35_2015_02,
        lambda k, phi: ft.winland_r35(k, phi),
        [(K_MD, PHI)],
    )
    # r35 -> k -> r35 round trip
    r35 = ft.winland_r35(K_MD, PHI)
    np.testing.assert_allclose(ft.winland_permeability(r35, PHI), K_MD, rtol=1e-10)
    # Swanson: default c=399 and the stress-MICP c=339 variant
    assert_matches_original(
        _original_swanson_2015_02,
        lambda apex, c=399.0, d=1.691: ft.swanson_permeability(apex, c=c, d=d),
        [(np.array([0.02, 0.05, 0.1]),), (0.05, 339.0, 1.691)],
    )
    shg = np.array([10.0, 30.0, 50.0, 40.0])
    pc = np.array([100.0, 80.0, 120.0, 200.0])
    apex, idx = ft.micp_apex(shg, pc)
    assert idx == int(np.argmax(shg / pc))
    assert apex == pytest.approx((shg / pc).max())
    assert_matches_original(
        _original_lucia_k_2023_02,
        lambda pg, rfn: ft.lucia_permeability(pg, rfn),
        [(0.20, 1.7), (np.array([0.1, 0.2, 0.3]), 2.85)],
    )


def test_poroperm_powerlaw_roundtrip() -> None:
    a, b = 2.0, 3.5
    k = ft.poroperm_powerlaw(PHI, a=a, b=b)
    np.testing.assert_allclose(k, 10.0 ** (a + b * np.log10(PHI)), rtol=1e-12)
    a_fit, b_fit = ft.fit_poroperm(PHI, k)
    assert (a_fit, b_fit) == pytest.approx((a, b), rel=1e-9)


# --- RQI / FZI / HFU / averaging ----------------------------------------------


def _original_rqi_2023_02(k_md, phi):
    return 0.0314 * np.sqrt(k_md / phi)


def _original_permeability_from_fzi_2015_12(phi, fzi_value):
    phi_z = phi / (1.0 - phi)
    return phi * (fzi_value * phi_z / 0.0314) ** 2


def test_rqi_fzi_roundtrip_and_hfu() -> None:
    assert_matches_original(_original_rqi_2023_02, lambda k, phi: ft.rqi(k, phi), [(K_MD, PHI)])
    np.testing.assert_allclose(ft.phi_z(PHI), PHI / (1.0 - PHI), rtol=1e-12)
    fzi = ft.fzi(K_MD, PHI)
    np.testing.assert_allclose(fzi, ft.rqi(K_MD, PHI) / ft.phi_z(PHI), rtol=1e-12)
    # k_from_fzi is the exact inverse of rqi/fzi (matches the 2015_12 body)
    assert_matches_original(
        _original_permeability_from_fzi_2015_12,
        lambda phi, fz: ft.k_from_fzi(phi, fz),
        [(PHI, fzi)],
    )
    np.testing.assert_allclose(ft.k_from_fzi(PHI, fzi), K_MD, rtol=1e-10)
    # the 1014 rounding (src2023_02) is within ~2e-4, NOT bit-exact
    k_1014 = 1014.0 * fzi**2 * PHI**3 / (1.0 - PHI) ** 2
    assert np.allclose(k_1014, K_MD, rtol=1e-3) and not np.allclose(k_1014, K_MD, rtol=1e-12)
    labels, thresholds = ft.classify_hfu(fzi, n_units=4)
    assert labels.min() >= 0 and labels.max() <= 3
    assert thresholds.shape == (5,)


def test_permeability_average_and_bounds() -> None:
    k = np.array([10.0, 50.0, 100.0, 5.0])
    assert ft.permeability_average(k, method="arithmetic") == pytest.approx(k.mean())
    assert ft.permeability_average(k, method="harmonic") == pytest.approx(len(k) / np.sum(1.0 / k))
    assert ft.permeability_average(k, method="geometric") == pytest.approx(
        np.exp(np.mean(np.log(k)))
    )
    with pytest.raises(ValueError):
        ft.permeability_average(k, method="bogus")
    lo, hi = ft.wiener_bounds(k)
    # harmonic <= geometric <= arithmetic
    assert lo <= ft.permeability_average(k, method="geometric") <= hi
    assert lo == pytest.approx(len(k) / np.sum(1.0 / k))
    assert hi == pytest.approx(k.mean())


# --- diffusion & transport ----------------------------------------------------


def _original_diffusion_length_f2_2014_04(diffusion, time):
    return np.sqrt(2.0 * diffusion * np.asarray(time, float))


def _original_stokes_einstein_diameter_2014_10(temperature, viscosity, particle_diameter):
    return 1.380649e-23 * temperature / (3.0 * np.pi * viscosity * particle_diameter)


def _original_millington_quirk_2014_10(dd, phi, sw):
    phi_w = phi * sw
    return phi_w ** (10.0 / 3.0) / phi**2 * dd


def _original_erfc_profile_2015_08(c0, x, diffusion_coeff, time):
    arg = np.asarray(x, float) / (2.0 * np.sqrt(diffusion_coeff * time))
    return c0 * np.array([math.erfc(v) for v in np.atleast_1d(arg)])


def test_diffusion_group() -> None:
    # geometry factor: f=1 default vs f=2
    np.testing.assert_allclose(
        ft.diffusion_length(1e-9, 3600.0), np.sqrt(1e-9 * 3600.0), rtol=1e-12
    )
    assert_matches_original(
        _original_diffusion_length_f2_2014_04,
        lambda D, t: ft.diffusion_length(D, t, geometry_factor=2.0),
        [(1e-9, np.array([60.0, 600.0, 3600.0]))],
    )
    # length -> time inverse for a shared geometry factor
    L = ft.diffusion_length(1e-9, 3600.0, geometry_factor=2.0)
    assert ft.diffusion_time(L, 1e-9, geometry_factor=2.0) == pytest.approx(3600.0, rel=1e-12)
    # Stokes-Einstein: 6*pi*r canonical equals the 3*pi*d body with d=2r
    assert_matches_original(
        _original_stokes_einstein_diameter_2014_10,
        lambda T, mu, d: ft.stokes_einstein(T, mu, d / 2.0),
        [(300.0, 1e-3, 2e-9)],
    )
    assert_matches_original(
        _original_millington_quirk_2014_10,
        lambda dd, phi, sw: ft.millington_quirk(dd, phi, sw),
        [(1e-9, 0.2, 0.6)],
    )
    assert ft.pore_velocity(1e-5, 0.2, 0.8) == pytest.approx(1e-5 / (0.2 * 0.8))
    assert ft.fick_flux(1e-9, 2.0, 0.01) == pytest.approx(-1e-9 * 2.0 / 0.01)
    assert ft.early_time_uptake(1e-9, 3600.0, 0.02) == pytest.approx(
        (2.0 / 0.02) * np.sqrt(1e-9 * 3600.0 / np.pi)
    )


def test_erfc_profile_matches_article() -> None:
    pytest.importorskip("scipy")
    x = np.linspace(0.0, 0.05, 20)
    assert_matches_original(
        _original_erfc_profile_2015_08,
        lambda c0, x, D, t: ft.erfc_profile(c0, x, D=D, t=t),
        [(1.0, x, 1e-9, 3600.0)],
    )


def test_advect_disperse_1d_matches_article() -> None:
    def _original_transport_1d(c0, length, n_cells, total_time, v_w, d_star, k_dep):
        dx = length / n_cells
        dt_adv = dx / max(v_w, 1e-30)
        dt_dif = dx**2 / max(2.0 * d_star, 1e-30)
        dt = 0.4 * min(dt_adv, dt_dif)
        steps = int(np.ceil(total_time / dt))
        dt = total_time / steps
        c = np.zeros(n_cells)
        x = (np.arange(n_cells) + 0.5) * dx
        for _ in range(steps):
            cl = np.empty_like(c)
            cl[0] = c0
            cl[1:] = c[:-1]
            cr = np.empty_like(c)
            cr[-1] = c[-1]
            cr[:-1] = c[1:]
            adv = -v_w * (c - cl) / dx
            dif = d_star * (cr - 2.0 * c + cl) / dx**2
            dep = -k_dep * c
            c = c + dt * (adv + dif + dep)
            c = np.clip(c, 0.0, c0)
        return x, c

    x_ref, c_ref = _original_transport_1d(1.0, 0.1, 50, 3600.0, 1e-5, 1e-8, 1e-4)
    x, c = ft.advect_disperse_1d(
        1.0, length=0.1, n_cells=50, t_total=3600.0, v=1e-5, D=1e-8, k_rxn=1e-4
    )
    np.testing.assert_allclose(x, x_ref, rtol=1e-12)
    np.testing.assert_allclose(c, c_ref, rtol=1e-12)
