"""Tests for petrolib.porosity_lithology: golden values, round trips, method
dispatch, and shadow equivalence against verbatim article bodies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import porosity_lithology as pl  # noqa: E402
from petrolib.testing import assert_matches_original  # noqa: E402

# --- shale / clay volume ------------------------------------------------------


def test_gamma_ray_index_golden_and_clip() -> None:
    gr = np.array([15.0, 40.0, 67.5, 120.0, 150.0])
    igr = pl.gamma_ray_index(gr, 15.0, 120.0)
    np.testing.assert_allclose(igr, [0.0, 25 / 105, 0.5, 1.0, 1.0], rtol=1e-12)
    # unclipped keeps the >1 overshoot
    raw = pl.gamma_ray_index(150.0, 15.0, 120.0, clip=None)
    np.testing.assert_allclose(raw, 135 / 105, rtol=1e-12)


def test_vshale_methods_golden() -> None:
    igr = 0.5
    # pass an already-computed IGR by using clean=0, shale=1
    np.testing.assert_allclose(pl.vshale_from_gr(igr, 0.0, 1.0, method="linear"), 0.5, rtol=1e-12)
    np.testing.assert_allclose(
        pl.vshale_from_gr(igr, 0.0, 1.0, method="larionov_tertiary"),
        0.083 * (2.0 ** (3.7 * 0.5) - 1.0),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        pl.vshale_from_gr(igr, 0.0, 1.0, method="larionov_older"),
        0.33 * (2.0 ** (2.0 * 0.5) - 1.0),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        pl.vshale_from_gr(igr, 0.0, 1.0, method="clavier"),
        1.7 - np.sqrt(3.38 - (0.5 + 0.7) ** 2),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        pl.vshale_from_gr(igr, 0.0, 1.0, method="steiber"), 0.5 * 0.5 / (1.5 - 0.5), rtol=1e-12
    )
    np.testing.assert_allclose(
        pl.vshale_from_gr(igr, 0.0, 1.0, method="stieber_gcur", gcur=2.0),
        (2.0 ** (2.0 * 0.5) - 1.0) / (2.0**2.0 - 1.0),
        rtol=1e-12,
    )
    # clean sand (IGR=0) reads Vsh=0 for every method except steiber, whose
    # 0.001 IGR floor leaves a small positive residual
    for m in ("linear", "larionov_tertiary", "larionov_older", "clavier", "stieber_gcur"):
        assert abs(float(pl.vshale_from_gr(0.0, 0.0, 1.0, method=m))) < 1e-9
    assert 0.0 < float(pl.vshale_from_gr(0.0, 0.0, 1.0, method="steiber")) < 1e-3


def test_vshale_bad_method_raises() -> None:
    with pytest.raises(ValueError, match="method must be"):
        pl.vshale_from_gr(0.5, 0.0, 1.0, method="nope")


def test_vshale_neutron_density_and_combine() -> None:
    vsh = pl.vshale_neutron_density(0.30, 0.10, 0.45, 0.05)
    np.testing.assert_allclose(vsh, (0.30 - 0.10) / (0.45 - 0.05), rtol=1e-12)
    np.testing.assert_allclose(pl.combine_clay_indicators(0.2, 0.4, how="mean"), 0.3, rtol=1e-12)
    np.testing.assert_allclose(pl.combine_clay_indicators(0.2, 0.4, how="max"), 0.4, rtol=1e-12)
    np.testing.assert_allclose(pl.combine_clay_indicators(0.2, 0.4, how="min"), 0.2, rtol=1e-12)
    with pytest.raises(ValueError, match="how must be"):
        pl.combine_clay_indicators(0.2, how="median")


# --- porosity from logs -------------------------------------------------------


def _orig_density_porosity(rho_b, rho_ma, rho_fl):  # src2015_06/article1
    return (rho_ma - rho_b) / (rho_ma - rho_fl)


def test_density_porosity_shadow() -> None:
    assert_matches_original(
        _orig_density_porosity,
        lambda rb, rma, rfl: pl.density_porosity(rb, rma, rfl),
        [(2.30, 2.65, 1.0), (2.45, 2.71, 1.0), (2.20, 2.85, 1.01)],
    )
    # default sandstone matrix / fresh water
    np.testing.assert_allclose(pl.density_porosity(2.35), (2.65 - 2.35) / (2.65 - 1.0), rtol=1e-12)


def test_neutron_density_porosity_methods() -> None:
    n, d = 0.24, 0.16
    np.testing.assert_allclose(
        pl.neutron_density_porosity(n, d, method="rms"), np.sqrt((n**2 + d**2) / 2.0), rtol=1e-12
    )
    np.testing.assert_allclose(pl.neutron_density_porosity(n, d, method="mean"), 0.20, rtol=1e-12)
    with pytest.raises(ValueError, match="method must be"):
        pl.neutron_density_porosity(n, d, method="geometric")


def test_effective_porosity_clip() -> None:
    np.testing.assert_allclose(
        pl.effective_porosity(0.25, 0.30, 0.30), 0.25 - 0.30 * 0.30, rtol=1e-12
    )
    # default clip removes negative effective porosity
    assert float(pl.effective_porosity(0.05, 0.9, 0.30)) == 0.0
    # unclipped keeps it
    assert float(pl.effective_porosity(0.05, 0.9, 0.30, clip=None)) < 0.0


# --- core / digital rock ------------------------------------------------------


def test_core_porosity_forms() -> None:
    np.testing.assert_allclose(pl.porosity_from_volumes(10.0, 8.0), 0.2, rtol=1e-12)
    np.testing.assert_allclose(pl.boyle_porosity(2.12, 2.65), 1.0 - 2.12 / 2.65, rtol=1e-12)
    np.testing.assert_allclose(
        pl.boyle_grain_volume(50.0, 80.0, 150.0, 100.0),
        50.0 - 80.0 / (150.0 / 100.0 - 1.0),
        rtol=1e-12,
    )
    np.testing.assert_allclose(pl.fluid_summation_porosity(0.05, 0.10, 0.02), 0.17, rtol=1e-12)
    np.testing.assert_allclose(pl.porosity_from_voxel_count(300, 1000), 0.3, rtol=1e-12)
    np.testing.assert_allclose(
        pl.gravimetric_porosity(100.0, 112.0, 60.0, rho_fluid=1.0), 12.0 / 60.0, rtol=1e-12
    )


def test_ct_porosity_saturation() -> None:
    np.testing.assert_allclose(
        pl.ct_porosity(2.0, 2.71, 1.0), (2.71 - 2.0) / (2.71 - 1.0), rtol=1e-12
    )
    np.testing.assert_allclose(pl.ct_saturation(1.5, 1.0, 2.0), 0.5, rtol=1e-12)
    # saturation clips to [0,1] by default
    assert float(pl.ct_saturation(2.5, 1.0, 2.0)) == 1.0


# --- mixing laws --------------------------------------------------------------


def _orig_matrix_density_harmonic(w, rho, w_ker, rho_ker):  # src2018_06/article2
    return 1.0 / (sum(wi / ri for wi, ri in zip(w, rho, strict=True)) + w_ker / rho_ker)


def test_matrix_density_mixing() -> None:
    v = np.array([0.6, 0.3, 0.1])
    rho = np.array([2.65, 2.71, 4.99])
    np.testing.assert_allclose(pl.matrix_density_from_volumes(v, rho), float(v @ rho), rtol=1e-12)
    # harmonic (mass) with kerogen term matches the article body
    w = np.array([0.7, 0.3])
    rg = np.array([2.65, 2.72])
    assert_matches_original(
        _orig_matrix_density_harmonic,
        lambda w, rho, wk, rk: pl.matrix_density_from_masses(w, rho, w_kerogen=wk, rho_kerogen=rk),
        [(w, rg, 0.05, 1.43)],
    )


def test_bulk_density_and_fluid_density() -> None:
    # 3-component reduces to 2-component at v_k=0
    np.testing.assert_allclose(
        pl.bulk_density(0.20, 2.65, 1.0), (1.0 - 0.20) * 2.65 + 0.20 * 1.0, rtol=1e-12
    )
    np.testing.assert_allclose(
        pl.bulk_density(0.15, 2.68, 1.0, v_k=0.08, rho_k=1.30),
        (1.0 - 0.15 - 0.08) * 2.68 + 0.08 * 1.30 + 0.15 * 1.0,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        pl.fluid_density([0.7, 0.3], [1.0, 0.8]), 0.7 * 1.0 + 0.3 * 0.8, rtol=1e-12
    )
    # bulk_density with a fluid_density input is self-consistent
    rho_fl = pl.fluid_density([0.6, 0.4], [1.0, 0.2])
    np.testing.assert_allclose(
        pl.bulk_density(0.25, 2.65, rho_fl), (1 - 0.25) * 2.65 + 0.25 * float(rho_fl), rtol=1e-12
    )


def test_log_response_and_fraction_roundtrip() -> None:
    endpoints = np.array([[2.65, 2.71, 1.0], [16.0, 8.0, 700.0]])
    volumes = np.array([0.5, 0.3, 0.2])
    np.testing.assert_allclose(pl.log_response(volumes, endpoints), endpoints @ volumes, rtol=1e-12)
    np.testing.assert_allclose(pl.electron_density_to_bulk(2.5), 1.0704 * 2.5 - 0.1883, rtol=1e-12)
    # volume -> weight -> volume round trips
    v = np.array([0.5, 0.3, 0.2])
    rho = np.array([2.65, 2.71, 4.99])
    w = pl.volume_to_weight_fractions(v, rho)
    np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-12)
    np.testing.assert_allclose(pl.weight_to_volume_fractions(w, rho), v, rtol=1e-12)


# --- Thomas-Stieber -----------------------------------------------------------


def test_thomas_stieber_roundtrip() -> None:
    phi_sand, phi_sh, v_lam = 0.30, 0.10, 0.35
    phi_t = pl.thomas_stieber_phit(v_lam, phi_sand, phi_sh)
    np.testing.assert_allclose(pl.thomas_stieber_vlam(phi_t, phi_sand, phi_sh), v_lam, rtol=1e-12)
    # sand porosity recovers phi_sand when the laminar model is exact
    np.testing.assert_allclose(
        pl.thomas_stieber_sand_porosity(phi_t, v_lam, phi_sh), phi_sand, rtol=1e-12
    )


# --- TOC / kerogen ------------------------------------------------------------


def _orig_toc_schmoker(rho_b):  # src2016_04/article2
    return 154.497 / rho_b - 57.261


def test_toc_and_kerogen() -> None:
    assert_matches_original(
        _orig_toc_schmoker, lambda rb: pl.toc_schmoker(rb), [(2.35,), (2.50,), (2.60,)]
    )
    np.testing.assert_allclose(pl.kerogen_mass_fraction(0.05, k=1.2), 0.06, rtol=1e-12)
    np.testing.assert_allclose(
        pl.kerogen_volume_from_toc(0.05, 2.4, rho_k=1.30, carbon_frac=0.80),
        (0.05 / 0.80) * 2.4 / 1.30,
        rtol=1e-12,
    )
    # Passey dlogR with sonic overlay
    rt, dt, rt_b, dt_b, lom = 50.0, 90.0, 5.0, 70.0, 10.0
    dlogr = np.log10(rt / rt_b) + 0.02 * (dt - dt_b)
    np.testing.assert_allclose(
        pl.toc_passey_dlogr(rt, dt, rt_b, dt_b, lom=lom),
        dlogr * 10.0 ** (2.297 - 0.1688 * lom),
        rtol=1e-12,
    )


# --- multimineral inversion ---------------------------------------------------


def test_multimineral_solve_recovers_mixture() -> None:
    # 3 logs, 3 minerals with well-separated end-members (rho_b, nphi, pef)
    endpoints = np.array(
        [
            [2.65, 2.71, 2.87],  # rho_b
            [0.05, 0.30, 0.12],  # nphi
            [1.81, 5.08, 3.14],  # pef
        ]
    )
    v_true = np.array([0.5, 0.3, 0.2])
    measured = endpoints @ v_true
    # the square (unconstrained) system recovers the mixture exactly (numpy only)
    v = pl.multimineral_solve(measured, endpoints, closure=False, method="lstsq")
    np.testing.assert_allclose(v, v_true, rtol=1e-9)
    # the simplex projected-gradient solver lands near the truth and sums to one
    v = pl.multimineral_solve(measured, endpoints, method="simplex")
    np.testing.assert_allclose(v, v_true, atol=1e-2)
    np.testing.assert_allclose(v.sum(), 1.0, atol=1e-2)


def test_multimineral_nnls() -> None:
    pytest.importorskip("scipy")  # the nnls method imports scipy lazily
    endpoints = np.array([[2.65, 2.71, 2.87], [0.05, 0.30, 0.12], [1.81, 5.08, 3.14]])
    v_true = np.array([0.5, 0.3, 0.2])
    v = pl.multimineral_solve(endpoints @ v_true, endpoints, method="nnls")
    np.testing.assert_allclose(v, v_true, atol=1e-2)
    np.testing.assert_allclose(v.sum(), 1.0, atol=1e-2)


def test_multimineral_bad_method_raises() -> None:
    with pytest.raises(ValueError, match="method must be"):
        pl.multimineral_solve(np.ones(3), np.eye(3), method="bayes")


# --- volumetrics / cutoffs / net pay ------------------------------------------


def test_volumetrics_and_pay() -> None:
    np.testing.assert_allclose(pl.bulk_volume_water(0.20, 0.35), 0.07, rtol=1e-12)
    np.testing.assert_allclose(pl.hydrocarbon_pore_volume(0.20, 0.35), 0.20 * 0.65, rtol=1e-12)
    phi = np.array([0.05, 0.12, 0.18, 0.03])
    vsh = np.array([0.50, 0.20, 0.10, 0.60])
    sw = np.array([0.90, 0.40, 0.30, 0.80])
    flag = pl.pay_flag(phi, vsh, sw, phi_cut=0.08, vsh_cut=0.40, sw_cut=0.60)
    np.testing.assert_array_equal(flag, [False, True, True, False])
    # phi-only flag ignores the missing cutoffs
    np.testing.assert_array_equal(pl.pay_flag(phi, phi_cut=0.08), [False, True, True, False])


def test_net_to_gross() -> None:
    depth = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    net = np.array([False, True, True, False, True])
    # gross = whole interval (5 samples, unit spacing -> ~5 m gross, 3 m net)
    ng = pl.net_to_gross(depth, net)
    assert 0.0 < ng < 1.0
    # with a reservoir gross flag the ratio rises
    reservoir = np.array([False, True, True, True, True])
    ng_res = pl.net_to_gross(depth, net, reservoir)
    assert ng_res > ng
    # net thickness is a positive scalar
    assert pl.interval_thickness(depth, net) > 0.0
