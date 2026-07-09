"""Tests for petrolib.nmr: golden values, round trips, and shadow equivalence
against verbatim article bodies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import nmr  # noqa: E402
from petrolib.constants import GAMMA_H  # noqa: E402
from petrolib.testing import assert_matches_original  # noqa: E402

T2 = np.array([1.0, 3.0, 10.0, 33.0, 100.0, 300.0, 1000.0])
AMP = np.array([0.5, 1.0, 2.0, 3.0, 2.5, 1.0, 0.3])


# --- relaxation physics -------------------------------------------------------


def _original_t2_total_2019_12(T2_bulk, rho, s_over_v, G, TE, D, gamma=GAMMA_H):
    inv = 1.0 / T2_bulk + rho * s_over_v + (gamma * G * TE) ** 2 * D / 12.0
    return 1.0 / inv


def _original_diffusion_rate_2018_04(diffusivity, gradient, te, gamma=GAMMA_H):
    return diffusivity * gamma**2 * gradient**2 * te**2 / 12.0


def test_relaxation_rate_and_apparent() -> None:
    # apparent time is the reciprocal of the rate
    kw = dict(t2_bulk=3.0, rho=5.0e-6, s_over_v=1.0e6, D=2.3e-9, G=0.1, TE=1e-3)
    np.testing.assert_allclose(nmr.t2_apparent(**kw), 1.0 / nmr.relaxation_rate(**kw), rtol=1e-12)
    # full three-mechanism form matches the article body
    assert_matches_original(
        _original_t2_total_2019_12,
        lambda tb, rho, sv, G, TE, D, gamma=GAMMA_H: nmr.t2_apparent(
            t2_bulk=tb, rho=rho, s_over_v=sv, G=G, TE=TE, D=D, gamma=gamma
        ),
        [(3.0, 5e-6, 1e6, 0.1, 1e-3, 2.3e-9)],
    )
    # diffusion-only rate, two algebraically identical orderings
    assert_matches_original(
        _original_diffusion_rate_2018_04,
        lambda D, G, te, gamma=GAMMA_H: nmr.diffusion_relaxation_rate(D, G=G, TE=te, gamma=gamma),
        [(2.3e-9, 0.1, 1e-3)],
    )


def test_surface_to_volume_pore_radius_roundtrip() -> None:
    # surface-only S/V is the inverse of 1/(rho*T2)
    sv = nmr.surface_to_volume(50.0, rho=10.0)
    np.testing.assert_allclose(sv, 1.0 / (10.0 * 50.0), rtol=1e-12)
    # pore radius round-trips back to the relaxivity
    r = nmr.pore_radius_from_t2(50.0, rho=10.0, shape_factor=3.0)
    np.testing.assert_allclose(r, 3.0 * 10.0 * 50.0, rtol=1e-12)
    np.testing.assert_allclose(
        nmr.surface_relaxivity_from_pore(50.0, r, shape_factor=3.0), 10.0, rtol=1e-12
    )
    # bulk-subtraction form (src2014_12/article3): r = 3*rho/(1/T2 - 1/T2bulk)
    np.testing.assert_allclose(
        nmr.pore_radius_from_t2(50.0, rho=10.0, shape_factor=3.0, t2_bulk=3000.0),
        3.0 * 10.0 / (1.0 / 50.0 - 1.0 / 3000.0),
        rtol=1e-12,
    )


def test_combine_and_larmor() -> None:
    # parallel-rate combination
    np.testing.assert_allclose(
        nmr.combine_relaxation_times(3000.0, 100.0, 500.0),
        1.0 / (1.0 / 3000.0 + 1.0 / 100.0 + 1.0 / 500.0),
        rtol=1e-12,
    )
    # Larmor frequency at 0.05 T
    np.testing.assert_allclose(
        nmr.larmor_frequency(0.05), GAMMA_H * 0.05 / (2.0 * np.pi), rtol=1e-12
    )


# --- T2 statistics ------------------------------------------------------------


def _original_t2_logmean_2014_06(t2, amplitudes):
    a = np.asarray(amplitudes, float)
    return float(np.exp(np.sum(a * np.log(t2)) / np.sum(a)))


def _original_ffi_bvi_2014_06(t2, amplitudes, t2_cutoff=100.0):
    a = np.asarray(amplitudes, float)
    bvi = float(a[t2 < t2_cutoff].sum())
    ffi = float(a[t2 >= t2_cutoff].sum())
    return ffi, bvi


def test_t2_logmean_and_partition() -> None:
    assert_matches_original(
        _original_t2_logmean_2014_06,
        lambda t2, a: nmr.t2_logmean(t2, a),
        [(T2, AMP)],
    )
    # nan on zero mass
    assert np.isnan(nmr.t2_logmean(T2, np.zeros_like(AMP)))
    # total porosity
    np.testing.assert_allclose(nmr.total_porosity(AMP), AMP.sum(), rtol=1e-12)
    # BVI/FFI split matches the article's (ffi, bvi) with strict/loose boundaries
    ffi_o, bvi_o = _original_ffi_bvi_2014_06(T2, AMP, 100.0)
    bvi, ffi = nmr.bvi_ffi(T2, AMP, cutoff_ms=100.0)
    np.testing.assert_allclose((bvi, ffi), (bvi_o, ffi_o), rtol=1e-12)
    # three-way CBW/capillary/free partition sums to the total
    bands = nmr.t2_partition(T2, AMP, cutoffs_ms=(3.0, 33.0))
    assert len(bands) == 3
    np.testing.assert_allclose(sum(bands), AMP.sum(), rtol=1e-12)
    fr = nmr.t2_partition(T2, AMP, cutoffs_ms=(3.0, 33.0), fractions=True)
    np.testing.assert_allclose(sum(fr), 1.0, rtol=1e-12)


# --- forward models -----------------------------------------------------------


def _original_multiexp_2018_04(t, amplitudes, t2s):
    t = np.asarray(t, float)
    a = np.asarray(amplitudes, float)
    t2 = np.asarray(t2s, float)
    return (a * np.exp(-t[:, None] / t2[None, :])).sum(axis=1)


def _original_cpmg_kernel_2022_06(t_echo_s, T2_axis_s):
    return np.exp(-t_echo_s[:, None] / T2_axis_s[None, :])


def _original_inversion_recovery_2019_06(t, M0, T1):
    return M0 * (1.0 - 2.0 * np.exp(-np.asarray(t, float) / T1))


def _original_saturation_recovery_2019_06(t, M0, T1):
    return M0 * (1.0 - np.exp(-np.asarray(t, float) / T1))


def test_forward_models() -> None:
    techo = np.array([1.0, 5.0, 20.0, 80.0])
    t2grid = np.array([2.0, 20.0, 200.0])
    assert_matches_original(
        _original_cpmg_kernel_2022_06,
        lambda t, g: nmr.cpmg_kernel(t, g),
        [(techo, t2grid)],
    )
    assert_matches_original(
        _original_multiexp_2018_04,
        lambda t, a, t2: nmr.multiexp_decay(t, a, t2),
        [(techo, np.array([1.0, 2.0, 0.5]), t2grid)],
    )
    assert_matches_original(
        _original_inversion_recovery_2019_06,
        lambda t, m0, t1: nmr.t1_inversion_recovery(t, m0, t1),
        [(np.array([1.0, 10.0, 100.0]), 1.0, 50.0)],
    )
    assert_matches_original(
        _original_saturation_recovery_2019_06,
        lambda t, m0, t1: nmr.t1_saturation_recovery(t, m0, t1),
        [(np.array([1.0, 10.0, 100.0]), 1.0, 50.0)],
    )


def test_t1t2_kernel_modes() -> None:
    techo, tw = np.array([1.0, 5.0]), np.array([10.0, 100.0, 1000.0])
    t1g, t2g = np.array([50.0, 500.0]), np.array([5.0, 50.0])
    k_sat = nmr.t1t2_kernel(techo, tw, t1g, t2g, mode="saturation")
    assert k_sat.shape == (tw.size * techo.size, t1g.size * t2g.size)
    k_inv = nmr.t1t2_kernel(techo, tw, t1g, t2g, mode="inversion")
    assert not np.allclose(k_sat, k_inv)
    with pytest.raises(ValueError):
        nmr.t1t2_kernel(techo, tw, t1g, t2g, mode="bogus")


def test_fit_t1_roundtrip() -> None:
    pytest.importorskip("scipy")
    t = np.linspace(1.0, 500.0, 40)
    signal = 2.5 * (1.0 - np.exp(-t / 120.0))
    m0, t1 = nmr.fit_t1(t, signal, model="saturation")
    assert m0 == pytest.approx(2.5, rel=1e-4)
    assert t1 == pytest.approx(120.0, rel=1e-4)


# --- permeability transforms --------------------------------------------------


def _original_coates_classic_2014_06(phi, ffi, bvi, c=10.0):
    return (phi / c) ** 4 * (ffi / bvi) ** 2


def _original_timur_coates_prefactor_2022_06(phi, FFV, BFV, C=10.0, m=4.0, n=2.0):
    return C * phi**m * (FFV / BFV) ** n


def _original_sdr_2021_04(phi, t2lm_ms, a=4.0, m=4.0, n=2.0):
    return a * phi**m * t2lm_ms**n


def _original_ksdr_2015_10(phi, rho, t2lm, a=4.0, b=4.0, c=2.0):
    return a * phi**b * (rho * t2lm) ** c


def _original_timur_2025_06(phi, sw, a=4800.0, b=4.4, c=2.0):
    return a * phi**b / sw**c


def test_permeability_transforms() -> None:
    phi = np.array([0.1, 0.2, 0.3])
    assert_matches_original(
        _original_coates_classic_2014_06,
        lambda phi, ffi, bvi, c=10.0: nmr.timur_coates(phi, ffi, bvi, C=c),
        [(phi, 0.7 * phi, 0.3 * phi)],
    )
    assert_matches_original(
        _original_timur_coates_prefactor_2022_06,
        lambda phi, ffv, bfv, C=10.0, m=4.0, n=2.0: nmr.timur_coates(
            phi, ffv, bfv, C=C, m=m, n=n, form="prefactor"
        ),
        [(phi, 0.7 * phi, 0.3 * phi)],
    )
    assert_matches_original(
        _original_sdr_2021_04,
        lambda phi, t2lm, a=4.0, m=4.0, n=2.0: nmr.sdr(phi, t2lm, a=a, m=m, n=n),
        [(phi, np.array([20.0, 80.0, 200.0]))],
    )
    assert_matches_original(
        _original_ksdr_2015_10,
        lambda phi, rho, t2lm, a=4.0, b=4.0, c=2.0: nmr.sdr(phi, t2lm, a=a, m=b, n=c, rho_um_s=rho),
        [(phi, 12.0, np.array([20.0, 80.0, 200.0]))],
    )
    assert_matches_original(
        _original_timur_2025_06,
        lambda phi, sw, a=4800.0, b=4.4, c=2.0: nmr.timur(phi, sw, a=a, b=b, c=c),
        [(phi, np.array([0.2, 0.3, 0.4]))],
    )
    # classic and prefactor forms are distinct
    assert not np.allclose(
        nmr.timur_coates(0.2, 0.14, 0.06),
        nmr.timur_coates(0.2, 0.14, 0.06, form="prefactor"),
    )
    with pytest.raises(ValueError):
        nmr.timur_coates(0.2, 0.14, 0.06, form="bogus")


# --- fluid typing & hydrogen index --------------------------------------------


def _original_partition_2d_2016_02(t1t2_values, amplitudes, cutoff=2.0):
    r = np.asarray(t1t2_values, float)
    a = np.asarray(amplitudes, float)
    v_oil = float(np.sum(a[r >= cutoff]))
    return v_oil, float(a.sum()) - v_oil


def _original_hydrogen_index_2017_06(rho, n_protons, mol_weight, rho_w=1.0, n_w=2.0, m_w=18.02):
    return (rho * n_protons / mol_weight) / (rho_w * n_w / m_w)


def test_fluid_typing_and_hi() -> None:
    ratios = np.array([1.2, 1.9, 2.0, 3.5])
    np.testing.assert_allclose(nmr.t1_t2_ratio(4.0, 2.0), 2.0)
    np.testing.assert_array_equal(
        nmr.classify_t1t2(ratios, cutoff=2.0), np.array([False, False, True, True])
    )
    assert_matches_original(
        _original_partition_2d_2016_02,
        lambda r, a, cutoff=2.0: nmr.partition_t1t2_map(r, a, cutoff=cutoff),
        [(ratios, np.array([1.0, 2.0, 3.0, 4.0]))],
    )
    np.testing.assert_allclose(nmr.nmr_saturation(0.3, 0.5), 0.6, rtol=1e-12)
    # pure water HI = 1
    assert_matches_original(
        _original_hydrogen_index_2017_06,
        lambda rho, n, m: nmr.hydrogen_index(rho, n, m),
        [(1.0, 2.0, 18.02), (0.8, 1.9, 16.0)],
    )
    np.testing.assert_allclose(nmr.hydrogen_index(1.0, 2.0, 18.02), 1.0, rtol=1e-12)
    # HI porosity correction under-read recovery
    np.testing.assert_allclose(nmr.porosity_hi_correction(0.08, 0.8), 0.1, rtol=1e-12)


# --- relaxation theory --------------------------------------------------------


def _original_bpp_rates_2016_08(omega0, tau_c, dipolar_constant=1.0):
    def j(w):
        return tau_c / (1.0 + (w * tau_c) ** 2)

    inv_t1 = (3.0 / 10.0) * dipolar_constant * (j(omega0) + 4.0 * j(2.0 * omega0))
    inv_t2 = (
        (3.0 / 20.0) * dipolar_constant * (3.0 * j(0.0) + 5.0 * j(omega0) + 2.0 * j(2.0 * omega0))
    )
    return 1.0 / inv_t1, 1.0 / inv_t2


def _original_mitra_2015_02(d0, time, sv):
    return d0 * (1.0 - (4.0 / (9.0 * np.sqrt(np.pi))) * sv * np.sqrt(d0 * np.asarray(time, float)))


def test_relaxation_theory() -> None:
    np.testing.assert_allclose(
        nmr.bpp_spectral_density(1e6, 1e-6), 1e-6 / (1.0 + (1e6 * 1e-6) ** 2), rtol=1e-12
    )
    assert_matches_original(
        _original_bpp_rates_2016_08,
        lambda w0, tc, dipolar_constant=1.0: nmr.bpp_t1_t2(
            w0, tc, dipolar_constant=dipolar_constant
        ),
        [(2.0e6, 1e-7)],
    )
    # Mitra absolute D(t); normalized returns the ratio
    assert_matches_original(
        _original_mitra_2015_02,
        lambda d0, t, sv: nmr.mitra_short_time(d0, t, sv, normalized=False),
        [(2.3, np.array([0.5, 1.0, 2.0]), 0.1)],
    )
    np.testing.assert_allclose(nmr.tortuosity(2.5, 1.0), 2.5, rtol=1e-12)
