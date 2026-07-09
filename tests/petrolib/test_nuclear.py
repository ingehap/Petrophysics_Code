"""Tests for petrolib.nuclear: golden values, round trips, dispatch, and error
paths across capture cross-section (Sigma/PNC), attenuation, density logging,
gamma ray, neutron / hydrogen index, C/O + spectroscopy, and counting / decay."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import nuclear as nuc  # noqa: E402

# --- capture cross-section (Sigma / PNC) --------------------------------------


def test_sigma_forward_inverse_round_trip() -> None:
    phi, sw = 0.22, 0.6
    sig = nuc.sigma_forward(phi, sw, sigma_ma=8.0, sigma_w=55.0, sigma_hc=21.0)
    np.testing.assert_allclose(
        nuc.sw_from_sigma(sig, phi, sigma_ma=8.0, sigma_w=55.0, sigma_hc=21.0, clip=None),
        sw,
        rtol=1e-12,
    )
    # shale term round-trips too
    sig2 = nuc.sigma_forward(
        0.2, 0.5, sigma_ma=8.0, sigma_w=55.0, sigma_hc=21.0, vsh=0.15, sigma_sh=30.0
    )
    np.testing.assert_allclose(
        nuc.sw_from_sigma(
            sig2, 0.2, sigma_ma=8.0, sigma_w=55.0, sigma_hc=21.0, vsh=0.15, sigma_sh=30.0, clip=None
        ),
        0.5,
        rtol=1e-12,
    )


def test_sigma_3phase_sensitivity_and_clip() -> None:
    # fully water-saturated 3-phase reduces to the matrix+water mix
    s3 = nuc.sigma_forward_3phase(0.2, 0.0, 0.0, 1.0, sigma_w=80.0, sigma_ma=10.0)
    np.testing.assert_allclose(s3, 0.8 * 10.0 + 0.2 * 80.0)
    np.testing.assert_allclose(nuc.sigma_sensitivity(0.2, 55.0, 21.0), 0.2 * 34.0)
    # an out-of-range Sigma clips Sw to [0, 1]
    np.testing.assert_allclose(
        nuc.sw_from_sigma(1000.0, 0.2, sigma_ma=8.0, sigma_w=55.0, sigma_hc=21.0), 1.0
    )


def test_sigma_salinity_number_density_and_tau() -> None:
    np.testing.assert_allclose(nuc.sigma_w_from_salinity(150000.0, model="linear450k"), 88.0)
    np.testing.assert_allclose(
        nuc.sigma_w_from_salinity(150000.0), (22.0 + 750.0 * 0.15), rtol=1e-12
    )
    with pytest.raises(ValueError, match="model"):
        nuc.sigma_w_from_salinity(1e5, model="bogus")
    np.testing.assert_allclose(nuc.number_density(1.0, 0.111, 1.008), 1.0 * nuc.NA * 0.111 / 1.008)
    np.testing.assert_allclose(nuc.sigma_from_tau(nuc.tau_from_sigma(30.0)), 30.0, rtol=1e-12)
    np.testing.assert_allclose(nuc.tau_from_sigma(20.0), 4550.0 / 20.0)
    with pytest.raises(ValueError, match="units"):
        nuc.macroscopic_sigma([1e22], [0.3], units="bad")


def test_pnc_decay_fit_round_trip() -> None:
    t = np.linspace(50.0, 500.0, 40)
    counts = nuc.pnc_decay(t, 1e5, 25.0, background=0.0)
    np.testing.assert_allclose(nuc.sigma_from_decay_fit(t, counts), 25.0, rtol=1e-6)
    # background floors the late-time decay
    assert float(nuc.pnc_decay(1e9, 1e5, 25.0, background=7.0)) == pytest.approx(7.0)


# --- attenuation --------------------------------------------------------------


def test_attenuation_round_trip_and_map() -> None:
    i = nuc.beer_lambert(100.0, 0.5, 2.0)
    np.testing.assert_allclose(i, 100.0 * np.exp(-1.0))
    np.testing.assert_allclose(nuc.mu_from_intensity(100.0, i, 2.0), 0.5, rtol=1e-12)
    np.testing.assert_allclose(nuc.attenuation_map(np.exp(-2.0), 1.0), 2.0)
    # gamma count / density are inverses
    np.testing.assert_allclose(nuc.density_from_count(nuc.gamma_count(2.4)), 2.4, rtol=1e-12)


# --- density logging ----------------------------------------------------------


def test_density_logging() -> None:
    rho_b, drho = nuc.spine_ribs(2.45, 2.40, rib_coeffs=(0.0, 1.0, 0.0))
    np.testing.assert_allclose([rho_b, drho], [2.50, 0.05])
    np.testing.assert_allclose(nuc.rhoe_from_rhob(nuc.rhob_from_rhoe(2.6)), 2.6, rtol=1e-12)
    np.testing.assert_allclose(
        nuc.electron_density_index(14.0, 28.09, 2.65), 2.0 * 14.0 / 28.09 * 2.65
    )
    np.testing.assert_allclose(
        nuc.dual_detector_density(200.0, 100.0, a=2.0, b=0.5), 2.0 + 0.5 * np.log(2.0)
    )


# --- gamma ray ----------------------------------------------------------------


def test_gamma_ray_api() -> None:
    np.testing.assert_allclose(nuc.gr_api(2.0, 5.0, 12.0), 16.0 * 2 + 8.0 * 5 + 4.0 * 12)
    np.testing.assert_allclose(nuc.cgr_api(2.0, 12.0), 16.0 * 2 + 4.0 * 12)
    np.testing.assert_allclose(
        nuc.gr_api(1.0, 1.0, 1.0, coeff=(16.32, 8.09, 3.93)), 16.32 + 8.09 + 3.93
    )


# --- neutron / hydrogen index -------------------------------------------------


def test_hydrogen_index_and_neutron() -> None:
    assert nuc.hydrogen_index_fluid("water") == 1.0
    assert nuc.hydrogen_index_fluid("gas", rho_gas=0.25) == 0.25
    with pytest.raises(ValueError, match="fluid"):
        nuc.hydrogen_index_fluid("plasma")
    np.testing.assert_allclose(
        nuc.phi_from_hi(nuc.hi_mix(0.3, hi_fluid=1.0, hi_matrix=0.1), hi_matrix=0.1),
        0.3,
        rtol=1e-12,
    )
    np.testing.assert_allclose(nuc.phi_hi_correction(0.18, 0.9), 0.2)
    # hydrogen (A=1) is the moderation limit: alpha=0, xi=1
    np.testing.assert_allclose(nuc.collision_parameter(1.0), 0.0)
    np.testing.assert_allclose(nuc.average_lethargy_gain(1.0), 1.0)
    np.testing.assert_allclose(
        nuc.moderating_power(12.0, 5.0), nuc.average_lethargy_gain(12.0) * 5.0
    )
    ls = nuc.slowing_down_length_empirical(0.2, 4.5)
    np.testing.assert_allclose(nuc.phi_from_ls(ls, 4.5), 0.2, rtol=1e-12)
    np.testing.assert_allclose(nuc.phi_n_from_lm(6.0, lithology="limestone"), 1.0)
    with pytest.raises(ValueError, match="lithology"):
        nuc.phi_n_from_lm(10.0, lithology="granite")


def test_transport_length_and_cnp() -> None:
    # two equal-length components in equal fractions -> that length
    np.testing.assert_allclose(nuc.transport_length_mix([0.5, 0.5], [10.0, 10.0]), 10.0)
    # compensated neutron porosity is the a + b*lnR + c*lnR^2 polynomial
    r = 1.2
    np.testing.assert_allclose(
        nuc.compensated_neutron_porosity(1200.0, 1000.0),
        -30.0 * np.log(r) + 45.0 * np.log(r) ** 2,
        rtol=1e-12,
    )


# --- C/O and spectroscopy -----------------------------------------------------


def test_carbon_oxygen_and_yields() -> None:
    np.testing.assert_allclose(nuc.co_ratio(0.6, 0.8), 0.75)
    np.testing.assert_allclose(nuc.so_from_co(0.5, 0.3, 0.7), 0.5)
    # endpoint interpolation clips beyond the oil endpoint
    np.testing.assert_allclose(nuc.so_from_co(0.9, 0.3, 0.7), 1.0)
    np.testing.assert_allclose(
        nuc.weights_to_yields(2.0, 0.5, nuc.yields_to_weights(2.0, 0.5, 0.3)), 0.3, rtol=1e-12
    )
    np.testing.assert_allclose(nuc.toc_from_yield(0.05, 12.0), 0.6)
    # C/O forward rises with oil saturation
    assert float(nuc.co_forward_3phase(0.2, 0.8, 0.0, 0.2)) > float(
        nuc.co_forward_3phase(0.2, 0.2, 0.0, 0.8)
    )


# --- counting statistics and decay --------------------------------------------


def test_counting_and_decay() -> None:
    np.testing.assert_allclose(nuc.counting_precision(10000.0), 0.01)
    np.testing.assert_allclose(nuc.counting_sigma(50.0, 100.0), 5.0)
    np.testing.assert_allclose(nuc.decay_constant(10.0), np.log(2.0) / 10.0)
    # one half-life halves the population
    np.testing.assert_allclose(nuc.radioactive_decay(100.0, 10.0, 10.0), 50.0, rtol=1e-12)
    np.testing.assert_allclose(nuc.activity(100.0, 10.0), np.log(2.0) / 10.0 * 100.0)
