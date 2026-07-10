"""Tests for petrolib.integrity_drilling: cement-bond acoustics, casing
condition, microannulus leaks, mud gas, pressure window, mudcake -- known
values, physical sanity, and error paths."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import integrity_drilling as idl  # noqa: E402

# --- cement acoustics -----------------------------------------------------------


def test_impedance_reflection_attenuation() -> None:
    # steel: 7850 kg/m3 * 5900 m/s = 46.315 MRayl; g/cc convention agrees
    np.testing.assert_allclose(idl.acoustic_impedance(7850.0, 5900.0), 46.315)
    np.testing.assert_allclose(
        idl.acoustic_impedance(7.85, 5900.0, rho_unit="g/cc"), 46.315, rtol=1e-12
    )
    with pytest.raises(ValueError, match="rho_unit"):
        idl.acoustic_impedance(1000.0, 1500.0, rho_unit="bogus")
    # reflection: matched impedances reflect nothing, transmission is 1 - R^2
    np.testing.assert_allclose(idl.reflection_coefficient(5.0, 5.0), 0.0)
    r = float(idl.reflection_coefficient(46.3, 5.4))
    np.testing.assert_allclose(idl.transmission_energy(46.3, 5.4), 1.0 - r**2)
    # attenuation: a factor-10 drop is 20 dB; spacing normalizes to dB/m
    np.testing.assert_allclose(idl.attenuation_db(10.0, 1.0), 20.0)
    np.testing.assert_allclose(idl.attenuation_db(10.0, 1.0, 2.0), 10.0)
    np.testing.assert_allclose(idl.attenuation_coefficient(10.0, 10.0 / np.e, 1.0), 1.0)


def test_bond_index_conventions() -> None:
    # linear / attenuation: anchors from the 2019 composite-cement paper
    np.testing.assert_allclose(idl.bond_index(1.0, 1.0, 10.0, input_kind="attenuation"), 0.0)
    np.testing.assert_allclose(idl.bond_index(10.0, 1.0, 10.0, input_kind="attenuation"), 1.0)
    np.testing.assert_allclose(idl.bond_index(5.5, 1.0, 10.0, input_kind="attenuation"), 0.5)
    # linear / amplitude: amplitude falls with bond
    np.testing.assert_allclose(idl.bond_index(80.0, 80.0, 5.0), 0.0)
    np.testing.assert_allclose(idl.bond_index(5.0, 80.0, 5.0), 1.0)
    # out-of-range measurements clip
    np.testing.assert_allclose(idl.bond_index(100.0, 80.0, 5.0), 0.0)
    # log convention reproduces the microannuli-paper formula
    expected = np.clip(np.log(80.0 / 15.0) / np.log(80.0 / 2.0), 0.0, 1.0)
    np.testing.assert_array_equal(idl.bond_index(15.0, 80.0, 2.0, method="log"), expected)
    # degenerate anchors: free pipe not above bonded -> BI = 1
    np.testing.assert_allclose(idl.bond_index(5.0, 2.0, 2.0, method="log"), 1.0)
    with pytest.raises(ValueError, match="input_kind"):
        idl.bond_index(5.0, 80.0, 2.0, input_kind="bogus")
    with pytest.raises(ValueError, match="method"):
        idl.bond_index(5.0, 80.0, 2.0, method="bogus")
    # combined indicator: 0.6/0.4 weights, clipped
    np.testing.assert_allclose(idl.bond_index_combined(0.8, 0.5), 0.6 * 0.8 + 0.4 * 0.5)
    np.testing.assert_allclose(idl.bond_index_combined(2.0, 2.0), 1.0)


def test_annulus_and_cement_quality() -> None:
    assert idl.classify_annulus(0.3) == "gas"
    assert idl.classify_annulus(1.5) == "liquid"
    assert idl.classify_annulus(2.8) == "transition"
    assert idl.classify_annulus(5.0) == "cement"
    # collapsing the transition band reproduces the two-threshold classifier
    assert idl.classify_annulus(2.8, cement_min=2.6) == "cement"
    q = idl.cement_quality_score([1.0, 2.5, 4.0, 6.0])
    np.testing.assert_allclose(q[2:], 1.0)  # at/above 'good' -> 1
    np.testing.assert_allclose(q[1], 0.5)  # at 'fair' -> 0.5
    assert q[0] < 0.5
    # CO2-resistant cement has lower thresholds -> higher score at same Z
    assert float(idl.cement_quality_score(3.0, cement_type="CO2_resistant")) > float(
        idl.cement_quality_score(3.0)
    )
    np.testing.assert_array_equal(
        idl.classify_cement_from_cbl([0.10, 0.15, 0.30, 0.31]),
        ["Good", "Good", "Medium", "Poor"],
    )


# --- casing condition -----------------------------------------------------------


def test_casing_resonance_and_condition() -> None:
    # 9 mm steel wall resonates near 328 kHz; inverse round-trips
    f = float(idl.casing_resonance_frequency(0.009))
    np.testing.assert_allclose(f, 5900.0 / (2 * 0.009))
    np.testing.assert_allclose(idl.casing_thickness_from_resonance(f), 0.009)
    # S1 correction and harmonics scale linearly
    np.testing.assert_allclose(idl.casing_resonance_frequency(0.009, correction=0.95), 0.95 * f)
    np.testing.assert_allclose(idl.casing_resonance_frequency(0.009, n=2), 2 * f)
    np.testing.assert_allclose(idl.metal_loss_pct(9.0, 12.0), 25.0)
    np.testing.assert_allclose(idl.metal_loss_pct(13.0, 12.0), 0.0)  # clipped
    np.testing.assert_array_equal(
        idl.casing_condition([5.0, 15.0, 30.0, 45.0]), ["good", "fair", "poor", "critical"]
    )
    np.testing.assert_allclose(idl.remaining_life_years(11.0, 6.9, 0.1), 41.0)
    assert idl.remaining_life_years(6.0, 6.9, 0.1) == 0.0  # already below minimum
    assert idl.remaining_life_years(11.0, 6.9, 0.0) == 999.0  # no-wear sentinel
    np.testing.assert_allclose(idl.corrosion_front_depth(25.0), 12.5)
    np.testing.assert_allclose(idl.corrosion_front_depth(-5.0), 0.0)  # negative time guard


# --- microannulus leaks ---------------------------------------------------------


def test_microannulus_leak_rates() -> None:
    r, tau = 0.2445 / 2.0, 100e-6
    omega = idl.microannulus_omega(r, tau)
    assert omega > 0.0
    assert idl.microannulus_omega(r, 0.0) == 0.0
    assert idl.microannulus_omega(r, -1e-6) == 0.0
    # thicker gap leaks much faster (~cubic in aperture)
    q1 = idl.leak_rate_liquid(tau, r, 1e6, 100.0, 1e-3)
    q2 = idl.leak_rate_liquid(2 * tau, r, 1e6, 100.0, 1e-3)
    assert q2 > 4.0 * q1 > 0.0
    # a horizontal well loses the gravity back-pressure -> more flow at same dP
    q_h = idl.leak_rate_liquid(tau, r, 1e6, 100.0, 1e-3, inclination_deg=0.0)
    assert q_h < q1  # gravity term is subtracted only when cos(theta) > 0... check orientation
    # actually cos(0 deg) = 1 -> full gravity opposes: vertical (90 deg) has ~zero correction
    assert idl.leak_rate_liquid(tau, r, 1e4, 100.0, 1e-3, inclination_deg=0.0) == 0.0
    # gas: no flow without a squared-pressure difference; clamped at zero
    assert idl.leak_rate_gas(tau, r, 101325.0, 101325.0, 100.0, 1.1e-5) == 0.0
    assert idl.leak_rate_gas(tau, r, 2e6, 101325.0, 100.0, 1.1e-5) > 0.0
    assert idl.leak_rate_gas(tau, r, 2e6, 0.0, 100.0, 1.1e-5) == 0.0  # P2 guard
    # cubic law: K = rho g w^3 / (12 mu)
    np.testing.assert_allclose(
        idl.cubic_law_conductivity(1e-3), 1000.0 * 9.81 * 1e-9 / (12.0 * 1e-3)
    )


# --- mud gas --------------------------------------------------------------------


def test_haworth_pixler_classify() -> None:
    wh, bh, ch = idl.haworth_ratios(85.0, 5.0, 4.0, 3.0, 3.0)
    np.testing.assert_allclose([wh, bh, ch], [15.0, 9.0, 1.5])
    # fraction mode and degenerate inputs
    whf, _, _ = idl.haworth_ratios(85.0, 5.0, 4.0, 3.0, 3.0, percent=False)
    np.testing.assert_allclose(whf, 0.15)
    assert np.isnan(idl.haworth_ratios(0, 0, 0, 0, 0)[0])
    assert idl.haworth_ratios(100, 0, 0, 0, 0)[1] == np.inf
    px = idl.pixler_ratios(85.0, 5.0, 4.0, 3.0, 3.0)
    np.testing.assert_allclose(px["C1/C2"], 17.0)
    np.testing.assert_allclose(px["bernard"], 85.0 / 9.0)
    assert idl.pixler_ratios(85.0, 0.0, 4.0, 3.0, 3.0)["C1/C2"] == np.inf
    # 4-class bands
    assert idl.classify_fluid_haworth(0.2, 500.0) == "dry gas"
    assert idl.classify_fluid_haworth(10.0, 15.0) == "gas"
    assert idl.classify_fluid_haworth(10.0, 5.0) == "gas-condensate"
    assert idl.classify_fluid_haworth(30.0, 3.0) == "oil"
    assert idl.classify_fluid_haworth(50.0, 1.0) == "residual oil"
    # 8-class scheme needs ch and covers the extended labels
    assert idl.classify_fluid_haworth(10.0, 20.0, 0.7, n_classes=8) == "gas condensate"
    assert idl.classify_fluid_haworth(3.0, 200.0, 0.1, n_classes=8) == "dry gas"
    assert idl.classify_fluid_haworth(float("nan"), 1.0, 1.0, n_classes=8) == "no gas"
    with pytest.raises(ValueError, match="n_classes"):
        idl.classify_fluid_haworth(10.0, 5.0, n_classes=5)


def test_normalize_gas_modes() -> None:
    # metric: doubling both gas and ROP leaves the index unchanged
    g1 = idl.normalize_gas(100.0, 20.0, 2000.0, 8.5)
    g2 = idl.normalize_gas(200.0, 40.0, 2000.0, 8.5)
    np.testing.assert_allclose(g1, g2)
    # field: at reference conditions the reading is unchanged
    np.testing.assert_allclose(
        idl.normalize_gas(50.0, 30.0, 500.0, 8.5, mud_weight=10.0, units="field"), 50.0
    )
    # heavier mud suppresses the reading quadratically -> normalization boosts it
    boosted = idl.normalize_gas(50.0, 30.0, 500.0, 8.5, mud_weight=14.0, units="field")
    assert float(boosted) < 50.0  # dividing by (14/10)^2 > 1
    with pytest.raises(ValueError, match="mud_weight"):
        idl.normalize_gas(50.0, 30.0, 500.0, 8.5, units="field")
    with pytest.raises(ValueError, match="units"):
        idl.normalize_gas(50.0, 30.0, 500.0, 8.5, units="bogus")


# --- pressures / drilling window ------------------------------------------------


def test_pressures_and_window() -> None:
    np.testing.assert_allclose(idl.hydrostatic_pressure(3000.0), 1030.0 * 9.80665 * 3000.0)
    np.testing.assert_allclose(
        idl.hydrostatic_pressure_psi(5000.0, sg=1.025), 0.433 * 1.025 * 5000.0
    )
    np.testing.assert_allclose(
        idl.hydrostatic_pressure_bar(3000.0, sg=1.2), 0.0980665 * 1.2 * 3000.0
    )
    np.testing.assert_allclose(
        idl.overburden_pressure(2000.0, 2300.0, water_depth_m=1500.0),
        9.80665 * (1030.0 * 1500.0 + 2300.0 * 2000.0),
    )
    np.testing.assert_allclose(
        idl.overburden_pressure_psi(5000.0, 10000.0), 0.433 * (1.025 * 5000.0 + 2.3 * 10000.0)
    )
    # Eaton: normal conditions (ratio 1) return hydrostatic in both orientations
    np.testing.assert_allclose(
        idl.eaton_pore_pressure(10000.0, 8000.0, 1.0, 1.0, log_type="resistivity"), 8000.0
    )
    np.testing.assert_allclose(idl.eaton_pore_pressure(1.0, 0.465, 100.0, 100.0), 0.465)
    # overpressure: low resistivity or slow sonic pulls Pp above hydrostatic
    assert (
        float(
            idl.eaton_pore_pressure(10000.0, 8000.0, 0.5, 1.0, exponent=1.2, log_type="resistivity")
        )
        > 8000.0
    )
    assert (
        float(idl.eaton_pore_pressure(1.0, 0.465, 130.0, 100.0, clip_ratio=(0.01, 100.0))) > 0.465
    )
    with pytest.raises(ValueError, match="log_type"):
        idl.eaton_pore_pressure(1.0, 0.5, 1.0, 1.0, log_type="bogus")
    # Bowers: faster rock -> higher effective stress -> lower pore pressure
    pp_slow = float(idl.bowers_pore_pressure(7000.0, 8000.0))
    pp_fast = float(idl.bowers_pore_pressure(11000.0, 8000.0))
    assert pp_fast < pp_slow < 8000.0
    # unloading branch follows the article-verbatim flatter curve
    sigma_unl = 5000.0 * ((9000.0 - 5000.0) / (10.0 * 5000.0**0.7)) ** 3.0
    np.testing.assert_allclose(
        idl.bowers_pore_pressure(9000.0, 8000.0, unloading=True), 8000.0 - sigma_unl
    )
    np.testing.assert_allclose(idl.drilling_window_margin(9000.0, 10500.0), 1500.0)
    assert idl.within_drilling_window(9800.0, 9000.0, 10500.0)
    assert not idl.within_drilling_window(8900.0, 9000.0, 10500.0)
    assert not idl.within_drilling_window(9000.0, 9000.0, 10500.0)  # strict


# --- mudcake --------------------------------------------------------------------


def test_mudcake_models() -> None:
    np.testing.assert_allclose(
        idl.mudcake_thickness(3600.0, rate_const=1e-4, model="sqrt_k"), 1e-4 * 60.0
    )
    # dewan: sqrt-t growth with the closed-form prefactor
    h1 = float(
        idl.mudcake_thickness(900.0, k_mc_m2=0.5e-15, dp_pa=3.5e6, mu_pa_s=0.05, solids_ratio=0.6)
    )
    h4 = float(
        idl.mudcake_thickness(3600.0, k_mc_m2=0.5e-15, dp_pa=3.5e6, mu_pa_s=0.05, solids_ratio=0.6)
    )
    np.testing.assert_allclose(h4, 2.0 * h1)
    np.testing.assert_allclose(h4, np.sqrt(2.0 * 0.5e-15 * 3.5e6 * 3600.0 / (0.05 * 0.6)))
    # chin: monotone growth from the 10-micron seed
    t = np.linspace(0.001, 10.0, 50)
    h = idl.mudcake_thickness(
        t, k_mc_m2=1e-4, dp_pa=1000.0, mu_pa_s=1e-3, solids_ratio=0.05, model="chin_ode"
    )
    assert h[0] == 1.0e-5
    assert np.all(np.diff(h) > 0)
    with pytest.raises(ValueError, match="model"):
        idl.mudcake_thickness(10.0, model="bogus")
