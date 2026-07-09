"""Tests for petrolib.geochem_fluids: golden values, round trips, dispatch, and
error paths across brine / asphaltene / mudgas / adsorption / pvt / gradients /
solubility / contamination / core_geochem."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib.geochem_fluids import (  # noqa: E402
    adsorption,
    asphaltene,
    brine,
    contamination,
    core_geochem,
    gradients,
    mudgas,
    pvt,
    solubility,
)

# --- brine --------------------------------------------------------------------


def test_bateman_konen_round_trip() -> None:
    r75 = brine.rw75_from_salinity(50000.0)
    np.testing.assert_allclose(r75, 0.0123 + 3647.5 / 50000.0**0.955, rtol=1e-12)
    np.testing.assert_allclose(brine.salinity_from_rw75(r75), 50000.0, rtol=1e-9)


def test_rw_temperature_round_trip_and_arps() -> None:
    rw = brine.rw_from_salinity(50000.0, 200.0, unit="F")
    np.testing.assert_allclose(brine.salinity_from_rw(rw, 200.0, unit="F"), 50000.0, rtol=1e-9)
    # Arps constant differs by unit
    np.testing.assert_allclose(brine.arps_correct(0.1, 75.0, 200.0, unit="F"), 0.1 * 81.77 / 206.77)
    np.testing.assert_allclose(brine.arps_correct(0.1, 24.0, 90.0, unit="C"), 0.1 * 45.5 / 111.5)
    with pytest.raises(ValueError, match="unit"):
        brine.arps_correct(0.1, 75.0, 200.0, unit="K")


def test_brine_sigma_and_density_and_meq() -> None:
    # fresh water at 75 degF baseline is 22 c.u.
    np.testing.assert_allclose(brine.sigma_w_from_salinity(0.0, temp_c=75.0), 22.0)
    np.testing.assert_allclose(brine.nacl_meq_per_liter(58440.0), 1000.0, rtol=1e-12)
    rho = brine.brine_density_bw92(100000.0, 25.0)
    assert 1000.0 < float(rho) < 1150.0


# --- asphaltene ---------------------------------------------------------------


def test_molar_volume_diameter_round_trip() -> None:
    va = asphaltene.molar_volume_from_diameter(2.0e-9)
    np.testing.assert_allclose(asphaltene.diameter_from_molar_volume(va), 2.0e-9, rtol=1e-12)


def test_fhz_ratio_and_invert() -> None:
    va, drho, t = 2.5e-3, 400.0, 380.0
    ratio = asphaltene.fhz_ratio(100.0, va, drho, t)
    # deeper (positive dz) enriches asphaltene -> ratio > 1
    assert float(ratio) > 1.0
    np.testing.assert_allclose(asphaltene.fhz_ratio(0.0, va, drho, t), 1.0, rtol=1e-12)
    va_rec = asphaltene.fhz_invert_molar_volume(1.0, 0.0, float(ratio), 100.0, drho, t)
    np.testing.assert_allclose(va_rec, va, rtol=1e-9)


def test_nearest_yen_mullins() -> None:
    name, ref, agrees = asphaltene.nearest_yen_mullins(2.1e-9)
    assert name == "nanoaggregate" and ref == 2.0e-9 and agrees
    _, _, agrees_far = asphaltene.nearest_yen_mullins(3.5e-9, rtol=0.1)
    assert not agrees_far


# --- mudgas -------------------------------------------------------------------


def test_mudgas_ratios_and_percent_flag() -> None:
    c = (80.0, 10.0, 5.0, 3.0, 2.0)
    np.testing.assert_allclose(mudgas.wetness_ratio(*c), 20.0)
    np.testing.assert_allclose(mudgas.wetness_ratio(*c, percent=False), 0.20)
    np.testing.assert_allclose(mudgas.balance_ratio(*c), 100.0 / 10.0)
    np.testing.assert_allclose(mudgas.character_ratio(5.0, 3.0, 2.0), 1.0)
    np.testing.assert_allclose(mudgas.bernard_ratio(80.0, 10.0, 5.0), 80.0 / 15.0)
    assert mudgas.pixler_ratios(*c)["C1/C2"] == 8.0


def test_mudgas_zero_denominator() -> None:
    assert np.isinf(mudgas.balance_ratio(80.0, 10.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(mudgas.character_ratio(0.0, 3.0, 2.0), 0.0)


def test_mudgas_normalize_and_classify() -> None:
    np.testing.assert_allclose(mudgas.normalize_composition([1.0, 2.0, 3.0, 4.0]).sum(), 1.0)
    assert mudgas.classify_fluid_gor(300.0) == "volatile oil"
    assert mudgas.classify_fluid_gor(20000.0) == "dry gas"
    assert mudgas.classify_fluid_wetness(0.2, 50.0) == "dry gas"
    assert mudgas.classify_fluid_wetness(2.0, 50.0) == "gas"


# --- adsorption ---------------------------------------------------------------


def test_langmuir_and_free_gas() -> None:
    np.testing.assert_allclose(adsorption.langmuir(300.0, 10.0, 300.0), 5.0, rtol=1e-12)
    np.testing.assert_allclose(adsorption.langmuir(300.0, 10.0, 300.0, rho_b=2.5), 12.5)
    np.testing.assert_allclose(adsorption.free_gas(0.1, 0.3, 0.005), 0.1 * 0.7 / 0.005)
    np.testing.assert_allclose(
        adsorption.gas_in_place(1e6, 30.0, 0.1, 0.7, 0.005), 1e6 * 30 * 0.1 * 0.7 / 0.005
    )


def test_bet_isotherm_and_fit() -> None:
    # generate a BET curve then recover Vm, C
    x = np.array([0.05, 0.1, 0.15, 0.2, 0.25])
    vm_true, c_true = 1.2, 15.0
    v = adsorption.bet_isotherm(x, vm_true, c_true)
    vm, c, ssa = adsorption.bet_fit(x, v)
    np.testing.assert_allclose(vm, vm_true, rtol=1e-6)
    np.testing.assert_allclose(c, c_true, rtol=1e-6)
    assert ssa > 0


# --- pvt ----------------------------------------------------------------------


def test_pvt_zfactor_and_density() -> None:
    ppr, tpr = pvt.pseudo_reduced(2000.0, 560.0, 668.0, 343.0)
    np.testing.assert_allclose(ppr, 2000.0 / 668.0)
    z = pvt.z_beggs_brill(ppr, tpr)
    assert 0.7 < float(z) < 1.05
    rho = pvt.gas_density(20e6, 350.0, z=0.9)
    np.testing.assert_allclose(pvt.pressure_from_gas_density(rho, 350.0, z=0.9), 20e6, rtol=1e-9)


def test_pvt_peng_robinson_and_flash() -> None:
    zv = pvt.z_peng_robinson(5e6, 300.0, 190.6, 4.6e6, 0.011, phase="vapor")
    zl = pvt.z_peng_robinson(5e6, 300.0, 190.6, 4.6e6, 0.011, phase="liquid")
    assert zv >= zl > 0
    with pytest.raises(ValueError, match="phase"):
        pvt.z_peng_robinson(5e6, 300.0, 190.6, 4.6e6, 0.011, phase="solid")
    np.testing.assert_allclose(pvt.gas_gravity(0.028964), 1.0, rtol=1e-12)
    np.testing.assert_allclose(pvt.mixture_mw([0.5, 0.5], [0.016, 0.030]), 0.023)
    # Rachford-Rice residual is ~0 at the returned beta
    z, k = np.array([0.5, 0.3, 0.2]), np.array([3.0, 1.0, 0.2])
    beta = pvt.rachford_rice(z, k)
    assert abs(float(np.sum(z * (k - 1.0) / (1.0 + beta * (k - 1.0))))) < 1e-9


# --- gradients ----------------------------------------------------------------


def test_gradient_fit_density_contact() -> None:
    slope, p0 = gradients.fit_pressure_gradient([1000.0, 1100.0, 1200.0], [1.0e7, 1.1e7, 1.2e7])
    np.testing.assert_allclose(slope, 1.0e4, rtol=1e-6)
    np.testing.assert_allclose(gradients.density_from_gradient(slope), 1.0e4 / 9.80665, rtol=1e-9)
    np.testing.assert_allclose(
        gradients.density_from_gradient(0.45, p_unit="psi/ft"), 0.45 * 6894.757 / 0.3048 / 9.80665
    )
    with pytest.raises(ValueError, match="p_unit"):
        gradients.density_from_gradient(1.0, p_unit="kPa/m")
    # two gradients intersecting at a known depth (hand-solved to 980 m)
    contact = gradients.fluid_contact(
        [900.0, 1000.0], [8.0e6, 8.7e6], [1100.0, 1200.0], [1.0e7, 1.12e7]
    )
    np.testing.assert_allclose(contact, 980.0, rtol=1e-9)


# --- solubility ---------------------------------------------------------------


def test_solubility_co2_and_setschenow() -> None:
    # salting-out reduces solubility (gamma > 1 for m > 0)
    fresh = solubility.co2_solubility_brine(10.0, 350.0, 0.0)
    saline = solubility.co2_solubility_brine(10.0, 350.0, 2.0)
    assert float(saline) < float(fresh)
    np.testing.assert_allclose(solubility.setschenow_factor(0.0, 350.0), 1.0)
    np.testing.assert_allclose(
        solubility.henry_solubility_ln(5.0, 350.0, 0.1, 0.85, -5000.0),
        0.1 + 0.85 * np.log(5.0) + 5000.0 / (8.314 * 350.0),
        rtol=1e-12,
    )


# --- contamination ------------------------------------------------------------


def test_contamination_round_trips() -> None:
    p = contamination.mix_linear(1.0, 5.0, 0.3)
    np.testing.assert_allclose(p, 0.7 * 1.0 + 0.3 * 5.0)
    np.testing.assert_allclose(contamination.contamination_fraction(p, 1.0, 5.0), 0.3, rtol=1e-12)
    v = contamination.volume_to_target(0.5, 100.0, 0.05)
    np.testing.assert_allclose(contamination.cleanup_powerlaw(v, 0.5, 100.0), 0.05, rtol=1e-9)


# --- core_geochem -------------------------------------------------------------


def test_dean_stark_oxide_osi() -> None:
    phi, sw, s_hc = core_geochem.dean_stark(3.0, 1.0, v_bulk=20.0)
    np.testing.assert_allclose([phi, sw, s_hc], [0.2, 0.75, 0.25])
    # no bulk volume -> phi is NaN, saturations still valid
    phi2, sw2, _ = core_geochem.dean_stark(3.0, 1.0)
    assert np.isnan(phi2) and np.isclose(sw2, 0.75)
    closed, factor = core_geochem.oxide_closure([2.0, 1.0, 1.0])
    np.testing.assert_allclose(closed.sum(), 1.0)
    np.testing.assert_allclose(factor, 0.25)
    np.testing.assert_allclose(core_geochem.osi(2.0, 4.0), 50.0)
