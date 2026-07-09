"""Tests for petrolib.saturation_resistivity: golden values, round-trip
properties, and shadow equivalence against verbatim article bodies
(LIBRARY_MERGE_PLAN.md sections 7-9)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import saturation_resistivity as sat  # noqa: E402
from petrolib.testing import assert_matches_original  # noqa: E402

RNG = np.random.default_rng(20260710)

PHI = RNG.uniform(0.08, 0.32, size=40)
SW = RNG.uniform(0.15, 1.0, size=40)
RW = 0.05


# --------------------------------------------------------------------------
# Archie family: golden values and round trips
# --------------------------------------------------------------------------


def test_formation_factor_golden() -> None:
    assert sat.formation_factor(0.25) == pytest.approx(16.0)
    assert sat.formation_factor(0.25, a=0.81, m=2.15) == pytest.approx(0.81 / 0.25**2.15)


def test_archie_round_trip_sw_rt() -> None:
    rt = sat.archie_rt(SW, RW, phi=PHI, a=0.9, m=1.9, n=1.8)
    sw_back = sat.archie_sw(rt, RW, phi=PHI, a=0.9, m=1.9, n=1.8)
    np.testing.assert_allclose(sw_back, SW, rtol=1e-12)


def test_archie_sw_golden_and_clip() -> None:
    # F = 16, Rt = 40, Rw = 0.05: Sw = sqrt(16*0.05/40) = 0.1414...
    assert sat.archie_sw(40.0, 0.05, phi=0.25) == pytest.approx(np.sqrt(0.02))
    # a wet zone measurement below Ro gives Sw > 1 unless clipped explicitly
    unclipped = sat.archie_sw(0.5, 0.05, phi=0.25)
    assert unclipped > 1.0
    assert sat.archie_sw(0.5, 0.05, phi=0.25, clip=(0.0, 1.0)) == 1.0


def test_archie_conductivity_consistent_with_rt() -> None:
    ct = sat.archie_conductivity(SW, 1.0 / RW, phi=PHI)
    rt = sat.archie_rt(SW, RW, phi=PHI)
    np.testing.assert_allclose(ct, 1.0 / rt, rtol=1e-12)


def test_resistivity_index_round_trip() -> None:
    ri = sat.resistivity_index_from_sw(SW, n=2.28)
    np.testing.assert_allclose(sat.sw_from_resistivity_index(ri, n=2.28), SW, rtol=1e-12)
    assert sat.resistivity_index(40.0, 10.0) == pytest.approx(4.0)
    # generalized I = b*Sw^-n
    ri_b = sat.resistivity_index_from_sw(0.5, n=2.0, b=1.1)
    assert ri_b == pytest.approx(1.1 * 4.0)


def test_exponent_fits_recover_truth() -> None:
    ff = sat.formation_factor(PHI, a=0.81, m=2.15)
    m_fit, a_fit = sat.fit_cementation_exponent(PHI, ff)
    assert m_fit == pytest.approx(2.15)
    assert a_fit == pytest.approx(0.81)
    ri = sat.resistivity_index_from_sw(SW, n=1.99)
    assert sat.fit_saturation_exponent(SW, ri) == pytest.approx(1.99)
    np.testing.assert_allclose(
        sat.cementation_exponent_at_point(PHI, sat.formation_factor(PHI, m=2.0)),
        2.0,
        rtol=1e-12,
    )


def test_quicklook_identities() -> None:
    np.testing.assert_allclose(sat.bulk_volume_water(PHI, SW), PHI * SW)
    # Rwa recovers Rw in a wet zone
    ro = sat.archie_rt(1.0, RW, phi=PHI)
    np.testing.assert_allclose(sat.apparent_water_resistivity(ro, PHI), RW, rtol=1e-12)


# --------------------------------------------------------------------------
# Shaly-sand models: limits and inversions
# --------------------------------------------------------------------------


def test_waxman_smits_reduces_to_archie_at_zero_qv() -> None:
    ct_ws = sat.waxman_smits_conductivity(SW, cw=20.0, qv=0.0, b=0.4, phi=PHI)
    ct_archie = sat.archie_conductivity(SW, 20.0, phi=PHI)
    np.testing.assert_allclose(ct_ws, ct_archie, rtol=1e-15)


def test_waxman_smits_sw_inverts_forward() -> None:
    ct = sat.waxman_smits_conductivity(SW, cw=20.0, qv=0.3, b=0.45, phi=PHI)
    sw_back = sat.waxman_smits_sw(ct, cw=20.0, qv=0.3, b=0.45, phi=PHI)
    np.testing.assert_allclose(sw_back, SW, atol=1e-9)
    # shaly term raises conductivity, so WS Sw < Archie Sw for the same Ct
    sw_archie = (ct / (20.0 * PHI**2.0)) ** 0.5
    assert np.all(sw_back <= sw_archie + 1e-12)


def test_dual_water_reduces_to_archie_and_inverts() -> None:
    # swb = 0 (or cb == cw) collapses to Archie
    ct_dw = sat.dual_water_conductivity(SW, cw=20.0, cb=8.0, swb=0.0, phi=PHI)
    np.testing.assert_allclose(ct_dw, sat.archie_conductivity(SW, 20.0, phi=PHI), rtol=1e-15)
    ct = sat.dual_water_conductivity(SW, cw=20.0, cb=8.0, swb=0.2, phi=PHI)
    sw_back = sat.dual_water_sw(ct, cw=20.0, cb=8.0, swb=0.2, phi=PHI)
    np.testing.assert_allclose(sw_back, SW, atol=1e-9)


def test_dual_water_qv_form_matches_bound_water_form_at_sw1() -> None:
    # both forms give the same wet-rock conductivity when parameterized
    # consistently: Cwf = (1 - alpha_vqh*Qv)*Cw, excess = beta*Qv
    qv, alpha_vqh, beta = 0.3, 0.28, 2.0
    cw = 20.0
    ct_qv = sat.dual_water_conductivity_qv(
        1.0, cw=cw, qv=qv, alpha_vqh=alpha_vqh, beta=beta, phi=PHI, m0=2.0
    )
    # bound-water equivalent at Sw=1: Ct = phi^m * (Cwf + beta*Qv)
    expected = PHI**2 * ((1.0 - alpha_vqh * qv) * cw + beta * qv)
    np.testing.assert_allclose(ct_qv, expected, rtol=1e-12)


def test_simandoux_closed_form_and_archie_limit() -> None:
    # vsh = 0 reduces exactly to Archie with n = 2
    sw_clean = sat.simandoux_sw(40.0, RW, phi=0.25, vsh=0.0, rsh=2.0)
    assert sw_clean == pytest.approx(float(sat.archie_sw(40.0, RW, phi=0.25)))
    # the returned Sw satisfies the Simandoux equation
    sw = sat.simandoux_sw(20.0, RW, phi=0.22, vsh=0.3, rsh=1.5)
    lhs = 1.0 / 20.0
    rhs = sw**2 * 0.22**2 / RW + 0.3 * sw / 1.5
    assert lhs == pytest.approx(rhs, rel=1e-12)
    # shale conduction lowers Sw versus clean Archie for the same Rt
    assert sw < float(sat.archie_sw(20.0, RW, phi=0.22))


def test_indonesia_closed_form_and_archie_limit() -> None:
    sw_clean = sat.indonesia_sw(40.0, RW, phi=0.25, vcl=0.0, rcl=2.0)
    assert sw_clean == pytest.approx(float(sat.archie_sw(40.0, RW, phi=0.25)))
    sw = sat.indonesia_sw(20.0, RW, phi=0.22, vcl=0.3, rcl=1.5, n=2.0)
    lhs = 1.0 / np.sqrt(20.0)
    rhs = (0.3 ** (1.0 - 0.15) / np.sqrt(1.5) + np.sqrt(0.22**2 / RW)) * sw
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_qv_helpers_round_trip_and_juhasz() -> None:
    qv = sat.qv_from_cec(0.05, rho_grain=2.65, phi=0.2)
    assert qv == pytest.approx(0.05 * 2.65 * 0.8 / 0.2)
    assert sat.cec_from_qv(qv, rho_grain=2.65, phi=0.2) == pytest.approx(0.05)
    assert sat.qv_juhasz(0.2, rho_clay=2.78, cec_clay=0.1, phit=0.25) == pytest.approx(
        0.2 * 2.78 * 0.1 / 0.25
    )


# --------------------------------------------------------------------------
# Shadow equivalence against verbatim article bodies
# --------------------------------------------------------------------------


def _original_archie_sw_2024_10(Rt, Rw, phi, a=1.0, m=2.0, n=2.0):
    # src2024_10/water_saturation_equations.py (clips to [0, 1])
    Rt = np.asarray(Rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    F = a / (phi**m)
    Sw = (F * Rw / Rt) ** (1.0 / n)
    return np.clip(Sw, 0, 1)


def _original_ws_conductivity_2014_12(sw, cw, phit, m, n, b, qv):
    # src2014_12/article1_shaly_sand_dry_clay.py
    sw = np.asarray(sw, float)
    return sw**n * cw * phit**m + b * qv * phit**m * sw ** (n - 1)


def _original_dual_water_2014_12(sw, cw, cb, sb, phit, m, n):
    # src2014_12/article1_shaly_sand_dry_clay.py (bound-water form)
    sw = np.asarray(sw, float)
    return sw**n * phit**m * cw + sw ** (n - 1) * sb * phit**m * (cb - cw)


def _original_dual_water_qv_2014_02(phi, sw, m0, n, cw, qv, alpha, vqh, beta):
    # src2014_02/article1_dualwater_dielectric_nmr.py (Qv form; its
    # effective_water_conductivity is (1 - alpha*vqh*qv)*cw inlined here)
    sw = np.asarray(sw, float)
    cwf = (1.0 - alpha * vqh * qv) * cw
    return phi**m0 * (sw**n * cwf + sw ** (n - 1) * beta * qv)


def _original_qv_from_cec_2014_02(cec, rho_grain, phi):
    return cec * rho_grain * (1.0 - phi) / phi


def _original_juhasz_qv_2014_12(vcldry, rho_cldry, cec_cl, phit):
    return vcldry * rho_cldry * cec_cl / phit


def _original_fit_cementation_2014_02(phi, frf):
    # src2014_02/article2_pc_resistivity_index_carbonate.py
    x = np.log10(np.asarray(phi, float))
    y = np.log10(np.asarray(frf, float))
    slope, intercept = np.polyfit(x, y, 1)
    return -slope, 10.0**intercept


def _original_fit_saturation_2014_02(sw, ri):
    x = np.log10(np.asarray(sw, float))
    y = np.log10(np.asarray(ri, float))
    slope, _ = np.polyfit(x, y, 1)
    return -slope


def test_equivalence_archie_2024_10() -> None:
    rt = sat.archie_rt(SW, RW, phi=PHI)
    assert_matches_original(
        _original_archie_sw_2024_10,
        lambda Rt, Rw, phi: sat.archie_sw(Rt, Rw, phi=phi, clip=(0.0, 1.0)),
        [(rt, RW, PHI)],
    )


def test_equivalence_waxman_smits_2014_12() -> None:
    assert_matches_original(
        _original_ws_conductivity_2014_12,
        lambda sw, cw, phit, m, n, b, qv: sat.waxman_smits_conductivity(
            sw, cw=cw, qv=qv, b=b, phi=phit, m_star=m, n_star=n
        ),
        [(SW, 20.0, PHI, 2.0, 2.0, 0.45, 0.3)],
    )


def test_equivalence_dual_water_2014_12() -> None:
    assert_matches_original(
        _original_dual_water_2014_12,
        lambda sw, cw, cb, sb, phit, m, n: sat.dual_water_conductivity(
            sw, cw=cw, cb=cb, swb=sb, phi=phit, m=m, n=n
        ),
        [(SW, 20.0, 8.0, 0.2, PHI, 2.0, 2.0)],
    )


def test_equivalence_dual_water_qv_2014_02() -> None:
    assert_matches_original(
        _original_dual_water_qv_2014_02,
        lambda phi, sw, m0, n, cw, qv, alpha, vqh, beta: sat.dual_water_conductivity_qv(
            sw, cw=cw, qv=qv, alpha_vqh=alpha * vqh, beta=beta, phi=phi, m0=m0, n=n
        ),
        [(PHI, SW, 2.0, 2.0, 20.0, 0.3, 0.9, 0.31, 2.0)],
    )


def test_equivalence_qv_helpers() -> None:
    assert_matches_original(
        _original_qv_from_cec_2014_02,
        lambda cec, rho, phi: sat.qv_from_cec(cec, rho_grain=rho, phi=phi),
        [(0.05, 2.65, PHI)],
    )
    assert_matches_original(
        _original_juhasz_qv_2014_12,
        lambda v, rho, cec, phit: sat.qv_juhasz(v, rho_clay=rho, cec_clay=cec, phit=phit),
        [(0.2, 2.78, 0.1, PHI)],
    )


def test_equivalence_exponent_fits_2014_02() -> None:
    ff = sat.formation_factor(PHI, a=0.95, m=2.05) * (1.0 + RNG.normal(0, 0.01, PHI.size))
    ri = sat.resistivity_index_from_sw(SW, n=2.1) * (1.0 + RNG.normal(0, 0.01, SW.size))
    m_old, a_old = _original_fit_cementation_2014_02(PHI, ff)
    m_new, a_new = sat.fit_cementation_exponent(PHI, ff)
    assert m_new == pytest.approx(m_old, rel=1e-14)
    assert a_new == pytest.approx(a_old, rel=1e-14)
    assert sat.fit_saturation_exponent(SW, ri) == pytest.approx(
        _original_fit_saturation_2014_02(SW, ri), rel=1e-14
    )


# --- Saturation-train B2 hazards and variants -------------------------------


def _original_archie_sw_swapped_2018_02(rw, rt, phi, a=1.0, m=2.0, n=2.0):
    # src2018_02/article9 (and src2017_12/article5): argument order (rw, rt)
    return np.clip((a * rw / (phi**m * np.asarray(rt, float))) ** (1.0 / n), 0.0, 1.0)


def test_hazard_swapped_rw_rt_order() -> None:
    rt = np.asarray(sat.archie_rt(SW, RW, phi=PHI))
    correct = _original_archie_sw_swapped_2018_02(RW, rt, PHI)
    facade = sat.archie_sw(rt, RW, phi=PHI, clip=(0.0, 1.0))
    np.testing.assert_allclose(facade, correct, rtol=1e-12, atol=0.0)
    # the trap: passing the article's positional order into the canonical
    # signature swaps rt and rw and produces garbage
    naive = sat.archie_sw(RW, rt, phi=PHI, clip=(0.0, 1.0))
    assert not np.allclose(naive, correct, rtol=1e-3)


def _original_ff_negative_power(phi, m=2.0):
    # src2017_10/article2, src2018_06/article5, src2020_08/article2 spell the
    # formation factor as phi**(-m) (or a*phi**(-m)) rather than a/phi**m
    return np.asarray(phi, float) ** (-m)


def test_formation_factor_negative_power_spelling() -> None:
    # different float path (pow(phi, -m) vs 1/pow(phi, m)): agrees within
    # 1 ULP, far inside the 1e-12 gate
    assert_matches_original(
        _original_ff_negative_power, lambda p, m=2.0: sat.formation_factor(p, m=m), [(PHI,)]
    )
