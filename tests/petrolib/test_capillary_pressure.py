"""Tests for petrolib.capillary_pressure: golden values, round trips, and
shadow equivalence against verbatim article bodies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import capillary_pressure as cap  # noqa: E402
from petrolib.testing import assert_matches_original  # noqa: E402

RNG = np.random.default_rng(20260711)
PC = RNG.uniform(1e4, 5e6, size=40)  # Pa
SW = RNG.uniform(0.25, 0.99, size=40)


# --- Young-Laplace / Washburn ------------------------------------------------


def test_washburn_young_laplace_round_trip() -> None:
    r = RNG.uniform(1e-8, 1e-5, size=40)
    pc = cap.young_laplace_pc(r, sigma=0.485, theta_deg=140.0)
    np.testing.assert_allclose(cap.washburn_radius(pc, sigma=0.485, theta_deg=140.0), r, rtol=1e-12)
    np.testing.assert_allclose(
        cap.washburn_diameter(pc, sigma=0.485, theta_deg=140.0), 2.0 * r, rtol=1e-12
    )


def test_young_laplace_signs_and_golden() -> None:
    # pure meniscus: 2*0.03/1e-6 = 60 kPa
    assert cap.young_laplace_pc(1e-6, sigma=0.03) == pytest.approx(6.0e4)
    # mercury: |cos 140| keeps MICP pressures positive; signed goes negative
    assert cap.young_laplace_pc(1e-6, sigma=0.485, theta_deg=140.0) > 0
    assert cap.young_laplace_pc(1e-6, sigma=0.485, theta_deg=140.0, absolute=False) < 0


def _original_pore_throat_radius_2014_02(pc, sigma, contact_angle_deg):
    # src2014_02/article2 (signed cos)
    return 2.0 * sigma * np.cos(np.radians(contact_angle_deg)) / np.asarray(pc, float)


def _original_pc_radius_2018_08(sigma, theta_deg, r):
    # src2018_08/article1 (|cos| tutorial convention)
    return 2.0 * sigma * abs(np.cos(np.radians(theta_deg))) / np.asarray(r, float)


def _original_young_laplace_signed_2014_02(sigma_wo, contact_angle_deg, pore_radius):
    # src2014_02/article3 capillary_pressure (signed cos: oil-wet Pc < 0)
    return 2.0 * sigma_wo * np.cos(np.radians(contact_angle_deg)) / pore_radius


def _original_washburn_mercury_2014_12(pc, sigma=0.480, theta_deg=140.0):
    # src2014_12/article3 washburn_radius: -2*sigma*cos(theta)/Pc.  The leading
    # minus folds the obtuse mercury angle to a positive radius, which equals the
    # |cos| (absolute=True) convention for theta in [90, 180] deg.
    return -2.0 * sigma * np.cos(np.radians(theta_deg)) / np.asarray(pc, float)


def test_equivalence_washburn_variants() -> None:
    assert_matches_original(
        _original_pore_throat_radius_2014_02,
        lambda pc, s, th: cap.washburn_radius(pc, sigma=s, theta_deg=th, absolute=False),
        [(PC, 0.03, 0.0), (PC, 0.485, 140.0)],
    )
    r = RNG.uniform(1e-8, 1e-5, size=30)
    assert_matches_original(
        _original_pc_radius_2018_08,
        lambda s, th, r_: cap.young_laplace_pc(r_, sigma=s, theta_deg=th),
        [(0.485, 140.0, r)],
    )
    # src2014_02/article3: signed Young-Laplace (water-wet positive, oil-wet negative)
    assert_matches_original(
        _original_young_laplace_signed_2014_02,
        lambda s, th, r_: cap.young_laplace_pc(r_, sigma=s, theta_deg=th, absolute=False),
        [(0.00921, 32.0, r), (0.00265, 134.0, r)],
    )
    # src2014_12/article3: the mercury -2*sigma*cos/Pc form == absolute=True for theta>=90
    assert_matches_original(
        _original_washburn_mercury_2014_12,
        lambda pc, s, th: cap.washburn_radius(pc, sigma=s, theta_deg=th, absolute=True),
        [(PC, 0.480, 140.0), (PC, 0.480, 120.0), (PC, 0.480, 160.0)],
    )


# --- Leverett J ---------------------------------------------------------------


def _original_leverett_2014_02(pc, sigma, contact_angle_deg, k, phi):
    return pc / (sigma * np.cos(np.radians(contact_angle_deg))) * np.sqrt(k / phi)


def test_leverett_round_trip_and_equivalence() -> None:
    j = cap.leverett_j(PC, sigma=0.03, theta_deg=20.0, k=1e-14, phi=0.2)
    np.testing.assert_allclose(
        cap.pc_from_leverett_j(j, sigma=0.03, theta_deg=20.0, k=1e-14, phi=0.2),
        PC,
        rtol=1e-12,
    )
    assert_matches_original(
        _original_leverett_2014_02,
        lambda pc, s, th, k, phi: cap.leverett_j(
            pc, sigma=s, theta_deg=th, k=k, phi=phi, absolute=False
        ),
        [(5e5, 0.03, 0.0, 1e-12, 0.30)],
    )


def test_pc_convert_system() -> None:
    # air-mercury (485 mN/m, 140 deg) to gas-brine (72 mN/m, 0 deg)
    pc_res = cap.pc_convert_system(
        PC, sigma_from=0.485, theta_from_deg=140.0, sigma_to=0.072, theta_to_deg=0.0
    )
    ratio = 0.072 / (0.485 * abs(np.cos(np.radians(140.0))))
    np.testing.assert_allclose(pc_res, PC * ratio, rtol=1e-12)
    assert np.all(pc_res > 0)


# --- Brooks-Corey -------------------------------------------------------------


def _original_bc_sw_lam_2018_06(pc, pc_entry, lam=2.0, swirr=0.1):
    pc = np.asarray(pc, float)
    sw = swirr + (1.0 - swirr) * (pc_entry / pc) ** lam
    return np.clip(sw, swirr, 1.0)


def _original_bc_sw_ninv_2016_06(pc, pce, swi, n):
    pc = np.asarray(pc, float)
    sw = swi + (1.0 - swi) * (pce / pc) ** (1.0 / n)
    return np.where(pc <= pce, 1.0, sw)


def test_brooks_corey_conventions_and_round_trip() -> None:
    # lam-convention article (clip form is value-equal to the where form)
    assert_matches_original(
        _original_bc_sw_lam_2018_06,
        lambda pc, pe, lam=2.0, swirr=0.1: cap.brooks_corey_sw(
            pc, pc_entry=pe, lam=lam, swirr=swirr
        ),
        [(PC, 5e4), ((PC, 5e4), {"lam": 1.7, "swirr": 0.15})],
    )
    # 1/N-convention article maps with lam = 1/N — the reciprocal hazard
    assert_matches_original(
        _original_bc_sw_ninv_2016_06,
        lambda pc, pce, swi, n: cap.brooks_corey_sw(pc, pc_entry=pce, lam=1.0 / n, swirr=swi),
        [(PC, 5e4, 0.12, 2.4)],
    )
    # forward/inverse round trip above the entry pressure
    pc = cap.brooks_corey_pc(SW, pc_entry=5e4, lam=1.8, swirr=0.1)
    sw_back = cap.brooks_corey_sw(pc, pc_entry=5e4, lam=1.8, swirr=0.1)
    np.testing.assert_allclose(sw_back, SW, rtol=1e-10)


# --- Thomeer -------------------------------------------------------------------


def _original_thomeer_log10_2021_04(Pc, Bv, G, Pd):
    Pc = np.asarray(Pc, float)
    out = Bv * np.exp(-G / (np.log10(Pc) - np.log10(Pd)))
    return np.where(Pc > Pd, out, 0.0)


def _original_thomeer_ln_2021_08(Pc, Binf, G, Pd):
    Pc = np.asarray(Pc, float)
    return np.where(Pc > Pd, Binf * np.exp(-G / np.log(Pc / Pd)), 0.0)


def test_thomeer_log_base_hazard() -> None:
    bv, g10, pd = 0.25, 0.4, 2e4
    assert_matches_original(
        _original_thomeer_log10_2021_04,
        lambda pc, b, g_, p: cap.thomeer_shg(pc, bv_inf=b, g=g_, pd=p, log_base=10.0),
        [(PC, bv, g10, pd)],
        rtol=1e-9,  # log10(a)-log10(b) vs log10(a/b): float reassociation
    )
    assert_matches_original(
        _original_thomeer_ln_2021_08,
        lambda pc, b, g_, p: cap.thomeer_shg(pc, bv_inf=b, g=g_, pd=p, log_base=np.e),
        [(PC, bv, g10 * np.log(10.0), pd)],
    )
    # the hazard: same G under the two bases gives materially different curves
    s10 = cap.thomeer_shg(PC, bv_inf=bv, g=g10, pd=pd, log_base=10.0)
    s_ln = cap.thomeer_shg(PC, bv_inf=bv, g=g10, pd=pd, log_base=np.e)
    assert not np.allclose(s10, s_ln, rtol=0.05)
    # equivalence holds when G is converted by ln(10)
    np.testing.assert_allclose(
        cap.thomeer_shg(PC, bv_inf=bv, g=g10 * np.log(10.0), pd=pd, log_base=np.e),
        s10,
        rtol=1e-9,
    )


# --- Buoyancy / lab kernels -----------------------------------------------------


def test_buoyancy_round_trip_and_oilfield() -> None:
    h = RNG.uniform(1.0, 120.0, size=30)
    pc = cap.buoyancy_pc(h, delta_rho=300.0)
    np.testing.assert_allclose(cap.height_above_fwl(pc, delta_rho=300.0), h, rtol=1e-12)
    # oilfield form, visible 0.433 default (src2016_06)
    assert cap.buoyancy_pc_gradient(100.0, sg_water=1.05, sg_hc=0.75) == pytest.approx(
        0.433 * 0.30 * 100.0
    )


def test_centrifuge_and_rise_and_lucas_washburn() -> None:
    # centrifuge: 3000 rpm, water-air, r1=0.10 m, r2=0.13 m
    omega = 3000.0 * 2.0 * np.pi / 60.0
    pc = cap.centrifuge_pc(omega, delta_rho=1000.0, r1=0.10, r2=0.13)
    assert pc == pytest.approx(0.5 * 1000.0 * omega**2 * (0.13**2 - 0.10**2))
    # wetting fluid rises, mercury is depressed
    assert cap.capillary_rise_height(1e-4, sigma=0.072, delta_rho=1000.0) > 0
    assert cap.capillary_rise_height(1e-4, sigma=0.485, theta_deg=140.0, delta_rho=13546.0) < 0
    # imbibition front grows as sqrt(t); oil-wet gives NaN, not zero
    length_1 = cap.lucas_washburn_length(1.0, sigma=0.03, radius=1e-6, mu=1e-3)
    length_4 = cap.lucas_washburn_length(4.0, sigma=0.03, radius=1e-6, mu=1e-3)
    assert length_4 == pytest.approx(2.0 * length_1)
    assert np.isnan(
        cap.lucas_washburn_length(1.0, sigma=0.03, radius=1e-6, theta_deg=120.0, mu=1e-3)
    )
