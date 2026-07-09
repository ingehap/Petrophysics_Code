"""Tests for petrolib.relperm_wettability: golden values, round trips, and
shadow equivalence against verbatim article bodies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import relperm_wettability as rp  # noqa: E402
from petrolib.testing import assert_matches_original  # noqa: E402

RNG = np.random.default_rng(20260712)
SW = np.linspace(0.05, 0.98, 40)


# --- normalized saturation ----------------------------------------------------


def test_normalized_saturation_conventions() -> None:
    # residual pair, clipped to the mobile window
    se = rp.normalized_saturation(SW, 0.15, 0.20)
    assert se.min() == 0.0 and se.max() == 1.0
    # snr=0 gives the (1-Sr) drainage denominator
    np.testing.assert_allclose(
        rp.normalized_saturation(0.5, 0.1, clip=None), (0.5 - 0.1) / (1 - 0.1)
    )
    # clip=None leaves Se unbounded
    assert rp.normalized_saturation(0.99, 0.15, 0.20, clip=None) > 1.0


# --- Corey free-exponent ------------------------------------------------------


def _original_corey_water_2014_08(sw, swc, sor, krw_max=1.0, nw=2.0):
    sw_star = np.clip((np.asarray(sw, float) - swc) / (1 - swc - sor), 0, 1)
    return krw_max * sw_star**nw


def _original_corey_oil_water_2014_08(sw, swc, sor, kro_max=1.0, no=2.0):
    sw_star = np.clip((np.asarray(sw, float) - swc) / (1 - swc - sor), 0, 1)
    return kro_max * (1 - sw_star) ** no


def _original_corey_kro_independent_2024_10(sw, swir, sor, kro0, alpha_o):
    # src2025_12/mri_rel_perm oil normalized independently as (1-Sw-Sor)/(...)
    se = np.clip((1.0 - np.asarray(sw, float) - sor) / (1.0 - swir - sor), 0.0, 1.0)
    return kro0 * se**alpha_o


def test_corey_water_oil_equivalence() -> None:
    assert_matches_original(
        _original_corey_water_2014_08,
        lambda sw, swc, sor, krw_max=1.0, nw=2.0: rp.corey_krw(
            sw, swr=swc, sor=sor, krw_max=krw_max, nw=nw
        ),
        [(SW, 0.15, 0.20), (SW, 0.1, 0.25, 0.3, 3.0)],
    )
    assert_matches_original(
        _original_corey_oil_water_2014_08,
        lambda sw, swc, sor, kro_max=1.0, no=2.0: rp.corey_kro(
            sw, swr=swc, sor=sor, kro_max=kro_max, no=no
        ),
        [(SW, 0.15, 0.20), (SW, 0.1, 0.25, 0.9, 2.5)],
    )
    # independent-oil-normalization articles agree to ~1 ULP with the (1-Se) form
    assert_matches_original(
        _original_corey_kro_independent_2024_10,
        lambda sw, swir, sor, kro0, alpha_o: rp.corey_kro(
            sw, swr=swir, sor=sor, kro_max=kro0, no=alpha_o
        ),
        [(SW, 0.15, 0.20, 0.9, 2.0)],
    )


def _original_corey_pair_2017_02(sw, swc, sor, krw_max, kro_max, nw, no):
    swn = np.clip((sw - swc) / (1.0 - swc - sor), 0.0, 1.0)
    return krw_max * swn**nw, kro_max * (1.0 - swn) ** no


def test_corey_pair_equivalence() -> None:
    assert_matches_original(
        _original_corey_pair_2017_02,
        lambda sw, swc, sor, krw_max, kro_max, nw, no: rp.corey_kr(
            sw, swr=swc, sor=sor, krw_max=krw_max, kro_max=kro_max, nw=nw, no=no
        ),
        [(SW, 0.15, 0.2, 0.3, 1.0, 3.0, 2.0)],
    )


def _original_corey_krg_4endpoint_2014_08(sg, swc, sgc, sorg, krg_max=1.0, ng=2.0):
    sg_star = np.clip((sg - sgc) / (1 - swc - sgc - sorg), 0, 1)
    return krg_max * sg_star**ng


def _original_corey_brooks_krg_swframe_2020_06(sw, swr, sgc, ng, krg_max=1.0):
    num = np.clip(1.0 - sw - sgc, 0.0, None)
    krg = krg_max * (num / (1.0 - swr - sgc)) ** ng
    return np.clip(krg, 0.0, krg_max)


def test_corey_gas_equivalence() -> None:
    sg = np.linspace(0.0, 0.85, 40)
    assert_matches_original(
        _original_corey_krg_4endpoint_2014_08,
        lambda sg, swc, sgc, sorg, krg_max=1.0, ng=2.0: rp.corey_krg(
            sg, sgc=sgc, swc=swc, sorg=sorg, krg_max=krg_max, ng=ng
        ),
        [(sg, 0.2, 0.05, 0.1)],
    )
    # Sw-framework gas maps by passing sg = 1 - sw (bit-exact numerator/denominator)
    sw = np.linspace(0.15, 0.95, 40)
    np.testing.assert_allclose(
        _original_corey_brooks_krg_swframe_2020_06(sw, 0.15, 0.05, 2.0),
        rp.corey_krg(1.0 - sw, sgc=0.05, swc=0.15, sorg=0.0, ng=2.0),
        rtol=1e-12,
    )


# --- Brooks-Corey Burdine -----------------------------------------------------


def _original_bc_burdine_2020_04(sw, swr, lam, snwr=0.0):
    se = np.clip((sw - swr) / (1 - swr - snwr), 0, 1)
    krw = se ** ((2.0 + 3.0 * lam) / lam)
    krg = (1.0 - se) ** 2 * (1.0 - se ** ((2.0 + lam) / lam))
    return krw, krg


def test_brooks_corey_burdine_equivalence() -> None:
    assert_matches_original(
        _original_bc_burdine_2020_04,
        lambda sw, swr, lam, snwr=0.0: rp.brooks_corey_burdine_kr(sw, swr=swr, lam=lam, snwr=snwr),
        [(SW, 0.15, 2.0), (SW, 0.12, 1.5, 0.1)],
    )
    # distinct from free-exponent Corey: the exponents are functions of lam
    krw, _ = rp.brooks_corey_burdine_kr(SW, swr=0.15, lam=2.0)
    krw_corey = rp.corey_krw(SW, swr=0.15, nw=2.0)
    assert not np.allclose(krw, krw_corey)


# --- LET ----------------------------------------------------------------------


def _original_let_wetting_2025_02(sw, swr, L, E, T, krw_max):
    swn = np.clip((sw - swr) / (1 - swr), 0, 1)
    nw = swn**L
    dw = nw + E * (1 - swn) ** T
    dw = np.where(dw == 0, 1e-30, dw)
    return np.clip(krw_max * nw / dw, 0, 1)


def _original_let_gas_2025_02(sw, swir, krg_max, L, E, T):
    swn = np.clip((sw - swir) / (1 - swir), 0, 1)
    n = (1 - swn) ** L
    d = n + E * swn**T
    d = np.where(d == 0, 1e-30, d)
    return krg_max * n / d


def test_let_equivalence() -> None:
    # wetting phase: the scal_model_ccs form clips output to [0,1]; over the
    # mobile window LET is already within [0, krw_max] so the clip is a no-op.
    assert_matches_original(
        _original_let_wetting_2025_02,
        lambda sw, swr, L, E, T, krw_max: rp.let_kr(
            sw, swr=swr, L=L, E=E, T=T, kr_max=krw_max, phase="wetting"
        ),
        [(SW, 0.15, 2.0, 2.0, 2.0, 0.6)],
    )
    assert_matches_original(
        _original_let_gas_2025_02,
        lambda sw, swir, krg_max, L, E, T: rp.let_kr(
            sw, swr=swir, L=L, E=E, T=T, kr_max=krg_max, phase="nonwetting"
        ),
        [(SW, 0.15, 1.0, 2.5, 1.8, 1.5)],
    )
    with pytest.raises(ValueError):
        rp.let_kr(0.5, swr=0.15, L=2.0, E=2.0, T=2.0, phase="bogus")


# --- mobility & fractional flow -----------------------------------------------


def _original_fractional_flow_reciprocal_2019_02(krw, kro, mu_w, mu_o):
    # src2019_02/article1 reciprocal formulation
    return 1.0 / (1.0 + (kro * mu_w) / (krw * mu_o))


def _original_fractional_flow_analog_2025_12(kr_w, mu_w, kr_nw, mu_nw):
    lam_w = np.asarray(kr_w, float) / mu_w
    lam_nw = np.asarray(kr_nw, float) / mu_nw
    denom = lam_w + lam_nw
    return np.where(denom > 0, lam_w / denom, 0.0)


def test_mobility_and_fractional_flow() -> None:
    np.testing.assert_allclose(rp.phase_mobility(0.3, 2e-3), 150.0)
    krw = rp.corey_krw(SW, swr=0.15, sor=0.2, nw=2.0)
    kro = rp.corey_kro(SW, swr=0.15, sor=0.2, no=2.0)
    # analog_kr guarded form is bit-exact
    assert_matches_original(
        _original_fractional_flow_analog_2025_12,
        lambda kw, mw, kn, mn: rp.fractional_flow(kw, kn, mu_w=mw, mu_nw=mn),
        [(krw, 1e-3, kro, 2e-3)],
    )
    # reciprocal formulation agrees where krw > 0 (~1 ULP)
    mask = krw > 0
    np.testing.assert_allclose(
        _original_fractional_flow_reciprocal_2019_02(krw[mask], kro[mask], 1e-3, 2e-3),
        rp.fractional_flow(krw[mask], kro[mask], mu_w=1e-3, mu_nw=2e-3),
        rtol=1e-12,
    )
    # endpoint mobility ratio
    assert rp.endpoint_mobility_ratio(0.4, 1.0, mu_w=0.5, mu_o=2.0) == pytest.approx(
        (0.4 / 0.5) / (1.0 / 2.0)
    )


# --- Buckley-Leverett / Welge -------------------------------------------------


def _original_welge_shock_front_2016_02(swc, sorw, mu_w, mu_o, krw_max, kro_max, nw, no, n):
    sw = np.linspace(swc + 1e-4, 1.0 - sorw - 1e-4, n)
    swn = np.clip((sw - swc) / (1.0 - swc - sorw), 0.0, 1.0)
    krw = krw_max * swn**nw
    kro = kro_max * (1.0 - swn) ** no
    mob_w = krw / mu_w
    mob_o = kro / mu_o
    fw = mob_w / (mob_w + mob_o)
    secant = fw / (sw - swc)
    i = int(np.argmax(secant))
    swf, fwf = sw[i], fw[i]
    dfw = secant[i]
    sw_avg = swf + (1.0 - fwf) / dfw
    return float(swf), float(fwf), float(sw_avg)


def test_welge_shock() -> None:
    swc, sorw, mu_w, mu_o, krw_max, kro_max, nw, no, n = (
        0.2,
        0.2,
        1e-3,
        2e-3,
        0.3,
        1.0,
        3.0,
        2.0,
        2000,
    )
    swf0, fwf0, avg0 = _original_welge_shock_front_2016_02(
        swc, sorw, mu_w, mu_o, krw_max, kro_max, nw, no, n
    )
    # reproduce on the same grid via the library primitives
    sw = np.linspace(swc + 1e-4, 1.0 - sorw - 1e-4, n)
    krw, kro = rp.corey_kr(sw, swr=swc, sor=sorw, krw_max=krw_max, kro_max=kro_max, nw=nw, no=no)
    fw = rp.fractional_flow(krw, kro, mu_w=mu_w, mu_nw=mu_o)
    swf, fwf, avg = rp.welge_shock(sw, fw, swc)
    assert (swf, fwf, avg) == pytest.approx((swf0, fwf0, avg0), rel=1e-9)
    # front is inside the mobile window and average is behind it
    assert swc < swf < 1.0 - sorw
    assert avg > swf


def test_fractional_flow_curve_monotone() -> None:
    sw, fw = rp.fractional_flow_curve(
        swr=0.15, sor=0.2, krw_max=0.3, kro_max=1.0, nw=3.0, no=2.0, mu_w=1e-3, mu_nw=2e-3
    )
    assert sw[0] == pytest.approx(0.15) and sw[-1] == pytest.approx(0.8)
    assert fw[0] == 0.0 and fw[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fw) >= -1e-12)  # fw is non-decreasing in Sw


# --- fit_corey (scipy) --------------------------------------------------------


def test_fit_corey_recovers_parameters() -> None:
    pytest.importorskip("scipy")
    swr, sor, krw_max, nw, kro_max, no = 0.15, 0.2, 0.35, 2.6, 0.9, 2.1
    sw = np.linspace(swr, 1 - sor, 25)
    krw, kro = rp.corey_kr(sw, swr=swr, sor=sor, krw_max=krw_max, kro_max=kro_max, nw=nw, no=no)
    fit = rp.fit_corey(sw, krw, kro, swr=swr, sor=sor)
    assert fit["krw_max"] == pytest.approx(krw_max, rel=1e-4)
    assert fit["nw"] == pytest.approx(nw, rel=1e-4)
    assert fit["krnw_max"] == pytest.approx(kro_max, rel=1e-4)
    assert fit["nnw"] == pytest.approx(no, rel=1e-4)
