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


def _original_washburn_diameter_2018_02(pc, sigma=0.485, theta_deg=140.0):
    # src2018_02/article3 washburn_diameter: -4*sigma*cos(theta)/Pc, the diameter
    # dialect of the mercury |cos| (absolute=True) convention (theta in [90,180]).
    return -4.0 * sigma * np.cos(np.radians(theta_deg)) / np.asarray(pc, float)


def _original_young_laplace_diameter_2022_10(d_m, gamma, theta_deg):
    # src2022_10/article2 young_laplace: Pc = 4*gamma*cos(theta)/d (diameter form,
    # signed cos) == 2*gamma*cos/r with r = d/2.
    return 4.0 * gamma * np.cos(np.deg2rad(theta_deg)) / np.asarray(d_m, float)


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
    # src2018_02/article3: the mercury -4*sigma*cos/Pc diameter form == absolute=True
    assert_matches_original(
        _original_washburn_diameter_2018_02,
        lambda pc, s, th: cap.washburn_diameter(pc, sigma=s, theta_deg=th, absolute=True),
        [(PC, 0.485, 140.0), (PC, 0.485, 120.0), (PC, 0.485, 160.0)],
    )
    # src2022_10/article2: Pc = 4*gamma*cos/d (diameter) == young_laplace_pc(r=d/2)
    d_m = RNG.uniform(1e-8, 1e-5, size=30)
    assert_matches_original(
        _original_young_laplace_diameter_2022_10,
        lambda d, g_, th: cap.young_laplace_pc(d / 2.0, sigma=g_, theta_deg=th, absolute=False),
        [(d_m, 0.072, 180.0), (d_m, 0.485, 140.0)],
    )


# --- Leverett J ---------------------------------------------------------------


def _original_leverett_2014_02(pc, sigma, contact_angle_deg, k, phi):
    return pc / (sigma * np.cos(np.radians(contact_angle_deg))) * np.sqrt(k / phi)


def _original_leverett_omitcos_2016_02(pc, sigma, k, phi):
    # src2016_02/article1: J omits cos entirely -> facade passes theta_deg=0
    return (pc / sigma) * np.sqrt(k / phi)


def _original_pc_from_j_2016_02(j, sigma, k, phi):
    # src2016_02/article1 inverse: Pc = J*sigma*sqrt(phi/k)
    return j * sigma * np.sqrt(phi / k)


def _original_leverett_abscos_2022_10(pc, k, phi, sigma, theta_deg):
    # src2022_10/article4 leverett_J: J = Pc*sqrt(k/phi)/(sigma*|cos(theta)|),
    # the |cos| convention (mercury theta=180 keeps J positive).
    return pc * np.sqrt(k / phi) / (sigma * abs(np.cos(np.deg2rad(theta_deg))))


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
    # src2016_02/article1: cos-omitted J and its inverse map with theta_deg=0
    assert_matches_original(
        _original_leverett_omitcos_2016_02,
        lambda pc, s, k, phi: cap.leverett_j(pc, sigma=s, theta_deg=0.0, k=k, phi=phi),
        [(PC, 0.03, 1e-14, 0.2)],
    )
    jvals = _original_leverett_omitcos_2016_02(PC, 0.03, 1e-14, 0.2)
    assert_matches_original(
        _original_pc_from_j_2016_02,
        lambda j_, s, k, phi: cap.pc_from_leverett_j(j_, sigma=s, theta_deg=0.0, k=k, phi=phi),
        [(jvals, 0.03, 1e-14, 0.2)],
    )  # sqrt(phi/k) vs 1/sqrt(k/phi): float reassociation within rtol 1e-12
    # src2022_10/article4: |cos| Leverett (mercury theta=180)
    assert_matches_original(
        _original_leverett_abscos_2022_10,
        lambda pc, k, phi, s, th: cap.leverett_j(
            pc, sigma=s, theta_deg=th, k=k, phi=phi, absolute=True
        ),
        [(PC, 1e-15, 0.1, 0.072, 180.0)],
    )


def _original_pc_convert_mercury_2016_10(pc_am, ift_ab, theta_ab, ift_am, theta_am):
    # src2016_10/article2 pc_air_brine_from_mercury (|cos| convention)
    return pc_am * (
        ift_ab * abs(np.cos(np.radians(theta_ab))) / (ift_am * abs(np.cos(np.radians(theta_am))))
    )


def _original_pc_convert_signed_2016_10(pc_lab, ift_res, theta_res, ift_lab, theta_lab):
    # src2016_10/article2 pc_lab_to_reservoir (signed cos)
    return pc_lab * (
        ift_res * np.cos(np.radians(theta_res)) / (ift_lab * np.cos(np.radians(theta_lab)))
    )


def test_pc_convert_system() -> None:
    # air-mercury (485 mN/m, 140 deg) to gas-brine (72 mN/m, 0 deg)
    pc_res = cap.pc_convert_system(
        PC, sigma_from=0.485, theta_from_deg=140.0, sigma_to=0.072, theta_to_deg=0.0
    )
    ratio = 0.072 / (0.485 * abs(np.cos(np.radians(140.0))))
    np.testing.assert_allclose(pc_res, PC * ratio, rtol=1e-12)
    assert np.all(pc_res > 0)
    # src2016_10: |cos| mercury->air-brine (from = mercury, to = air-brine)
    assert_matches_original(
        _original_pc_convert_mercury_2016_10,
        lambda pc, ift_ab, th_ab, ift_am, th_am: cap.pc_convert_system(
            pc,
            sigma_from=ift_am,
            theta_from_deg=th_am,
            sigma_to=ift_ab,
            theta_to_deg=th_ab,
            absolute=True,
        ),
        [(PC, 72.0, 0.0, 480.0, 140.0)],
    )
    # src2016_10: signed lab->reservoir (from = lab, to = reservoir)
    assert_matches_original(
        _original_pc_convert_signed_2016_10,
        lambda pc, ift_res, th_res, ift_lab, th_lab: cap.pc_convert_system(
            pc,
            sigma_from=ift_lab,
            theta_from_deg=th_lab,
            sigma_to=ift_res,
            theta_to_deg=th_res,
            absolute=False,
        ),
        [(PC, 47.0, 0.0, 72.0, 0.0)],
    )


# --- Brooks-Corey -------------------------------------------------------------


def _original_bc_sw_lam_2018_06(pc, pc_entry, lam=2.0, swirr=0.1):
    pc = np.asarray(pc, float)
    sw = swirr + (1.0 - swirr) * (pc_entry / pc) ** lam
    return np.clip(sw, swirr, 1.0)


def _original_bc_sw_ninv_2016_06(pc, pce, swi, n):
    pc = np.asarray(pc, float)
    sw = swi + (1.0 - swi) * (pce / pc) ** (1.0 / n)
    return np.where(pc <= pce, 1.0, sw)


def _original_bc_pc_normalized_2016_06(sw_star, pc0, ep):
    # src2016_06/article3: Pc = Pc0 * Sw*^(-ep) on a pre-normalized Sw* ->
    # facade passes swirr=0 (identity window) and lam = 1/ep (exponent -1/lam).
    return pc0 * np.asarray(sw_star, float) ** (-ep)


def _original_drainage_pc_offset_2018_02(swn, pc_threshold, a, b):
    # src2018_02/article2: Pc = Pcth + A*Swn^(-B).  The A*Swn^(-B) term is the
    # normalized Brooks-Corey Pc (swirr=0, lam=1/B); Pcth is the article's offset.
    return pc_threshold + a * np.asarray(swn, float) ** (-b)


def _original_toledo_capillary_2021_08(sw_star, pew, D):
    # src2021_08/article8: fractal Pc = Pew*Sw*^(-1/lambda) with lambda = 3 - D ->
    # brooks_corey_pc with lam = 3 - D and pre-normalized Sw* (swirr=0).
    lam = 3.0 - D
    return pew * np.asarray(sw_star, float) ** (-1.0 / lam)


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
    # normalized-saturation Pc form (Sw* already normalized -> swirr=0, lam=1/ep)
    swn = RNG.uniform(0.15, 1.0, size=40)
    assert_matches_original(
        _original_bc_pc_normalized_2016_06,
        lambda sw_, pc0, ep: cap.brooks_corey_pc(sw_, pc_entry=pc0, lam=1.0 / ep, swirr=0.0),
        [(swn, 3e4, 1.5)],
    )
    # src2018_02/article2: offset drainage Pc = Pcth + A*Swn^(-B) (BC-Pc + offset)
    assert_matches_original(
        _original_drainage_pc_offset_2018_02,
        lambda swn_, pcth, a, b: (
            pcth + cap.brooks_corey_pc(swn_, pc_entry=a, lam=1.0 / b, swirr=0.0)
        ),
        [(swn, 1e4, 3e4, 2.0)],
    )
    # src2021_08/article8: Toledo fractal Pc, lam = 3 - D (fractal dimension)
    assert_matches_original(
        _original_toledo_capillary_2021_08,
        lambda sw_, pew, d_: cap.brooks_corey_pc(sw_, pc_entry=pew, lam=3.0 - d_, swirr=0.0),
        [(swn, 5e4, 2.5)],
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


def _original_thomeer_sw_2016_10(pc, pe, g, swirr):
    # src2016_10/article2 thomeer_sw: Sw = 1 - (1-Swirr)*exp(-G/log10(Pc/Pe)),
    # i.e. the water-saturation complement of thomeer_shg with ceiling (1-Swirr)
    pc = np.asarray(pc, float)
    snw = np.where(pc > pe, (1.0 - swirr) * np.exp(-g / np.log10(pc / pe)), 0.0)
    return 1.0 - snw


def test_thomeer_sw_complement_2016_10() -> None:
    pe, g, swirr = 50.0, 0.35, 0.15
    assert_matches_original(
        _original_thomeer_sw_2016_10,
        lambda pc, pe_, g_, sw_: (
            1.0 - cap.thomeer_shg(pc, bv_inf=1.0 - sw_, g=g_, pd=pe_, log_base=10.0)
        ),
        [(PC, pe, g, swirr)],
        rtol=1e-9,  # log10(Pc/Pe) vs log10(Pc)-log10(Pe): float reassociation
    )


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


G_ACCEL_2018 = 9.81  # the tutorial trilogy's rounded gravity, passed at the facade


def _original_capillary_rise_2018_08(sigma, theta_deg, rho, r, rho_above=0.0):
    # src2018_08/article1 capillary_rise_height (signed cos, g = 9.81)
    drho = rho - rho_above
    num = 2.0 * sigma * np.cos(np.radians(theta_deg))
    return num / (drho * G_ACCEL_2018 * np.asarray(r, float))


def _original_pc_from_rise_2018_08(rho_w, rho_a, h):
    # src2018_08/article1 capillary_pressure_from_rise (SI buoyancy, g = 9.81)
    return (rho_w - rho_a) * G_ACCEL_2018 * np.asarray(h, float)


def _original_saturation_height_2018_10(pc, rho_w, rho_hc):
    # src2018_10/article1 saturation_height = height_above_fwl (SI, g = 9.81)
    return np.asarray(pc, float) / ((rho_w - rho_hc) * G_ACCEL_2018)


def test_trilogy_gravity_forms_2018() -> None:
    r = RNG.uniform(1e-6, 5e-5, size=30)
    assert_matches_original(
        _original_capillary_rise_2018_08,
        lambda s, th, rho, r_, rho_above=0.0: cap.capillary_rise_height(
            r_, sigma=s, theta_deg=th, delta_rho=rho - rho_above, g=G_ACCEL_2018
        ),
        [(0.072, 0.0, 1000.0, r), (0.485, 140.0, 13546.0, r, 1.2)],
    )
    h = RNG.uniform(0.1, 50.0, size=30)
    assert_matches_original(
        _original_pc_from_rise_2018_08,
        lambda rw, ra, h_: cap.buoyancy_pc(h_, delta_rho=rw - ra, g=G_ACCEL_2018),
        [(1000.0, 1.2, h)],
    )
    assert_matches_original(
        _original_saturation_height_2018_10,
        lambda pc, rw, rhc: cap.height_above_fwl(pc, delta_rho=rw - rhc, g=G_ACCEL_2018),
        [(PC, 1000.0, 700.0)],
    )


def test_buoyancy_round_trip_and_oilfield() -> None:
    h = RNG.uniform(1.0, 120.0, size=30)
    pc = cap.buoyancy_pc(h, delta_rho=300.0)
    np.testing.assert_allclose(cap.height_above_fwl(pc, delta_rho=300.0), h, rtol=1e-12)
    # oilfield form, visible 0.433 default (src2016_06)
    assert cap.buoyancy_pc_gradient(100.0, sg_water=1.05, sg_hc=0.75) == pytest.approx(
        0.433 * 0.30 * 100.0
    )


def _original_centrifuge_rpm_r1r2_2020_08(drho, rpm, r1, r2):
    # src2020_08/article3 centrifuge_pc: rpm -> omega at the facade, radii r1,r2
    w = 2.0 * np.pi * rpm / 60.0
    return 0.5 * drho * w**2 * (r2**2 - r1**2)


def _original_centrifuge_rpm_llr_2020_06(drho, rpm, length, lr):
    # src2020_06/article3 centrifuge_pc: two radii as (LR - L) inner and LR outer
    w = 2.0 * np.pi * rpm / 60.0
    return 0.5 * drho * w**2 * (lr**2 - (lr - length) ** 2)


def _original_washburn_length_cliptozero_2020_04(sigma, r, theta_deg, mu, t):
    # src2020_04/article8 washburn_length: Lucas-Washburn with clip-to-zero for
    # oil-wet pores (cos < 0) instead of the library's NaN.
    c = np.cos(np.radians(theta_deg))
    drive = sigma * r * c * np.asarray(t, float) / (2.0 * mu)
    return np.sqrt(np.clip(drive, 0.0, None))


def test_centrifuge_and_rise_and_lucas_washburn() -> None:
    # centrifuge: 3000 rpm, water-air, r1=0.10 m, r2=0.13 m
    omega = 3000.0 * 2.0 * np.pi / 60.0
    pc = cap.centrifuge_pc(omega, delta_rho=1000.0, r1=0.10, r2=0.13)
    assert pc == pytest.approx(0.5 * 1000.0 * omega**2 * (0.13**2 - 0.10**2))
    # rpm-at-facade centrifuge forms (src2020_08 r1/r2, src2020_06 L/LR)
    assert_matches_original(
        _original_centrifuge_rpm_r1r2_2020_08,
        lambda drho, rpm, r1, r2: cap.centrifuge_pc(
            2.0 * np.pi * rpm / 60.0, delta_rho=drho, r1=r1, r2=r2
        ),
        [(300.0, 2500.0, 0.08, 0.12)],
    )
    assert_matches_original(
        _original_centrifuge_rpm_llr_2020_06,
        lambda drho, rpm, length, lr: cap.centrifuge_pc(
            2.0 * np.pi * rpm / 60.0, delta_rho=drho, r1=lr - length, r2=lr
        ),
        [(200.0, 3000.0, 0.05, 0.13)],
    )
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
    # src2020_04/article8: clip-to-zero facade (nan_to_num) matches, both wettings
    t = RNG.uniform(0.5, 12.0, size=30)
    assert_matches_original(
        _original_washburn_length_cliptozero_2020_04,
        lambda s, r_, th, mu, t_: np.nan_to_num(
            cap.lucas_washburn_length(t_, sigma=s, radius=r_, theta_deg=th, mu=mu), nan=0.0
        ),
        [(0.03, 1e-6, 40.0, 1e-3, t), (0.03, 1e-6, 120.0, 1e-3, t)],
    )
