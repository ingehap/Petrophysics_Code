"""Tests for the depth_imaging domain: wellbore_geometry / depth_correction /
borehole_image / depth_matching -- known-value checks, round-trips on synthetic
problems, and error paths."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import borehole_image as bi  # noqa: E402
from petrolib import depth_correction as dc  # noqa: E402
from petrolib import depth_matching as dm  # noqa: E402
from petrolib import wellbore_geometry as wg  # noqa: E402

# --- wellbore_geometry --------------------------------------------------------


def test_dogleg_and_ratio_factor() -> None:
    # a pure inclination build of 30 deg (no azimuth change) has dogleg 30 deg
    np.testing.assert_allclose(wg.dogleg_angle(0.0, 0.0, 30.0, 0.0), np.radians(30.0))
    assert wg.ratio_factor(0.0) == 1.0  # straight limit
    np.testing.assert_allclose(wg.ratio_factor(1e-12), 1.0)
    # RF > 1 for a real dogleg, and RF -> 1 as DL -> 0
    assert wg.ratio_factor(np.radians(30.0)) > 1.0
    assert wg.ratio_factor(np.radians(1.0)) < wg.ratio_factor(np.radians(30.0))


def test_minimum_curvature_and_path() -> None:
    # vertical hold: only TVD advances
    d_tvd, d_n, d_e = wg.minimum_curvature_step(0.0, 0.0, 0.0, 100.0, 0.0, 0.0)
    np.testing.assert_allclose([d_tvd, d_n, d_e], [100.0, 0.0, 0.0])
    # a full survey: TVD is monotone and less than MD once inclined
    path = wg.survey_to_path([0, 100, 200, 300], [0, 20, 40, 60], [0, 90, 90, 90])
    assert path.shape == (4, 3)
    assert np.all(np.diff(path[:, 0]) > 0)  # TVD increases
    assert path[-1, 0] < 300.0  # deviated, so TVD < MD
    # east drift dominates (azimuth 90) over north
    assert path[-1, 2] > abs(path[-1, 1])


def test_md_to_tvd_methods() -> None:
    md = np.array([0.0, 100.0, 200.0])
    inc = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(wg.md_to_tvd(md, inc, method="tangential"), md)
    np.testing.assert_allclose(wg.md_to_tvd(md, inc, method="min_curvature"), md)
    # at 60 deg constant inclination, tangential TVD = MD*cos(60)
    inc2 = np.array([60.0, 60.0, 60.0])
    np.testing.assert_allclose(wg.md_to_tvd(md, inc2, method="tangential"), md * 0.5)
    with pytest.raises(ValueError, match="method"):
        wg.md_to_tvd(md, inc, method="bogus")


# --- depth_correction ---------------------------------------------------------


def test_stretch_and_thermal() -> None:
    # point load F*L/(E*A)
    np.testing.assert_allclose(
        dc.elastic_stretch(1.0e5, 3000.0, 5.0e-4), 1.0e5 * 3000.0 / (2.0e11 * 5.0e-4)
    )
    # distributed: end-load term + own weight term
    got = dc.distributed_stretch(3000.0, 20.0, 3.0e7, end_load=5000.0)
    np.testing.assert_allclose(got, (5000.0 * 3000.0 + 0.5 * 20.0 * 3000.0**2) / 3.0e7)
    # thermal alpha*L*dT
    np.testing.assert_allclose(dc.thermal_elongation(3000.0, 50.0), 1.2e-5 * 3000.0 * 50.0)


def test_cable_tension_and_corrected_depth() -> None:
    t = dc.cable_tension([0.0, 1500.0, 3000.0], 3000.0, 5000.0, 20.0)
    # max at surface, equals tool weight at the tool
    np.testing.assert_allclose(t[0], 5000.0 + 20.0 * 3000.0)
    np.testing.assert_allclose(t[-1], 5000.0)
    assert t[0] > t[1] > t[2]
    np.testing.assert_allclose(dc.corrected_depth(3000.0, stretch=0.5, thermal=0.2), 3000.7)
    np.testing.assert_allclose(
        dc.corrected_depth(3000.0, stretch=0.5, thermal=0.2, convention="payout"), 2999.3
    )
    with pytest.raises(ValueError, match="convention"):
        dc.corrected_depth(3000.0, convention="bogus")


# --- borehole_image -----------------------------------------------------------


def test_bed_sinusoid_roundtrip_and_dip() -> None:
    az = np.linspace(0.0, 330.0, 12)
    radius, dip, dip_az = 0.1, 30.0, 60.0
    z = bi.bed_sinusoid(az, 100.0, radius, dip, dip_az)
    z0, amp, _phase = bi.fit_sinusoid(az, z)
    np.testing.assert_allclose(z0, 100.0, atol=1e-9)
    np.testing.assert_allclose(amp, radius * np.tan(np.radians(dip)), atol=1e-9)
    # dip recovered from amplitude and radius
    np.testing.assert_allclose(bi.dip_from_amplitude(amp, radius), dip, atol=1e-6)
    # mask drops samples but still fits
    m = np.ones(12, bool)
    m[:3] = False
    z0b, ampb, _ = bi.fit_sinusoid(az, z, mask=m)
    np.testing.assert_allclose(ampb, amp, atol=1e-9)


def test_apparent_dip_and_fit_plane() -> None:
    # apparent dip equals true dip in the dip section (beta=0) and 0 in strike (beta=90)
    np.testing.assert_allclose(bi.apparent_dip(30.0, 0.0), 30.0)
    np.testing.assert_allclose(bi.apparent_dip(30.0, 90.0), 0.0, atol=1e-12)
    # a plane dipping 45 deg to the East: z = 1*E, normal ~ (E,N,Z)
    e = np.array([0.0, 1.0, 0.0, 1.0])
    n = np.array([0.0, 0.0, 1.0, 1.0])
    z = e  # TVD increases with East => 45 deg dip toward East (az 90)
    dip, az = bi.fit_plane(np.column_stack([e, n, z]))
    np.testing.assert_allclose(dip, 45.0, atol=1e-6)
    assert abs((az - 90.0 + 180.0) % 180.0) < 1e-6  # az is 90 or 270 (ambiguous)


def test_otsu_class_fractions_masks() -> None:
    rng = np.random.default_rng(0)
    img = np.concatenate([rng.normal(0.2, 0.03, 500), rng.normal(0.8, 0.03, 500)])
    t = bi.otsu_threshold(img)
    # the empty valley between the modes makes Otsu land at its left edge; the
    # threshold still cleanly separates the two clusters
    assert 0.25 < t < 0.75
    assert img[img < t].mean() < 0.4 and img[img >= t].mean() > 0.6
    fr = bi.class_fractions(img, [0.5])
    np.testing.assert_allclose(fr.sum(), 1.0)
    np.testing.assert_allclose(fr, [0.5, 0.5], atol=0.02)
    pore = np.array([[True, True], [False, True]])
    phase = np.array([[True, False], [False, True]])
    np.testing.assert_allclose(bi.phase_saturation(phase, pore), 2.0 / 3.0)
    np.testing.assert_allclose(bi.porosity_from_mask(pore), 0.75)


def test_glcm_and_sobel() -> None:
    img = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], float)
    g = bi.glcm(img, levels=8, offset=(0, 1))
    np.testing.assert_allclose(g.sum(), 1.0)  # normalized
    np.testing.assert_allclose(g, g.T)  # symmetric
    feats = bi.glcm_features(g)
    assert feats["contrast"] > 0 and 0.0 < feats["energy"] <= 1.0
    assert -1.0 - 1e-9 <= feats["correlation"] <= 1.0 + 1e-9
    # a horizontal ramp has a constant positive x-gradient in the interior
    ramp = np.tile(np.arange(6.0), (6, 1))
    gx, gy, mag = bi.sobel_gradient(ramp)
    assert np.all(gx[1:-1, 1:-1] > 0) and np.allclose(gy[1:-1, 1:-1], 0.0)
    np.testing.assert_allclose(mag[1:-1, 1:-1], np.abs(gx[1:-1, 1:-1]))


# --- depth_matching -----------------------------------------------------------


def _signal(seed: int = 1, n: int = 200) -> np.ndarray:
    return np.asarray(np.cumsum(np.random.default_rng(seed).normal(size=n)))


def test_dtw_self_and_shift() -> None:
    x = _signal()[:80]
    res = dm.dtw(x, x)
    assert res.distance == 0.0
    assert res.path[0] == (0, 0) and res.path[-1] == (79, 79)
    # a stretched copy warps back with small distance and a monotone path
    res2 = dm.dtw(x, np.repeat(x, 2)[:120], band=60)
    assert res2.distance >= 0.0
    ii = [i for i, _ in res2.path]
    jj = [j for _, j in res2.path]
    assert ii == sorted(ii) and jj == sorted(jj)
    np.testing.assert_allclose(dm.dtw(x, x, cost="absdiff").distance, 0.0)
    assert dm.dtw(x, x, root=True).distance == 0.0
    with pytest.raises(ValueError, match="cost"):
        dm.dtw(x, x, cost="bogus")


def test_xcorr_shift_wrap_and_trim() -> None:
    ref = _signal(2)
    tgt = np.roll(ref, 7)
    sr = dm.xcorr_shift(ref, tgt, max_lag=20, edge="wrap")
    assert sr.lag == -7  # roll(tgt,-7) restores ref
    np.testing.assert_allclose(sr.corr, 1.0, atol=1e-9)
    # trim edge recovers the same integer lag on a non-wrapped shift
    tgt2 = np.concatenate([np.full(5, ref[0]), ref[:-5]])
    st = dm.xcorr_shift(ref, tgt2, max_lag=20, edge="trim")
    assert st.lag == -5
    sc = dm.xcorr_shift(ref, tgt, max_lag=20, edge="wrap", return_curve=True)
    assert sc.curve is not None and sc.curve.size == 41
    with pytest.raises(ValueError, match="edge"):
        dm.xcorr_shift(ref, tgt, edge="bogus")


def test_xcorr_shift_depth_physical() -> None:
    depth = np.arange(0.0, 50.0, 0.125)
    curve = np.sin(depth / 3.0)
    core_depth = depth.copy()
    core_vals = np.sin((depth - 1.0) / 3.0)  # core feature at depth d matches log at d-1
    shift, corr = dm.xcorr_shift_depth(
        core_depth, core_vals, depth, curve, max_shift=2.0, step=0.125
    )
    # aligning needs core depths moved by -1 so (d+shift) matches the log grid
    np.testing.assert_allclose(shift, -1.0, atol=0.125)
    assert corr > 0.99
    # multi-curve target is accepted
    multi = np.column_stack([curve, np.cos(depth / 3.0)])
    s2, _ = dm.xcorr_shift_depth(core_depth, core_vals, depth, multi, use_abs_corr=True)
    np.testing.assert_allclose(abs(s2), 1.0, atol=0.25)


def test_apply_shifts_and_warp() -> None:
    x = np.arange(6.0)
    np.testing.assert_array_equal(dm.apply_integer_shift(x, 2)[2:], x[:4])
    assert np.isnan(dm.apply_integer_shift(x, 2)[:2]).all()
    # continuous depth shift then unshift is near-identity in the interior
    depth = np.linspace(0.0, 10.0, 101)
    vals = np.sin(depth)
    shifted = dm.apply_depth_shift(vals, depth, 0.5)
    back = dm.apply_depth_shift(shifted, depth, -0.5)
    np.testing.assert_allclose(back[5:-5], vals[5:-5], atol=1e-2)
    # warp_to_reference and path_depth_shifts on a self-path are trivial
    x2 = _signal(3)[:50]
    res = dm.dtw(x2, x2)
    aligned = dm.warp_to_reference(x2, res.path, n_ref=50)
    np.testing.assert_allclose(aligned, x2)
    depths, shifts = dm.path_depth_shifts(res.path, np.arange(50.0), np.arange(50.0))
    np.testing.assert_allclose(shifts, 0.0)


def test_rddtw_cow_local_shifts_run() -> None:
    ref = _signal(4)[:60]
    tgt = _signal(4)[:60] + 0.01
    r = dm.rddtw(ref, tgt, tau=4.0, lam=0.5)
    assert r.distance >= 0.0 and len(r.path) >= 60
    aligned, warp = dm.cow(ref, tgt, n_segments=6, slack=4)
    assert aligned.size == ref.size and warp.size == ref.size
    ls = dm.local_shifts(ref, tgt, window=30, step=15, max_lag=10)
    assert ls.ndim == 1 and np.all(np.abs(ls) <= 10)
