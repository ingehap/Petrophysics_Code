"""Tests for petrolib.data_qc_io: cleaning, scaling, filtering, signal,
synthetic data, and the wellbore JSON container -- known values, article
anchors, and error paths."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import data_qc_io as dq  # noqa: E402

# --- clean ----------------------------------------------------------------------


def test_sentinels_and_impute() -> None:
    x = np.array([1.0, -999.25, 3.0, -9999.0, 5.0])
    out = dq.clean.sentinels_to_nan(x)
    assert np.isnan(out[[1, 3]]).all() and np.array_equal(out[[0, 2, 4]], [1.0, 3.0, 5.0])
    # linear interpolation over the gap, flat extrapolation at the ends
    filled = dq.clean.impute_gaps(np.array([np.nan, 1.0, np.nan, 3.0, np.nan]))
    np.testing.assert_allclose(filled, [1.0, 1.0, 2.0, 3.0, 3.0])
    # explicit non-uniform index weights the interpolation
    filled = dq.clean.impute_gaps(np.array([0.0, np.nan, 4.0]), index=[0.0, 3.0, 4.0])
    np.testing.assert_allclose(filled, [0.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="no finite"):
        dq.clean.impute_gaps(np.array([np.nan, np.nan]))


def test_outlier_mask_and_despike() -> None:
    x = np.concatenate([np.zeros(50), [100.0]])
    for method in ("zscore", "iqr", "mad"):
        mask = dq.clean.outlier_mask(x, method)
        assert mask[-1] and mask[:50].sum() == 0
    # one-sided: an upper screen never flags low values
    lo = np.concatenate([np.ones(50), [-100.0]])
    assert not dq.clean.outlier_mask(lo, "mad", side="upper").any()
    assert dq.clean.outlier_mask(lo, "mad", side="lower")[-1]
    with pytest.raises(ValueError, match="method"):
        dq.clean.outlier_mask(x, "bogus")
    with pytest.raises(ValueError, match="side"):
        dq.clean.outlier_mask(x, "mad", side="above")
    # despike: interp bridges the spike, median uses the global median
    y = np.array([1.0, 1.0, 50.0, 1.0, 1.0])
    np.testing.assert_allclose(dq.clean.despike(y), np.ones(5))
    np.testing.assert_allclose(dq.clean.despike(y, replace="median"), np.ones(5))
    assert np.isnan(dq.clean.despike(y, replace="nan")[2])
    with pytest.raises(ValueError, match="replace"):
        dq.clean.despike(y, replace="zero")


def test_closure_renormalize_discrepancy() -> None:
    np.testing.assert_allclose(dq.clean.closure_residual(0.3, 0.5, 0.2), 0.0, atol=1e-15)
    np.testing.assert_allclose(float(dq.clean.closure_residual(0.6, target=0.5)), 0.1)
    with pytest.raises(ValueError, match="at least one"):
        dq.clean.closure_residual()
    closed = dq.clean.renormalize([[2.0, 2.0], [1.0, 3.0]])
    np.testing.assert_allclose(closed.sum(axis=-1), [1.0, 1.0])
    # deviation-indicator anchor (src2016_02 Eq. 6): 2*0.2/2.2
    np.testing.assert_allclose(dq.clean.relative_discrepancy(1.0, 1.2), 2.0 * 0.2 / 2.2)


# --- scale ----------------------------------------------------------------------


def test_scale() -> None:
    x = np.linspace(0.0, 10.0, 101)
    # explicit endpoints: affine two-point map hits the reference endpoints
    out = dq.scale.normalize_to_reference(x, 100.0, 200.0, in_lo=0.0, in_hi=10.0)
    np.testing.assert_allclose([out[0], out[-1]], [100.0, 200.0])
    # pct=None uses min/max, so the full range maps exactly
    out = dq.scale.normalize_to_reference(x, 0.0, 1.0, pct=None)
    np.testing.assert_allclose([out.min(), out.max()], [0.0, 1.0])
    # default 5/95 percentiles: the p5/p95 samples land on the endpoints
    out = dq.scale.normalize_to_reference(x, 0.0, 1.0)
    np.testing.assert_allclose(out[5], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[95], 1.0, atol=1e-12)
    y = dq.scale.match_moments(np.random.default_rng(0).normal(5.0, 2.0, 500), 75.0, 20.0)
    np.testing.assert_allclose([y.mean(), y.std()], [75.0, 20.0])


# --- filt -----------------------------------------------------------------------


def test_smooth() -> None:
    x = np.ones(30)
    x[15] = 31.0
    # boxcar: mean-preserving pulse spread; window 1 is a no-op copy
    np.testing.assert_allclose(dq.filt.smooth(x, 3)[14:17], [11.0, 11.0, 11.0])
    assert np.array_equal(dq.filt.smooth(x, 1), x)
    # median kills the single spike outright
    np.testing.assert_allclose(dq.filt.smooth(x, 3, "median"), np.ones(30))
    # gaussian: unit-normalized kernel preserves a constant interior
    g = dq.filt.smooth(np.full(50, 7.0), kind="gaussian", sigma=2.0)
    np.testing.assert_allclose(g[10:40], 7.0)
    with pytest.raises(ValueError, match="sigma"):
        dq.filt.smooth(x, kind="gaussian")
    with pytest.raises(ValueError, match="odd"):
        dq.filt.smooth(x, 4, "median")
    with pytest.raises(ValueError, match="kind"):
        dq.filt.smooth(x, 5, "hann")


def test_moving_stat() -> None:
    x = np.arange(10.0)
    assert np.array_equal(dq.filt.moving_stat(np.full(20, 3.0), 5, "std"), np.zeros(20))
    # centered window mean of a ramp is the ramp (away from edges)
    np.testing.assert_allclose(dq.filt.moving_stat(x, 3, "mean")[1:-1], x[1:-1])
    # causal window lags: trailing mean at i=9 is mean(7,8,9)=8
    np.testing.assert_allclose(dq.filt.moving_stat(x, 3, "mean", center=False)[-1], 8.0)
    # cv guards a zero mean with 0
    assert dq.filt.moving_stat(np.zeros(5), 3, "cv")[2] == 0.0
    np.testing.assert_allclose(
        dq.filt.moving_stat(x, 3, "var")[5], dq.filt.moving_stat(x, 3, "std")[5] ** 2
    )
    assert dq.filt.moving_stat(np.array([0.0, 4.0, 0.0]), 3, "mad")[1] > 0
    with pytest.raises(ValueError, match="stat"):
        dq.filt.moving_stat(x, 3, "median")


def test_tool_response_and_median2d() -> None:
    # unit-area Gaussian kernel preserves a constant log interior
    out = dq.filt.tool_response(np.full(100, 2.5), dz=0.1, fwhm=0.5)
    np.testing.assert_allclose(out[20:80], 2.5)
    # a step gets smoothed monotonically across the edge
    step = np.repeat([0.0, 1.0], 50)
    sm = dq.filt.tool_response(step, dz=0.1, fwhm=0.8)
    assert 0.0 < sm[49] < sm[50] < sm[51] < 1.0
    # median filter removes salt-and-pepper without moving the background
    img = np.full((8, 8), 5.0)
    img[3, 4] = 500.0
    np.testing.assert_allclose(dq.filt.median_filter2d(img), np.full((8, 8), 5.0))
    with pytest.raises(ValueError, match="2-D"):
        dq.filt.median_filter2d(np.zeros(5))
    with pytest.raises(ValueError, match="odd"):
        dq.filt.median_filter2d(img, size=4)


def test_window_features_and_boundaries() -> None:
    x = np.arange(20.0)
    feats = dq.filt.window_features(x, 5)
    assert feats.shape == (20, 5)
    np.testing.assert_allclose(feats[10], [8.0, 9.0, 10.0, 11.0, 12.0])
    np.testing.assert_allclose(feats[0], [0.0, 0.0, 0.0, 1.0, 2.0])  # edge-padded
    both = dq.filt.window_features(np.column_stack([x, 2.0 * x]), 5)
    assert both.shape == (20, 10)
    with pytest.raises(ValueError, match="odd"):
        dq.filt.window_features(x, 4)
    # two clean steps -> boundaries at (or next to) the true edges
    curve = np.concatenate([np.full(40, 10.0), np.full(30, 80.0), np.full(50, 30.0)])
    curve += np.random.default_rng(1).normal(0.0, 0.5, curve.size)
    idx = dq.filt.detect_bed_boundaries(curve, 11)
    assert any(abs(i - 40) <= 2 for i in idx) and any(abs(i - 70) <= 2 for i in idx)


# --- signal ---------------------------------------------------------------------


def test_signal() -> None:
    sig = np.full(100, 2.0)
    # P_sig=4, sigma=0.2 -> 10*log10(4/0.04) = 20 dB
    np.testing.assert_allclose(dq.signal.snr_db(sig, noise_std=0.2), 20.0)
    assert dq.signal.snr_db(sig, noise_std=0.0) == float("inf")
    np.testing.assert_allclose(dq.signal.snr_db(sig, noise=np.full(100, 0.2)), 20.0)
    with pytest.raises(ValueError, match="exactly one"):
        dq.signal.snr_db(sig)
    with pytest.raises(ValueError, match="exactly one"):
        dq.signal.snr_db(sig, noise=sig, noise_std=1.0)
    np.testing.assert_allclose(dq.signal.stack_gain(16), 4.0)
    # block means over n=3; the trailing partial block (7.0) is dropped
    out = dq.signal.block_stack(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), 3)
    np.testing.assert_allclose(out, [2.0, 5.0])
    two = dq.signal.block_stack(np.arange(12.0).reshape(2, 6), 3, axis=-1)
    np.testing.assert_allclose(two, [[1.0, 4.0], [7.0, 10.0]])
    with pytest.raises(ValueError, match=">= 1"):
        dq.signal.block_stack(sig, 0)
    # deterministic default rng; sigma scales with the mean absolute level
    x = np.full(1000, 10.0)
    n1 = dq.signal.add_gaussian_noise(x, 0.05)
    n2 = dq.signal.add_gaussian_noise(x, 0.05)
    assert np.array_equal(n1, n2)
    np.testing.assert_allclose(np.std(n1 - x), 0.5, rtol=0.1)


# --- synth ----------------------------------------------------------------------


def test_synth_logs() -> None:
    curve, bounds = dq.synth.blocky_log(300, n_beds=6, rng=2)
    assert curve.shape == (300,) and bounds.shape == (5,)
    assert np.all(np.diff(bounds) > 0)
    with pytest.raises(ValueError, match="n_beds"):
        dq.synth.blocky_log(300, n_beds=0)
    logs, bp, fac = dq.synth.log_suite()
    assert set(logs) == set(dq.synth.LOG_SUITE_CURVES) and len(bp) == 3
    assert logs["GR"].shape == (600,) and set(np.unique(fac)) <= {0, 1, 2, 3}
    # shale (facies 1) has the highest mean GR of the default properties
    assert logs["GR"][fac == 1].mean() > logs["GR"][fac == 0].mean()
    with pytest.raises(ValueError, match="n_facies"):
        dq.synth.log_suite(n_facies=9)
    ref, other = dq.synth.shifted_pair(200, 7)
    np.testing.assert_allclose(other[:-7], ref[7:])
    ref, other = dq.synth.shifted_pair(200, -5)
    np.testing.assert_allclose(other[5:], ref[:-5])


def test_synth_spectra_and_images() -> None:
    t2 = np.logspace(-1.0, 4.0, 200)
    spec = dq.synth.gaussian_mixture_spectrum(t2, [1.0, 100.0], [0.6, 0.4], [0.3, 0.3])
    # peaks land at the component centres on the log axis
    assert abs(np.log10(t2[np.argmax(spec)]) - 0.0) < 0.05
    np.testing.assert_allclose(spec.max(), 0.6, rtol=1e-3)
    noisy = dq.synth.gaussian_mixture_spectrum(t2, [1.0], [0.5], [0.3], noise=0.01, rng=0)
    assert noisy.min() >= 0.0
    with pytest.raises(ValueError):
        dq.synth.gaussian_mixture_spectrum(t2, [1.0, 2.0], [0.5], [0.3])
    cube = dq.synth.sphere_pack_volume((24, 24, 24), (3, 5), 0.4, rng=0)
    phi = 1.0 - cube.mean()
    assert cube.dtype == np.uint8 and phi <= 0.4
    img, mask = dq.synth.disk_image(64, 20, rng=4)
    assert img.shape == (64, 64) and mask.dtype == bool
    assert 0.0 <= img.min() and img.max() <= 1.0
    # disks are darker than the background
    assert img[mask].mean() < img[~mask].mean()


# --- io -------------------------------------------------------------------------


def test_wellbore_data(tmp_path: Path) -> None:
    w = dq.WellboreData("Well-A", uwi="00/01")
    w.add_channel("GR", np.linspace(10.0, 90.0, 8), "gAPI", ["MD"])
    w.add_channel("UDAR", np.ones((4, 3, 2)), "ohm.m", ["MD", "azimuth", "DOI"])
    with pytest.raises(ValueError, match="axes count"):
        w.add_channel("BAD", np.ones((2, 2)), "-", ["MD"])
    s = w.to_json(str(tmp_path / "well.json"))
    assert (tmp_path / "well.json").read_text() == s
    w2 = dq.WellboreData.from_json(s)
    assert w2.meta == w.meta
    assert w2.channels["UDAR"]["shape"] == (4, 3, 2)
    np.testing.assert_array_equal(w2.channels["GR"]["data"], w.channels["GR"]["data"])
