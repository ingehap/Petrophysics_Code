"""Tests for petrolib.ml_stats: golden values, properties, and shadow
equivalence against verbatim pre-migration article bodies — the permanent
record of what the B1 facades must reproduce (LIBRARY_MERGE_PLAN.md
sections 7-8)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import ml_stats  # noqa: E402
from petrolib.testing import assert_matches_original  # noqa: E402

RNG = np.random.default_rng(20260709)


# --------------------------------------------------------------------------
# Regression metrics
# --------------------------------------------------------------------------


def test_rmse_mae_rss_golden() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.5, 2.5, 5.0])
    assert ml_stats.rss(y_true, y_pred) == pytest.approx(1.5)
    assert ml_stats.rmse(y_true, y_pred) == pytest.approx(np.sqrt(1.5 / 4.0))
    assert ml_stats.mae(y_true, y_pred) == pytest.approx(0.5)
    # RMS of |e| always dominates the mean of |e|
    assert ml_stats.rmse(y_true, y_pred) >= ml_stats.mae(y_true, y_pred)


def test_mape_golden_and_flag() -> None:
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([11.0, 18.0])
    assert ml_stats.mape(y_true, y_pred) == pytest.approx(10.0)
    assert ml_stats.mape(y_true, y_pred, as_percent=False) == pytest.approx(0.10)
    assert not np.isfinite(ml_stats.mape([0.0, 1.0], [1.0, 1.0]))


def test_r2_and_pearson_identities() -> None:
    x = np.linspace(0.0, 10.0, 50)
    y = 3.0 * x - 2.0 + RNG.normal(0.0, 0.5, size=x.size)
    assert ml_stats.r2_score(y, y) == pytest.approx(1.0)
    assert ml_stats.pearson_r(x, 2.0 * x + 1.0) == pytest.approx(1.0)
    assert ml_stats.pearson_r(x, -x) == pytest.approx(-1.0)
    # R² of the best-fit line equals the squared correlation coefficient
    slope, intercept = ml_stats.fit_line(x, y)
    fitted = slope * x + intercept
    assert ml_stats.r2_score(y, fitted) == pytest.approx(ml_stats.pearson_r(x, y) ** 2)


def test_metric_degenerate_inputs_are_nan_not_biased() -> None:
    constant = np.ones(5)
    varying = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert not np.isfinite(ml_stats.r2_score(constant, varying))
    assert np.isnan(ml_stats.pearson_r(constant, varying))
    # the explicit additive guard reproduces the articles' +1e-12 idiom
    assert np.isfinite(ml_stats.r2_score(constant, varying, eps=1e-12))
    assert ml_stats.pearson_r(constant, varying, eps=1e-12) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------


def test_confusion_matrix_binary_convention_and_multiclass() -> None:
    y_true = [0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1, 1]
    matrix = ml_stats.confusion_matrix(y_true, y_pred)
    # [[TN, FP], [FN, TP]] for 0/1 labels
    assert matrix.tolist() == [[2, 1], [1, 3]]
    assert matrix.sum() == len(y_true)
    multi = ml_stats.confusion_matrix(["a", "b", "c"], ["a", "c", "c"], labels=["a", "b", "c"])
    assert multi.tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]


def test_precision_recall_f1_golden_and_zero_division() -> None:
    y_true = [0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1, 1]
    precision, recall, f1 = ml_stats.precision_recall_f1(y_true, y_pred)
    assert precision == pytest.approx(3.0 / 4.0)
    assert recall == pytest.approx(3.0 / 4.0)
    assert f1 == pytest.approx(0.75)
    assert ml_stats.accuracy(y_true, y_pred) == pytest.approx(5.0 / 7.0)
    # nothing predicted positive -> zero_division value, not a crash
    precision, recall, f1 = ml_stats.precision_recall_f1([1, 1], [0, 0])
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------


def test_zscore_whole_and_columnwise() -> None:
    x = RNG.normal(5.0, 3.0, size=(40, 3)) * np.array([1.0, 10.0, 100.0])
    whole = ml_stats.zscore(x)
    assert whole.mean() == pytest.approx(0.0, abs=1e-12)
    assert whole.std() == pytest.approx(1.0)
    columns = ml_stats.zscore(x, axis=0)
    np.testing.assert_allclose(columns.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(columns.std(axis=0), 1.0, rtol=1e-12)
    assert np.all(np.isnan(ml_stats.zscore(np.ones(4))))
    assert np.all(ml_stats.zscore(np.ones(4), eps=1e-12) == 0.0)


def test_minmax_and_affine_rescale() -> None:
    x = np.array([2.0, 4.0, 6.0])
    np.testing.assert_allclose(ml_stats.minmax(x), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(ml_stats.minmax(x, lo=-1.0, hi=1.0), [-1.0, 0.0, 1.0])
    # reference-range normalization hits the target percentiles exactly
    rescaled = ml_stats.affine_rescale(
        np.array([20.0, 60.0, 100.0]), src_lo=20.0, src_hi=100.0, dst_lo=15.0, dst_hi=120.0
    )
    np.testing.assert_allclose(rescaled, [15.0, 67.5, 120.0])


# --------------------------------------------------------------------------
# Fit wrappers
# --------------------------------------------------------------------------


def test_fit_line_and_powerlaw_recover_truth() -> None:
    x = np.linspace(1.0, 9.0, 30)
    slope, intercept = ml_stats.fit_line(x, 2.5 * x - 4.0)
    assert slope == pytest.approx(2.5)
    assert intercept == pytest.approx(-4.0)
    coefficient, exponent = ml_stats.fit_powerlaw(x, 3.0 * x**-1.7)
    assert coefficient == pytest.approx(3.0)
    assert exponent == pytest.approx(-1.7)


def test_ols_separates_intercept() -> None:
    X = RNG.normal(size=(60, 2))
    y = X @ np.array([1.5, -2.0]) + 0.75
    coef, intercept = ml_stats.ols(X, y)
    np.testing.assert_allclose(coef, [1.5, -2.0], rtol=1e-10)
    assert intercept == pytest.approx(0.75)
    np.testing.assert_allclose(ml_stats.predict_linear(X, coef, intercept), y, rtol=1e-10)
    coef_no, intercept_no = ml_stats.ols(X, y, intercept=False)
    assert intercept_no == 0.0
    assert coef_no.shape == (2,)
    # 1-D feature convenience
    x1 = np.linspace(0.0, 1.0, 20)
    coef1, b1 = ml_stats.ols(x1, 4.0 * x1 + 1.0)
    assert coef1[0] == pytest.approx(4.0)
    assert b1 == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Clustering / decomposition
# --------------------------------------------------------------------------


def _two_blobs() -> tuple[np.ndarray, np.ndarray]:
    a = RNG.normal(0.0, 0.3, size=(25, 2))
    b = RNG.normal(5.0, 0.3, size=(25, 2))
    return np.vstack([a, b]), np.repeat([0, 1], 25)


def test_kmeans_separates_blobs_and_is_deterministic() -> None:
    data, truth = _two_blobs()
    labels_1, centroids_1 = ml_stats.kmeans(data, 2)
    labels_2, centroids_2 = ml_stats.kmeans(data, 2)
    np.testing.assert_array_equal(labels_1, labels_2)
    np.testing.assert_array_equal(centroids_1, centroids_2)
    # each true blob lands in exactly one cluster
    assert len(set(labels_1[truth == 0])) == 1
    assert len(set(labels_1[truth == 1])) == 1
    assert ml_stats.silhouette_score(data, labels_1) > 0.8


def test_kmeans_weights_move_centroids() -> None:
    data = np.array([[0.0], [1.0], [10.0], [11.0]])
    weights = np.array([1.0, 100.0, 1.0, 1.0])
    _, centroids = ml_stats.kmeans(data, 2, weights=weights)
    # the heavy point dominates its cluster's centroid
    assert np.min(np.abs(centroids - 1.0)) < 0.05


def test_silhouette_degenerate_single_cluster() -> None:
    data = RNG.normal(size=(10, 2))
    assert ml_stats.silhouette_score(data, np.zeros(10, dtype=int)) == 0.0


def test_pca_variance_and_reconstruction() -> None:
    data = RNG.normal(size=(200, 3)) * np.array([10.0, 1.0, 0.1])
    scores, components, mean, explained = ml_stats.pca(data)
    assert explained.sum() == pytest.approx(1.0)
    assert explained[0] > 0.9
    # first axis is the high-variance direction
    assert abs(components[0, 0]) == pytest.approx(1.0, abs=0.02)
    np.testing.assert_allclose(scores @ components + mean, data, atol=1e-10)
    scores_2, components_2, _, explained_2 = ml_stats.pca(data, n_components=2)
    assert scores_2.shape == (200, 2)
    assert components_2.shape == (2, 3)
    assert explained_2.shape == (2,)


# --------------------------------------------------------------------------
# Shadow equivalence against the pre-migration article bodies
#
# The functions below are VERBATIM copies of the local implementations that
# the B1 adoption PR replaced with petrolib delegations (one per distinct
# variant; byte-identical duplicates in other files are noted).  They pin the
# migration contract permanently: the canonical function must keep matching
# the historical body at rtol=1e-12 on realistic inputs.
# --------------------------------------------------------------------------

Y_TRUE = RNG.normal(100.0, 20.0, size=80)
Y_PRED = Y_TRUE + RNG.normal(0.0, 5.0, size=80)


def _original_rmse_2021_08(y_true, y_pred):  # src2021_08/article5 (same math in ~25 files)
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d**2)))


def _original_r2_2021_08(y_true, y_pred):  # src2021_08/article5
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def _original_mae_2021_12(y_true, y_pred):  # src2021_12/article01
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def _original_pearson_2021_12(x, y):  # src2021_12/article01 (conditional guard)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    d = np.sqrt(np.sum(xc**2) * np.sum(yc**2))
    return float(np.sum(xc * yc) / d) if d > 1e-12 else 0.0


def _original_correlation_2019_12(y, yhat):  # src2019_12/article7 (np.corrcoef path)
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return float(np.corrcoef(y, yhat)[0, 1])


def _original_aape_2019_12(y, yhat):  # src2019_12/article7
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return float(np.mean(np.abs((y - yhat) / y)) * 100.0)


def test_equivalence_regression_metrics() -> None:
    assert_matches_original(_original_rmse_2021_08, ml_stats.rmse, [(Y_TRUE, Y_PRED)])
    assert_matches_original(_original_r2_2021_08, ml_stats.r2_score, [(Y_TRUE, Y_PRED)])
    assert_matches_original(_original_mae_2021_12, ml_stats.mae, [(Y_TRUE, Y_PRED)])
    assert_matches_original(_original_pearson_2021_12, ml_stats.pearson_r, [(Y_TRUE, Y_PRED)])
    assert_matches_original(_original_aape_2019_12, ml_stats.mape, [(Y_TRUE, Y_PRED)])
    # np.corrcoef normalizes through a different float path than the centered
    # sums; observed <2e-15 relative — within the rtol=1e-12 gate.
    assert_matches_original(_original_correlation_2019_12, ml_stats.pearson_r, [(Y_TRUE, Y_PRED)])


def _original_zscore_2018_12(x):  # src2018_12/article11 (whole-array, +1e-12)
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)


def _original_minmax_2018_12(x):  # src2018_12/article11
    x = np.asarray(x, float)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def _original_zscore_norm_2021_08(x):  # src2021_08/article5 (column-wise, +1e-12)
    x = np.asarray(x, float)
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)


def _original_minmax_norm_2021_08(x):  # src2021_08/article5 (column-wise)
    x = np.asarray(x, float)
    lo, hi = x.min(axis=0), x.max(axis=0)
    return (x - lo) / (hi - lo + 1e-12)


def _original_zscore_2020_10(x):  # src2020_10/article3 (conditional guard)
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def test_equivalence_scaling() -> None:
    flat = RNG.normal(50.0, 15.0, size=120)
    table = RNG.normal(5.0, 3.0, size=(40, 3)) * np.array([1.0, 10.0, 100.0])
    assert_matches_original(
        _original_zscore_2018_12, lambda v: ml_stats.zscore(v, eps=1e-12), [(flat,)]
    )
    assert_matches_original(
        _original_minmax_2018_12, lambda v: ml_stats.minmax(v, eps=1e-12), [(flat,)]
    )
    assert_matches_original(
        _original_zscore_norm_2021_08,
        lambda v: ml_stats.zscore(v, axis=0, eps=1e-12),
        [(table,)],
    )
    assert_matches_original(
        _original_minmax_norm_2021_08,
        lambda v: ml_stats.minmax(v, axis=0, eps=1e-12),
        [(table,)],
    )
    assert_matches_original(_original_zscore_2020_10, ml_stats.zscore, [(flat,)])


def _original_confusion_2019_12(y_true, y_pred):  # src2019_12/article8
    yt = np.asarray(y_true, int)
    yp = np.asarray(y_pred, int)
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))
    return np.array([[tn, fp], [fn, tp]])


def _original_classification_metrics_2019_12(y_true, y_pred):  # src2019_12/article8
    cm = _original_confusion_2019_12(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / cm.sum()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


def test_equivalence_classification() -> None:
    y_true = RNG.integers(0, 2, size=60)
    y_pred = np.where(RNG.random(60) < 0.8, y_true, 1 - y_true)
    assert_matches_original(
        _original_confusion_2019_12,
        lambda t, p: ml_stats.confusion_matrix(t, p, labels=[0, 1]),
        [(y_true, y_pred)],
    )

    def replacement(t, p):
        return (ml_stats.accuracy(t, p), *ml_stats.precision_recall_f1(t, p))

    assert_matches_original(
        _original_classification_metrics_2019_12, replacement, [(y_true, y_pred)]
    )


def _original_kmeans_2018_10(X, k, weights=None, iters=100):  # src2018_10/article5 (weighted)
    X = np.asarray(X, float)
    w = np.ones(len(X)) if weights is None else np.asarray(weights, float)
    idx = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
    for _ in range(1, k):
        d = np.min([np.linalg.norm(X - X[j], axis=1) for j in idx], axis=0)
        idx.append(int(np.argmax(d)))
    centers = X[idx].copy()
    labels = np.zeros(len(X), int)
    for _ in range(iters):
        D = np.stack([np.linalg.norm(X - c, axis=1) for c in centers], axis=1)
        new = D.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = np.average(X[m], axis=0, weights=w[m])
    return labels, centers


def _original_kmeans_2020_08(X, k, iters=100):  # src2020_08/article4 (.mean(0) update;
    # byte-identical copies in src2020_10/article3 and article7)
    X = np.asarray(X, float)
    idx = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
    for _ in range(1, k):
        d = np.min([np.linalg.norm(X - X[j], axis=1) for j in idx], axis=0)
        idx.append(int(np.argmax(d)))
    centers = X[idx].copy()
    labels = np.zeros(len(X), int)
    for _ in range(iters):
        d = np.stack([np.linalg.norm(X - c, axis=1) for c in centers], axis=1)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            if np.any(labels == c):
                centers[c] = X[labels == c].mean(0)
    return labels, centers


def _original_silhouette_2018_10(X, labels):  # src2018_10/article11 (same math in
    # src2020_08/article4 and src2020_10/article7)
    X = np.asarray(X, float)
    labels = np.asarray(labels, int)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        own[i] = False
        a = D[i, own].mean() if own.any() else 0.0
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def _original_pca_reduce_2016_08(distributions, n_components):  # src2016_08/article2
    x = np.asarray(distributions, float)
    mean = x.mean(axis=0)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:n_components]
    scores = xc @ components.T
    return scores, components, mean


def test_equivalence_kmeans_weighted_and_unweighted() -> None:
    data, _ = _two_blobs()
    weights = RNG.random(len(data)) + 0.5
    for kwargs in ({}, {"weights": weights}):
        labels_old, centroids_old = _original_kmeans_2018_10(data, 3, **kwargs)
        labels_new, centroids_new = ml_stats.kmeans(data, 3, **kwargs)
        np.testing.assert_array_equal(labels_new, labels_old)
        np.testing.assert_allclose(centroids_new, centroids_old, rtol=1e-12, atol=0.0)
    labels_old, centroids_old = _original_kmeans_2020_08(data, 3)
    labels_new, centroids_new = ml_stats.kmeans(data, 3)
    np.testing.assert_array_equal(labels_new, labels_old)
    np.testing.assert_allclose(centroids_new, centroids_old, rtol=1e-12, atol=0.0)


def test_equivalence_silhouette() -> None:
    data, truth = _two_blobs()
    assert_matches_original(
        _original_silhouette_2018_10, ml_stats.silhouette_score, [(data, truth)]
    )


def test_equivalence_pca() -> None:
    distributions = np.abs(RNG.normal(size=(30, 12)))
    scores_old, components_old, mean_old = _original_pca_reduce_2016_08(distributions, 4)
    scores_new, components_new, mean_new, _ = ml_stats.pca(distributions, n_components=4)
    np.testing.assert_allclose(scores_new, scores_old, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(components_new, components_old, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(mean_new, mean_old, rtol=1e-12, atol=0.0)


# --- B2 variants -----------------------------------------------------------


def _original_r_squared_2020_04(y, yhat):  # src2020_04/article4 (conditional guard;
    # same shape in src2025_12/dl_permeability)
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _original_r_squared_2024_10(y_true, y_pred):  # src2024_10/rddtw (+1e-12;
    # src2025_10/a5 uses +1e-30)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _original_standardize_2022_02(x):  # src2022_02/article2 and article3
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


def _original_mape_masked_2025_12(y_true, y_pred):  # src2025_12/dl_permeability
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _original_pearson_nanmask_2023_08(x, y):  # src2023_08/article6
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float("nan")
    xbar, ybar = x.mean(), y.mean()
    num = np.sum((x - xbar) * (y - ybar))
    den = np.sqrt(np.sum((x - xbar) ** 2) * np.sum((y - ybar) ** 2))
    return float(num / den) if den > 0 else float("nan")


def test_equivalence_b2_variants() -> None:
    assert_matches_original(_original_r_squared_2020_04, ml_stats.r2_score, [(Y_TRUE, Y_PRED)])
    assert_matches_original(
        _original_r_squared_2024_10,
        lambda t, p: ml_stats.r2_score(t, p, eps=1e-12),
        [(Y_TRUE, Y_PRED)],
    )
    flat = RNG.normal(50.0, 15.0, size=120)
    assert_matches_original(_original_standardize_2022_02, ml_stats.zscore, [(flat,)])
    with_zeros = np.where(RNG.random(80) < 0.1, 0.0, Y_TRUE)
    assert_matches_original(
        _original_mape_masked_2025_12,
        lambda t, p: ml_stats.mape(t[t != 0], p[t != 0]),
        [(with_zeros, Y_PRED)],
    )
    with_nans = np.where(RNG.random(80) < 0.1, np.nan, Y_TRUE)
    assert_matches_original(
        _original_pearson_nanmask_2023_08,
        lambda x, y: (lambda m: ml_stats.pearson_r(x[m], y[m]) if m.sum() >= 2 else float("nan"))(
            np.isfinite(x) & np.isfinite(y)
        ),
        [(with_nans, Y_PRED)],
    )


# --- Hazard call sites (one-file PRs; LIBRARY_MERGE_PLAN.md section 9) ------


def _original_r2_reversed_2022_10(y_pred, y_obs):  # src2022_10/article2:
    # the article's r2 takes PREDICTIONS first
    ss_res = float(np.sum((y_obs - y_pred) ** 2))
    ss_tot = float(np.sum((y_obs - y_obs.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def _original_fit_linear_2021_06(X, y):  # src2021_06/article3: intercept-first beta
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def _original_predict_linear_2021_06(X, beta):  # src2021_06/article3
    return np.column_stack([np.ones(len(X)), X]) @ beta


def _original_correlation_textbook_2021_06(y_true, y_pred):  # src2021_06/article3
    y = np.asarray(y_true, float)
    yh = np.asarray(y_pred, float)
    n = len(y)
    num = n * np.sum(y * yh) - np.sum(y) * np.sum(yh)
    den = np.sqrt((n * np.sum(y**2) - np.sum(y) ** 2) * (n * np.sum(yh**2) - np.sum(yh) ** 2))
    return float(num / den)


def _original_minmax_pm1_2021_06(x, lo=-1.0, hi=1.0):  # src2021_06/article3
    x = np.asarray(x, float)
    mn, mx = x.min(axis=0), x.max(axis=0)
    return lo + (hi - lo) * (x - mn) / (mx - mn + 1e-12)


def test_hazard_r2_reversed_arguments() -> None:
    correct = _original_r2_reversed_2022_10(Y_PRED, Y_TRUE)
    # the facade maps the historical (y_pred, y_obs) order onto keywords
    assert ml_stats.r2_score(Y_TRUE, Y_PRED) == pytest.approx(correct, rel=1e-12)
    # the trap the explicit mapping prevents: migrating positionally would
    # compute ss_tot from the predictions and silently change the value
    naive = ml_stats.r2_score(Y_PRED, Y_TRUE)
    assert naive != pytest.approx(correct, rel=1e-6)


def test_hazard_fit_linear_intercept_first() -> None:
    rng = np.random.default_rng(11)
    X = rng.uniform(-1, 1, size=(600, 6))
    truth = np.array([3500.0, 200.0, -150.0, 80.0, 40.0, -60.0, 25.0])
    y = truth[0] + X @ truth[1:] + rng.normal(0, 50.0, 600)
    beta_old = _original_fit_linear_2021_06(X, y)
    coef, intercept = ml_stats.ols(X, y)
    beta_new = np.concatenate([[intercept], coef])
    # the design-matrix column reorder changes the lstsq float path;
    # measured <3e-14 relative at the article's data scales
    np.testing.assert_allclose(beta_new, beta_old, rtol=1e-12, atol=0.0)
    pred_old = _original_predict_linear_2021_06(X, beta_old)
    pred_new = ml_stats.predict_linear(X, beta_new[1:], float(beta_new[0]))
    np.testing.assert_allclose(pred_new, pred_old, rtol=1e-12, atol=0.0)


def test_hazard_textbook_correlation_and_minmax() -> None:
    rng = np.random.default_rng(12)
    vp = rng.normal(4000.0, 300.0, 400)  # the article's velocity scale
    vp_hat = vp + rng.normal(0, 150.0, 400)
    assert_matches_original(
        _original_correlation_textbook_2021_06, ml_stats.pearson_r, [(vp, vp_hat)]
    )
    table = rng.uniform(0.0, 500.0, size=(80, 6))
    assert_matches_original(
        _original_minmax_pm1_2021_06,
        lambda v: ml_stats.minmax(v, axis=0, lo=-1.0, hi=1.0, eps=1e-12),
        [(table,)],
    )
