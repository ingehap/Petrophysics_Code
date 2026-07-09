"""Tests for petrolib.ml_stats: golden values, properties, and — the point
of the pilot — exact shadow equivalence against the article implementations
this module consolidates (LIBRARY_MERGE_PLAN.md sections 7-8)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from petrolib import ml_stats  # noqa: E402
from petrolib.testing import assert_matches_original  # noqa: E402

RNG = np.random.default_rng(20260709)


def load_article(relpath: str) -> ModuleType:
    """Import an article module by path (they live in non-package dirs)."""
    path = REPO_ROOT / relpath
    name = f"_article_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
# Shadow equivalence against the article implementations (the pilot's gate)
# --------------------------------------------------------------------------

Y_TRUE = RNG.normal(100.0, 20.0, size=80)
Y_PRED = Y_TRUE + RNG.normal(0.0, 5.0, size=80)


def test_equivalence_metrics_src2021_08() -> None:
    article = load_article("src2021_08/article5_synthetic_sonic_ml_contest.py")
    assert_matches_original(article.rmse, ml_stats.rmse, [(Y_TRUE, Y_PRED)])
    assert_matches_original(article.r2_score, ml_stats.r2_score, [(Y_TRUE, Y_PRED)])


def test_equivalence_metrics_src2021_12() -> None:
    article = load_article("src2021_12/article01_data_quality_ml.py")
    assert_matches_original(article.mae, ml_stats.mae, [(Y_TRUE, Y_PRED)])
    assert_matches_original(article.rmse, ml_stats.rmse, [(Y_TRUE, Y_PRED)])
    assert_matches_original(article.pearson, ml_stats.pearson_r, [(Y_TRUE, Y_PRED)])


def test_equivalence_scaling_src2018_12() -> None:
    article = load_article("src2018_12/article11_data_preconditioning.py")
    x = RNG.normal(50.0, 15.0, size=120)

    assert_matches_original(article.zscore, lambda v: ml_stats.zscore(v, eps=1e-12), [(x,)])
    assert_matches_original(article.minmax, lambda v: ml_stats.minmax(v, eps=1e-12), [(x,)])


def test_equivalence_classification_src2019_12() -> None:
    article = load_article("src2019_12/article8_ml_vuggy_facies_classifiers.py")
    y_true = RNG.integers(0, 2, size=60)
    y_pred = np.where(RNG.random(60) < 0.8, y_true, 1 - y_true)
    assert_matches_original(
        article.confusion_matrix,
        lambda t, p: ml_stats.confusion_matrix(t, p, labels=[0, 1]),
        [(y_true, y_pred)],
    )

    def replacement_metrics(t: np.ndarray, p: np.ndarray) -> tuple[float, ...]:
        return (ml_stats.accuracy(t, p), *ml_stats.precision_recall_f1(t, p))

    assert_matches_original(article.classification_metrics, replacement_metrics, [(y_true, y_pred)])


def test_equivalence_kmeans_src2018_10() -> None:
    article = load_article("src2018_10/article5_unsupervised_nmr_t1t2_fluid_volumes.py")
    data, _ = _two_blobs()
    weights = RNG.random(len(data)) + 0.5
    for kwargs in ({}, {"weights": weights}):
        labels_old, centroids_old = article.kmeans(data, 3, **kwargs)
        labels_new, centroids_new = ml_stats.kmeans(data, 3, **kwargs)
        np.testing.assert_array_equal(labels_new, labels_old)
        np.testing.assert_allclose(centroids_new, centroids_old, rtol=1e-12, atol=0.0)


def test_equivalence_silhouette_src2018_10() -> None:
    article = load_article("src2018_10/article11_hierarchical_rock_classification.py")
    data, truth = _two_blobs()
    assert_matches_original(article.silhouette, ml_stats.silhouette_score, [(data, truth)])


def test_equivalence_pca_src2016_08() -> None:
    article = load_article("src2016_08/article2_carbonate_nmr_rbf.py")
    distributions = np.abs(RNG.normal(size=(30, 12)))
    scores_old, components_old, mean_old = article.pca_reduce(distributions, 4)
    scores_new, components_new, mean_new, _ = ml_stats.pca(distributions, n_components=4)
    np.testing.assert_allclose(scores_new, scores_old, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(components_new, components_old, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(mean_new, mean_old, rtol=1e-12, atol=0.0)
