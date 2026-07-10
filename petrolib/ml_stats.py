"""Statistics and numpy-only machine-learning helpers.

The pilot domain of the library migration (LIBRARY_MERGE_PLAN.md section 7):
regression/agreement metrics, classification metrics, feature scaling,
least-squares fit wrappers, k-means clustering, silhouette validation and
PCA — the most-duplicated pure-math family in the repository (metrics alone
are re-implemented in ~30 article modules).

Hazards this module's API is designed around (plan section 9):

- **R vs R².**  Article functions named ``pearson`` / ``correlation`` /
  ``correlation_coefficient`` return R while ``r_squared`` / ``r2_score``
  return R².  Here the names are unambiguous: :func:`pearson_r` returns R,
  :func:`r2_score` returns R².
- **Argument order.**  ``src2022_10``'s ``r2(y_pred, y_obs)`` reverses the
  usual order.  Canonical metrics take ``(y_true, y_pred)``; facades map
  their historical order onto these names once, in review.
- **Zero-division guards.**  Articles guard denominators three ways: not at
  all (NaN), an additive ``+1e-12``, or a conditional fallback value.  The
  additive guard is exposed as an explicit ``eps=`` parameter (default 0.0 =
  pure math); conditional-fallback variants keep their one-line branch in
  the article facade.
- **Intercept position.**  ``fit_linear`` (src2021_06/src2021_08) prepends
  the ones column (``beta[0]`` = intercept) while ``ols_fit`` (src2018_12)
  appends it (``coef[-1]`` = intercept).  :func:`ols` returns
  ``(coef, intercept)`` separately so the position trap cannot exist.

Everything is numpy-only and deterministic (k-means uses the repository's
farthest-point init, not a random one).  The single-hidden-layer NN trainers
and non-farthest k-means inits found in some articles stay article-local
until an adoption PR proves an exact-equivalence mapping.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2014_06/article4 -- Article 4: Method for Predicting Permeability of Complex Carbonate
  Reservoirs Using NMR Logging Measurements Willian Trevizan, Paulo Neto, Bernardo Coutinho,
  Vinicius F. Machado, Edmilson H. Rios, Songhua Chen, Wei Shao, Pedro Romero (2014). Petrophysics
  Vol. 55, No. 3 (June 2014), pp. 240-252. DOI: none assigned (this issue predates SPWLA DOI
  assignment).
src2014_08/article3 -- Article 3: Multiphase Flow in Porous Rock Imaged Under Dynamic Flow
  Conditions with Fast X-Ray Computed Microtomography S. Berg, R. Armstrong, H. Ott, A. Georgiadis,
  S. A. Klapp, A. Schwing, R. Neiteler, N. Brussee, A. Makurat, L. Leu, F. Enzmann, J.-O. Schwarz,
  M. Wolf, F. Khan, M. Kersten, S. Irvine, M. Stampanoni (2014). Petrophysics Vol. 55, No. 4
  (August 2014), pp. 304-312. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2015_02/article1 -- Article 1: Onset of Oil Mobilization and Nonwetting-Phase Cluster-Size
  Distribution. Berg, Armstrong, Georgiadis, Ott, Schwing, Neiteler, Brussee, Makurat, Rucker, Leu,
  Wolf, Khan, Enzmann, Kersten (2015). Petrophysics Vol. 56, No. 1 (February 2015), pp. 15-22. DOI:
  none assigned (this issue predates SPWLA DOI assignment).
src2015_04/article1 -- Article 1: Automatically Quantifying Wireline and LWD Pressure-Test Quality.
  Proett, Musharfi, Gill, Ma, Meridji, Eyuboglu (2015). Petrophysics Vol. 56, No. 2 (April 2015),
  pp. 101-115. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2015_04/article4 -- Article 4: Microresistivity Curve Extraction from Borehole Microimager Data.
  Roslin (2015). Petrophysics Vol. 56, No. 2 (April 2015), pp. 140-146. DOI: none assigned (this
  issue predates SPWLA DOI assignment).
src2015_10 -- Petrophysics Vol. 56 No. 5 (Oct 2015) (issue-level reference).
src2016_08/article2 -- Article 2: Predicting Carbonate Rock Properties Using NMR Data and
  Generalized Interpolation-Based Techniques. Kwak, Hursan, Shao, Chen, Balliet, Eid, Guergueb
  (2016). Petrophysics Vol. 57, No. 4 (August 2016), pp. 351-368. DOI: none assigned (this issue
  predates SPWLA DOI assignment).
src2016_12/article6 -- Article 6 (Technical Note): Normalizing Gamma-Ray Logs Acquired from a
  Mixture of Vertical and Horizontal Wells in the Haynesville Shale. Xu, Bayer, Wunderle, Bansal
  (2016). Petrophysics Vol. 57, No. 6 (December 2016), pp. 638-643. DOI: none assigned (this issue
  predates SPWLA DOI assignment).
src2017_08/article4 -- Article 4: The Impact of Depth and Pressure Measurement Errors on the
  Estimation of Pressure Gradients. Bowers, Schnacke, Hermance (2017). Petrophysics Vol. 58, No. 4
  (August 2017), pp. 376-396. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2018_10/article11 -- Article 11: A New Hierarchical Method for Rock Classification Using Well-
  Log-Based Rock Fabric Quantification. Purba, Garcia, Heidari (2018). DOI:
  10.30632/PJV59N5-2018a10. Petrophysics Vol. 59 No. 5 (Oct 2018) - "Best of 2018 SPWLA Symposium"
  issue.
src2018_10/article5 -- Article 5: An Unsupervised Learning Algorithm to Compute Fluid Volumes From
  NMR T1-T2 Logs in Unconventional Reservoirs. Venkataramanan, Evirgen, Allen, Mutina, Cai,
  Johnson, Green, Jiang (2018). DOI: 10.30632/PJV59N5-2018a4. Petrophysics Vol. 59 No. 5 (Oct 2018)
  - "Best of 2018 SPWLA Symposium" issue.
src2018_12 -- Petrophysics Vol. 59 No. 6 (Dec 2018) — Special Issue: Data-Driven Analytics in
  Logging and Petrophysics (issue-level reference).
src2018_12/article11 -- Article 11: Data Preconditioning for Predictive and Interpretive
  Algorithms: Importance in Data-Driven Analytics and Methods for Application. Frost, Quinn (2018).
  DOI: 10.30632/PJV59N6Y2018a10. Petrophysics Vol. 59 No. 6 (Dec 2018) — Special Issue: Data-Driven
  Analytics in Logging and Petrophysics.
src2018_12/article4 -- Article 4: Borehole Resistivity Measurement Modeling Using Machine-Learning
  Techniques. Xu, Sun, Xie, Zhong, Mirto, Feng, Hong (2018). DOI: 10.30632/PJV59N6Y2018a3.
  Petrophysics Vol. 59 No. 6 (Dec 2018) — Special Issue: Data-Driven Analytics in Logging and
  Petrophysics.
src2019_08/article2 -- Article 2: Total Organic Carbon Characterization Using Neural-Network
  Analysis of XRF Data. Lawal, Mahmoud, Alade, Abdulraheem (2019). DOI: 10.30632/PJV60N4-2019a2.
  Petrophysics Vol. 60 No. 4 (Aug 2019).
src2019_10/article9 -- Article 9: Application of Artificial Neural Network to Predict Formation
  Bulk Density While Drilling. Gowida, Elkatatny, Abdulraheem (2019). DOI: 10.30632/PJV60N5-2019a9.
  Petrophysics Vol. 60 No. 5 (Oct 2019).
src2019_12/article7 -- Article 7: New Robust Model to Estimate Formation Tops in Real Time Using
  Artificial Neural Networks (ANN). Elkatatny, Al-AbdulJabbar, Mahmoud (2019). DOI:
  10.30632/PJV60N6-2019a7. Petrophysics Vol. 60 No. 6 (Dec 2019).
src2019_12/article8 -- Article 8: A Comparative Study of Three Supervised Machine-Learning
  Algorithms for Classifying Carbonate Vuggy Facies in the Kansas Arbuckle Formation. Deng, Xu,
  Jobe, Xu (2019). DOI: 10.30632/PJV60N6-2019a8. Petrophysics Vol. 60 No. 6 (Dec 2019).
src2020_06/article5 -- Article 5: Estimation of Reservoir Porosity From Drilling Parameters Using
  Artificial Neural Networks. Al-AbdulJabbar, Al-Azani, Elkatatny (2020). DOI:
  10.30632/PJV61N3-2020a5. Petrophysics Vol. 61 No. 3 (Jun 2020).
src2020_08/article4 -- Article 4: Detecting Specific Facies in Well-Log Data Sets Using Knowledge-
  Driven Hierarchical Clustering. Emelyanova, Peyaud, Dance, Pervukhina (2020). DOI:
  10.30632/PJV61N4-2020a4. Petrophysics Vol. 61 No. 4 (Aug 2020).
src2020_10/article3 -- Article 3: Automatic Detection of Anomalous Density Measurements due to
  Wellbore Cave-in. Sen, Ong, Kainkaryam, Sharma (2020). DOI: 10.30632/PJV61N5-2020a3. Petrophysics
  Vol. 61 No. 5 (Oct 2020).
src2020_10/article7 -- Article 7: Integrated Multiphysics Workflow for Automatic Rock
  Classification and Formation Evaluation Using Multiscale Image Analysis and Conventional Well
  Logs. Gonzalez, Kanyan, Heidari, Lopez (2020). DOI: 10.30632/PJV61N5-2020a7. Petrophysics Vol. 61
  No. 5 (Oct 2020).
src2021_06 -- Petrophysics Vol. 62 No. 3 (Jun 2021) (issue-level reference).
src2021_06/article3 -- Article 3: Real-Time Prediction of Acoustic Velocities While Drilling
  Vertical Complex Lithology Using AI Technique. Alsaihati, Elkatatny (2021). DOI:
  10.30632/PJV62N3-2021a2. Petrophysics Vol. 62 No. 3 (Jun 2021).
src2021_06/article3_ai_acoustic_velocity -- Article 3: Real-Time Prediction of Acoustic Velocities
  While Drilling Vertical Complex Lithology Using AI Technique. Alsaihati, Elkatatny (2021). DOI:
  10.30632/PJV62N3-2021a2. Petrophysics Vol. 62 No. 3 (Jun 2021).
src2021_08 -- Petrophysics Vol. 62 No. 4 (Aug 2021) (issue-level reference).
src2021_08/article5 -- Article 5: Synthetic Sonic Log Generation With Machine Learning - A Contest
  Summary From Five Methods. Yu, Xu, Misra, Li, Ashby, et al. (2021). DOI: 10.30632/PJV62N4-2021a4.
  Petrophysics Vol. 62 No. 4 (Aug 2021).
src2021_12/article01 -- Article 1: Data Quality Considerations for Petrophysical Machine-Learning
  Models. McDonald (2021). DOI: 10.30632/PJV62N6-2021a1. Petrophysics Vol. 62 No. 6 (Dec 2021).
src2021_12/article02 -- Article 2: Enhanced Mineral Quantification and Uncertainty Analysis From
  Downhole Spectroscopy Logs Using Variational Autoencoders. Craddock, Srivastava, Datir, Rose,
  Zhou, Mosse, Venkataramanan (2021). DOI: 10.30632/PJV62N6-2021a2. Petrophysics Vol. 62 No. 6 (Dec
  2021).
src2022_02/article4 -- Article 4: Ultrasonic Logging of Creeping Shale. Diez, Johansen, Larsen
  (2022). DOI: 10.30632/PJV63N1-2022a4. Petrophysics Vol. 63 No. 1 (Feb 2022).
src2022_10 -- Petrophysics Vol. 63 No. 5 (Oct 2022) (issue-level reference).
src2022_10/article1 -- Article 1: A Guide to Nanoindentation. Sondergeld, Rai (2022). DOI:
  10.30632/PJV63N5-2022a1. Petrophysics Vol. 63 No. 5 (Oct 2022).
src2023_04/article02 -- Article 2: Data-Driven Algorithms for Image-Based Rock Classification and
  Formation Evaluation in Formations With Rapid Spatial Variation in Rock Fabric. Gonzalez,
  Heidari, and Lopez (2023). DOI: 10.30632/PJV64N2-2023a2. Petrophysics Vol. 64 No. 2 (Apr 2023) —
  SPWLA AI/ML Special Issue.
src2024_10 -- Petrophysics Vol. 65 No. 5 (Oct 2024) (issue-level reference).
src2024_10/lithofacies_prediction -- High-Resolution Lithofacies Prediction Using Ensemble
  Classifiers. Based on: Satti et al. (2024), "Enhancing Reservoir Characterization With High-
  Resolution Lithofacies Prediction Using Advanced Feature Engineering and Ensemble Classifiers",
  Petrophysics, Vol. 65, No. 5, pp. 813-834.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]

# --------------------------------------------------------------------------
# Regression / agreement metrics
# --------------------------------------------------------------------------


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root-mean-square error.

    Sources: src2021_12/article01, src2021_08/article5, src2019_12/article7,
    and ~25 further modules.
    """
    diff = np.asarray(y_pred, np.float64) - np.asarray(y_true, np.float64)
    return float(np.sqrt(np.mean(diff**2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error.

    Sources: src2021_12/article01, src2018_12/article4.
    """
    diff = np.asarray(y_pred, np.float64) - np.asarray(y_true, np.float64)
    return float(np.mean(np.abs(diff)))


def rss(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Residual sum of squares.

    Sources: src2015_04/article1, src2019_08/article2.
    """
    diff = np.asarray(y_pred, np.float64) - np.asarray(y_true, np.float64)
    return float(np.sum(diff**2))


def mape(y_true: ArrayLike, y_pred: ArrayLike, *, as_percent: bool = True) -> float:
    """Mean absolute percentage error (the articles' AAPE).

    Division by ``y_true`` is deliberately unguarded — a zero observation
    yields inf/NaN rather than a silently biased score (CONVENTIONS.md
    rule 6).  Sources: src2019_08/article2, src2021_06/article3.
    """
    y_t = np.asarray(y_true, np.float64)
    y_p = np.asarray(y_pred, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs((y_p - y_t) / y_t)
    return float(np.mean(ape) * (100.0 if as_percent else 1.0))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike, *, eps: float = 0.0) -> float:
    """Coefficient of determination R² (NOT the correlation coefficient R).

    ``eps`` is the additive denominator guard several articles use
    (``ss_tot + 1e-12``); the default 0.0 returns a non-finite value
    (-inf, or NaN when the fit is also perfect) for constant ``y_true``.
    Sources: src2021_08/article5, src2019_12/article7,
    src2022_10/article1 (whose argument order was reversed).
    """
    y_t = np.asarray(y_true, np.float64)
    y_p = np.asarray(y_pred, np.float64)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - y_t.mean()) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(1.0 - ss_res / (ss_tot + eps))


def pearson_r(x: ArrayLike, y: ArrayLike, *, eps: float = 0.0) -> float:
    """Pearson correlation coefficient R (NOT R²).

    ``eps`` is the additive denominator guard; the default 0.0 returns NaN
    when either input is constant.  Articles with a *conditional* fallback
    (``return 0.0 if denom <= 1e-12``) keep that branch in their facade.
    Sources: src2021_12/article01, src2021_06/article3, src2019_08/article2.
    """
    x_c = np.asarray(x, np.float64)
    y_c = np.asarray(y, np.float64)
    x_c = x_c - x_c.mean()
    y_c = y_c - y_c.mean()
    denom = np.sqrt(np.sum(x_c**2) * np.sum(y_c**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sum(x_c * y_c) / (denom + eps))


# --------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of matching labels.

    Sources: src2019_12/article8, src2020_08/article4, src2024_10.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return float(np.mean(y_t == y_p))


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = None,
) -> NDArray[np.int64]:
    """Confusion matrix ``C[i, j]`` = count of true class i predicted as j.

    ``labels`` fixes the class order (default: sorted union of both label
    sets).  For binary 0/1 labels this reproduces the repository's
    ``[[TN, FP], [FN, TP]]`` convention.  Sources: src2019_12/article8
    (binary), src2024_10/lithofacies_prediction (multiclass).
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    label_values = np.unique(np.concatenate([y_t, y_p])) if labels is None else np.asarray(labels)
    index = {value: i for i, value in enumerate(label_values.tolist())}
    matrix = np.zeros((len(label_values), len(label_values)), dtype=np.int64)
    for t, p in zip(y_t.tolist(), y_p.tolist(), strict=True):
        matrix[index[t], index[p]] += 1
    return matrix


def precision_recall_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    positive: object = 1,
    zero_division: float = 0.0,
) -> tuple[float, float, float]:
    """Binary precision, recall and F1 for the ``positive`` class.

    ``zero_division`` is returned when a denominator is empty (the
    repository convention is 0.0).  Sources: src2019_12/article8,
    src2020_08/article4 (which takes raw tp/fp/fn counts — its facade
    builds them from this).
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    tp = float(np.sum((y_t == positive) & (y_p == positive)))
    fp = float(np.sum((y_t != positive) & (y_p == positive)))
    fn = float(np.sum((y_t == positive) & (y_p != positive)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else zero_division
    )
    return precision, recall, f1


# --------------------------------------------------------------------------
# Feature scaling
# --------------------------------------------------------------------------


def zscore(
    x: ArrayLike,
    *,
    axis: int | None = None,
    eps: float = 0.0,
) -> _Float:
    """Standardize to zero mean and unit standard deviation.

    ``axis=None`` scales the array as a whole (the flat-log idiom);
    ``axis=0`` scales feature columns independently.  ``eps`` is the
    additive guard on the denominator (``std + eps``); default 0.0 yields
    NaN for a constant input.  Population std (``ddof=0``), matching every
    article copy.  Sources: src2018_12/article11, src2021_08/article5,
    src2020_10/article3.
    """
    array = np.asarray(x, np.float64)
    if axis is None:
        mean: np.float64 | NDArray[np.float64] = array.mean()
        std: np.float64 | NDArray[np.float64] = array.std()
    else:
        mean = array.mean(axis=axis, keepdims=True)
        std = array.std(axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray((array - mean) / (std + eps))


def minmax(
    x: ArrayLike,
    *,
    axis: int | None = None,
    lo: float = 0.0,
    hi: float = 1.0,
    eps: float = 0.0,
) -> _Float:
    """Rescale to the range [lo, hi] (min-max normalization).

    ``eps`` is the additive guard on the range denominator; default 0.0
    yields NaN for a constant input.  Sources: src2018_12/article11,
    src2019_10/article9, src2020_06/article5.
    """
    array = np.asarray(x, np.float64)
    if axis is None:
        x_min: np.float64 | NDArray[np.float64] = array.min()
        x_max: np.float64 | NDArray[np.float64] = array.max()
    else:
        x_min = array.min(axis=axis, keepdims=True)
        x_max = array.max(axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        unit = (array - x_min) / (x_max - x_min + eps)
    return np.asarray(lo + unit * (hi - lo))


def affine_rescale(
    x: ArrayLike,
    *,
    src_lo: float,
    src_hi: float,
    dst_lo: float,
    dst_hi: float,
) -> _Float:
    """Map the source range [src_lo, src_hi] linearly onto [dst_lo, dst_hi].

    The reference-range (histogram) normalization of Shier (2004) used for
    multi-well GR normalization — unlike :func:`minmax` the ranges are
    caller-supplied percentiles, not data extremes.  Sources:
    src2016_12/article6, src2015_04/article4.
    """
    array = np.asarray(x, np.float64)
    return np.asarray(dst_lo + (array - src_lo) * (dst_hi - dst_lo) / (src_hi - src_lo))


# --------------------------------------------------------------------------
# Least-squares fit wrappers
# --------------------------------------------------------------------------


def fit_line(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """Degree-1 least-squares fit; returns ``(slope, intercept)`` — in that
    order, always.

    The repository's most-duplicated numerical wrapper (~45 files fit a
    line in some transformed space, disagreeing on return order).
    Transform-space fits (log-log Archie exponents, semilog Klinkenberg)
    are built on this in their own domains.  Sources:
    src2015_04/article1, src2017_08/article4, src2022_02/article4.
    """
    slope, intercept = np.polyfit(np.asarray(x, np.float64), np.asarray(y, np.float64), 1)
    return float(slope), float(intercept)


def fit_powerlaw(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """Fit ``y = coefficient * x**exponent``; returns ``(coefficient, exponent)``.

    A log10-log10 :func:`fit_line`; inputs must be positive (zeros or
    negatives propagate as NaN).  Note: articles disagree on returning the
    exponent or its negation — facades apply their historical sign.
    Sources: src2014_08/article3, src2015_02/article1.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_x = np.log10(np.asarray(x, np.float64))
        log_y = np.log10(np.asarray(y, np.float64))
    exponent, log_coefficient = fit_line(log_x, log_y)
    return float(10.0**log_coefficient), float(exponent)


def ols(X: ArrayLike, y: ArrayLike, *, intercept: bool = True) -> tuple[_Float, float]:
    """Multivariate ordinary least squares; returns ``(coef, intercept)``.

    The intercept is returned separately — never at ``beta[0]`` or
    ``coef[-1]`` — because the repository holds both conventions and a
    silent mix-up is a wrong prediction (module docstring).  Sources:
    src2021_06/article3 (fit_linear, prepended), src2018_12/article4
    (ols_fit, appended).
    """
    design = np.asarray(X, np.float64)
    if design.ndim == 1:
        design = design[:, None]
    target = np.asarray(y, np.float64)
    if intercept:
        design = np.column_stack([design, np.ones(len(design))])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    if intercept:
        return np.asarray(beta[:-1]), float(beta[-1])
    return np.asarray(beta), 0.0


def predict_linear(X: ArrayLike, coef: ArrayLike, intercept: float = 0.0) -> _Float:
    """Evaluate the linear model from :func:`ols`.

    Sources: src2021_06/article3_ai_acoustic_velocity.
    """
    design = np.asarray(X, np.float64)
    if design.ndim == 1:
        design = design[:, None]
    return np.asarray(design @ np.asarray(coef, np.float64) + intercept)


# --------------------------------------------------------------------------
# Clustering / decomposition
# --------------------------------------------------------------------------


def kmeans(
    x: ArrayLike,
    k: int,
    *,
    weights: ArrayLike | None = None,
    max_iter: int = 100,
) -> tuple[NDArray[np.int64], _Float]:
    """Deterministic k-means; returns ``(labels, centroids)``.

    Uses the repository's farthest-point initialization (first centre =
    point farthest from the data mean, then greedy max-min), Lloyd
    iterations, convergence on unchanged labels, and optional per-sample
    ``weights`` in the centroid update.  Deterministic by construction —
    no RNG — so results are exactly reproducible; note cluster NUMBERING
    is an artifact of the init, not a physical ordering.  The quantile
    init of src2015_10 stays article-local.  Sources:
    src2018_10/article5 (weighted), src2020_08/article4,
    src2020_10/article3, src2020_10/article7.
    """
    data = np.asarray(x, np.float64)
    w = np.ones(len(data)) if weights is None else np.asarray(weights, np.float64)
    seed_indices = [int(np.argmax(np.linalg.norm(data - data.mean(axis=0), axis=1)))]
    for _ in range(1, k):
        distance = np.min([np.linalg.norm(data - data[j], axis=1) for j in seed_indices], axis=0)
        seed_indices.append(int(np.argmax(distance)))
    centroids = data[seed_indices].copy()
    labels = np.zeros(len(data), dtype=np.int64)
    for _ in range(max_iter):
        distances = np.stack([np.linalg.norm(data - c, axis=1) for c in centroids], axis=1)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = labels == c
            if members.any():
                centroids[c] = np.average(data[members], axis=0, weights=w[members])
    return np.asarray(labels, dtype=np.int64), centroids


def silhouette_score(x: ArrayLike, labels: ArrayLike) -> float:
    """Mean silhouette coefficient over all samples (O(N²) pairwise).

    Returns 0.0 for fewer than two clusters; a singleton's own-cluster
    distance counts as 0.  Sources: src2018_10/article11,
    src2020_08/article4, src2023_04/article02.
    """
    data = np.asarray(x, np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    unique = np.unique(label_array)
    if len(unique) < 2:
        return 0.0
    pairwise = np.sqrt(((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=-1))
    scores = np.zeros(len(data))
    for i in range(len(data)):
        own = label_array == label_array[i]
        own[i] = False
        a = pairwise[i, own].mean() if own.any() else 0.0
        b = min(pairwise[i, label_array == c].mean() for c in unique if c != label_array[i])
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(scores.mean())


def pca(
    x: ArrayLike,
    *,
    n_components: int | None = None,
) -> tuple[_Float, _Float, _Float, _Float]:
    """Principal component analysis via SVD of the centered data.

    Returns ``(scores, components, mean, explained)`` where ``components``
    rows are the principal axes and ``explained`` is the fraction of
    variance per retained component.  Article variants return subsets of
    this tuple — facades slice it.  Sources: src2016_08/article2,
    src2014_06/article4 (explained fractions only), src2021_12/article02.
    """
    data = np.asarray(x, np.float64)
    mean = data.mean(axis=0)
    centered = data - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    explained_all = variance / variance.sum()
    keep = len(singular_values) if n_components is None else n_components
    components = vt[:keep]
    scores = centered @ components.T
    return (
        np.asarray(scores),
        np.asarray(components),
        np.asarray(mean),
        np.asarray(explained_all[:keep]),
    )
