"""
article4_facies_classification.py
=================================

Implementation of the three automatic facies-analysis methods compared in:

    Morelli, C., Yang, S., Maehara, Y., Cai, H., Moe, K., Yamada, Y.,
    and Matter, J. (2024). "Automatic Geological Facies Analysis in
    Crust-Mantle Transition Zone."  Petrophysics 65(3), 342-363.
    DOI: 10.30632/PJV65N3-2024a4

The paper applies three methods to logs + borehole images acquired in the
Oman Drilling Project (CM2A / CM2B wells) to classify rock facies
(dunite, gabbro, harzburgite).  The three methods are:

* FaciesSpect: PCA of image-derived features and log curves, followed by
  agglomerative (hierarchical) clustering.
* CBML (Class-Based Machine Learning): PCA, then a Gaussian mixture
  model, followed by a hidden Markov model used as a simple depth
  regulariser to suppress abrupt class transitions.
* HRA (Heterogeneous Rock Analysis): supervised-in-spirit K-means on the
  log attributes, with a user-specified number of classes.

We implement a faithful but minimal version of each.  The borehole image
is represented as a (depth, azimuth) matrix from which simple per-depth
statistics (mean, contrast) are used as features.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature extraction --------------------------------------------------------
# ---------------------------------------------------------------------------

def image_features(borehole_image: np.ndarray) -> np.ndarray:
    """Per-depth image features: mean, std (contrast), range.

    `borehole_image` is a 2D array of shape (n_depths, n_azimuths).
    Returns a (n_depths, 3) feature matrix.
    """
    img = np.asarray(borehole_image, dtype=float)
    mean = img.mean(axis=1)
    std = img.std(axis=1)
    rng = img.max(axis=1) - img.min(axis=1)
    return np.column_stack([mean, std, rng])


def build_feature_matrix(image: np.ndarray, logs: dict[str, np.ndarray]
                         ) -> np.ndarray:
    """Concatenate image features with log curves (Fe, Ca, ...)."""
    feats = [image_features(image)]
    for key in sorted(logs.keys()):
        col = np.asarray(logs[key], dtype=float).reshape(-1, 1)
        feats.append(col)
    return np.hstack(feats)


# ---------------------------------------------------------------------------
# FaciesSpect: PCA + agglomerative clustering -------------------------------
# ---------------------------------------------------------------------------

def faciesspect(features: np.ndarray, n_classes: int = 15,
                n_components: int | None = None) -> np.ndarray:
    """FaciesSpect workflow: standardise -> PCA -> agglomerative clustering."""
    x = StandardScaler().fit_transform(features)
    if n_components is None:
        n_components = min(x.shape[1], 5)
    x_pca = PCA(n_components=n_components).fit_transform(x)
    model = AgglomerativeClustering(n_clusters=n_classes, linkage="ward")
    return model.fit_predict(x_pca)


# ---------------------------------------------------------------------------
# CBML: PCA + GMM + HMM depth regularisation --------------------------------
# ---------------------------------------------------------------------------

def cbml(features: np.ndarray, n_classes: int = 15,
         transition_stickiness: float = 20.0) -> np.ndarray:
    """CBML workflow: PCA -> GMM classification -> HMM Viterbi smoothing.

    The HMM uses the GMM log-likelihoods as emission probabilities and a
    simple "sticky" transition matrix with mass `transition_stickiness`
    on the diagonal -- this is the minimal version of the depth
    regularisation the paper describes.
    """
    x = StandardScaler().fit_transform(features)
    n_components = min(x.shape[1], 5)
    x_pca = PCA(n_components=n_components).fit_transform(x)

    gmm = GaussianMixture(n_components=n_classes, covariance_type="diag",
                          random_state=0, n_init=2)
    gmm.fit(x_pca)

    # log-likelihood of each sample under each component (emission probs)
    log_emit = np.zeros((x_pca.shape[0], n_classes))
    for k in range(n_classes):
        diff = x_pca - gmm.means_[k]
        var = gmm.covariances_[k]
        log_emit[:, k] = -0.5 * np.sum(diff ** 2 / var + np.log(2 * np.pi * var),
                                       axis=1)

    # Sticky transition matrix
    T = np.full((n_classes, n_classes), 1.0)
    np.fill_diagonal(T, 1.0 + transition_stickiness)
    T = T / T.sum(axis=1, keepdims=True)
    log_T = np.log(T)

    return _viterbi(log_emit, log_T)


def _viterbi(log_emit: np.ndarray, log_T: np.ndarray) -> np.ndarray:
    """Standard Viterbi decoding for a first-order HMM."""
    n, k = log_emit.shape
    delta = np.full((n, k), -np.inf)
    psi = np.zeros((n, k), dtype=int)
    delta[0] = log_emit[0]
    for t in range(1, n):
        scores = delta[t - 1][:, None] + log_T
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores.max(axis=0) + log_emit[t]
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


# ---------------------------------------------------------------------------
# HRA: K-means on log attributes --------------------------------------------
# ---------------------------------------------------------------------------

def hra(features: np.ndarray, n_classes: int = 15,
        n_init: int = 10, seed: int = 0) -> np.ndarray:
    """Heterogeneous Rock Analysis: standardise -> K-means."""
    x = StandardScaler().fit_transform(features)
    km = KMeans(n_clusters=n_classes, n_init=n_init, random_state=seed)
    return km.fit_predict(x)


# ---------------------------------------------------------------------------
# Validation helpers --------------------------------------------------------
# ---------------------------------------------------------------------------

def cluster_purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Purity score (fraction of samples whose cluster majority-matches
    the ground truth lithology)."""
    labels = np.asarray(labels)
    truth = np.asarray(truth)
    total = 0
    for c in np.unique(labels):
        members = truth[labels == c]
        if len(members) == 0:
            continue
        _, counts = np.unique(members, return_counts=True)
        total += counts.max()
    return total / len(labels)


# ---------------------------------------------------------------------------
# Test harness ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def _make_synthetic_well(n_depths: int = 300,
                         seed: int = 0
                         ) -> tuple[np.ndarray, dict, np.ndarray]:
    """Three-zone synthetic well: dunite / gabbro / harzburgite."""
    rng = np.random.default_rng(seed)
    truth = np.zeros(n_depths, dtype=int)
    truth[100:200] = 1
    truth[200:] = 2

    # Generate a borehole image with different mean / contrast per zone
    image = np.zeros((n_depths, 32))
    fe = np.zeros(n_depths)
    ca = np.zeros(n_depths)
    params = [(0.3, 0.05, 0.08, 0.02),      # dunite: dark, low Fe, low Ca
              (0.7, 0.15, 0.04, 0.30),      # gabbro: bright, high Ca
              (0.4, 0.03, 0.09, 0.01)]      # harzburgite
    for idx, (mu, cstd, mu_fe, mu_ca) in enumerate(params):
        m = truth == idx
        image[m] = rng.normal(mu, cstd, size=(m.sum(), 32))
        fe[m] = rng.normal(mu_fe, 0.005, size=m.sum())
        ca[m] = rng.normal(mu_ca, 0.01, size=m.sum())
    logs = {"Fe": fe, "Ca": ca}
    return image, logs, truth


def test_all(verbose: bool = True) -> None:
    image, logs, truth = _make_synthetic_well()
    feats = build_feature_matrix(image, logs)
    assert feats.shape[0] == truth.size
    assert feats.shape[1] == 3 + 2                  # 3 image + Fe + Ca

    # Run the three methods with 3 classes (matching the ground truth).
    labels_fs = faciesspect(feats, n_classes=3)
    labels_cb = cbml(feats, n_classes=3)
    labels_hra = hra(feats, n_classes=3)

    p_fs = cluster_purity(labels_fs, truth)
    p_cb = cluster_purity(labels_cb, truth)
    p_hra = cluster_purity(labels_hra, truth)

    for name, p in [("FaciesSpect", p_fs), ("CBML", p_cb), ("HRA", p_hra)]:
        assert p > 0.8, f"{name} purity too low: {p:.3f}"

    # CBML with a very sticky HMM should produce fewer label changes than
    # a purely point-wise classifier.
    labels_cb_sticky = cbml(feats, n_classes=3, transition_stickiness=200.0)
    changes_sticky = np.sum(np.diff(labels_cb_sticky) != 0)
    changes_plain = np.sum(np.diff(labels_hra) != 0)
    assert changes_sticky <= changes_plain, \
        "Sticky CBML should have <= transitions than K-means"

    if verbose:
        print("Article 4 (Automatic facies analysis): all tests passed.")
        print(f"  FaciesSpect purity = {p_fs:.3f}")
        print(f"  CBML        purity = {p_cb:.3f}")
        print(f"  HRA         purity = {p_hra:.3f}")
        print(f"  HRA transitions    = {changes_plain}")
        print(f"  CBML transitions   = {changes_sticky}")


if __name__ == "__main__":
    test_all()
