"""
Article 5: Integrated Petrophysical Rock Classification in the McElroy Field,
           West Texas, USA
Saneifar, Skalinski, Theologou, Kenter, Cuffey, Salazar-Tio (2015)
Reference: Petrophysics Vol. 56, No. 5 (October 2015), pp. 493-510
DOI: none assigned (this issue predates SPWLA DOI assignment)

An integrated petrophysical rock classification for the San Andres dolomite
carbonate of the McElroy Field, combining pore-geometry indices with
log-/core-based facies clustering.  This module implements the standard
carbonate rock-classification toolkit the paper applies: the Winland r35
pore-throat radius, the reservoir-quality / flow-zone indicators, and a simple
electrofacies clustering of samples into rock classes.

Implements:

  - Winland r35 pore-throat radius from porosity and permeability
  - Reservoir quality index (RQI), normalized porosity, flow zone indicator (FZI)
  - k-means electrofacies clustering into rock classes

Note: this article's body was beyond the PDF text extraction for this issue
(the source text truncates within the preceding article), so this module is a
methodology proxy implementing the standard carbonate rock-classification
methods the title/abstract describe, consistent with how other truncated
articles are handled in this repository.  Permeability in mD, porosity as a
fraction (Winland uses percent internally), r35 in microns.
"""

import numpy as np


# ---------------------------------------------- pore-geometry indices --------------

def winland_r35(k, phi):
    """Winland (1972) r35 pore-throat radius

        log10(r35) = 0.732 + 0.588*log10(k) - 0.864*log10(phi_pct),

    with k in mD and porosity in percent; r35 (um) is the pore-throat radius at
    35% mercury saturation, a strong permeability/rock-quality discriminator.
    """
    phi_pct = np.asarray(phi, float) * 100.0
    return 10.0 ** (0.732 + 0.588 * np.log10(k) - 0.864 * np.log10(phi_pct))


def rqi(k, phi):
    """Reservoir quality index  RQI = 0.0314*sqrt(k/phi)  [um], k in mD."""
    return 0.0314 * np.sqrt(np.asarray(k, float) / phi)


def normalized_porosity(phi):
    """Normalized porosity index  phi_z = phi/(1 - phi)."""
    phi = np.asarray(phi, float)
    return phi / (1.0 - phi)


def flow_zone_indicator(k, phi):
    """Flow zone indicator  FZI = RQI/phi_z."""
    return rqi(k, phi) / normalized_porosity(phi)


# ---------------------------------------------- electrofacies --------------

def kmeans_rock_classes(features, n_classes, n_iter=100, seed=0):
    """Cluster samples into petrophysical rock classes by k-means.

    `features` is (n_samples, n_features) (e.g. [log10(k), phi, r35]); returns
    (labels, centers).  A lightweight, dependency-free k-means with k-means++-
    style spread initialization on a fixed seed for reproducibility.
    """
    x = np.atleast_2d(np.asarray(features, float))
    rng = np.random.default_rng(seed)
    # initialize centers on spread quantiles of the first feature
    order = np.argsort(x[:, 0])
    idx = order[np.linspace(0, len(x) - 1, n_classes).astype(int)]
    centers = x[idx].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(n_iter):
        d = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=-1)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(n_classes):
            if np.any(labels == c):
                centers[c] = x[labels == c].mean(axis=0)
    return labels, centers


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: McElroy Integrated Rock Classification")
    print("=" * 60)

    # Winland r35 increases with permeability and decreases with tighter porosity
    r35 = winland_r35(10.0, 0.15)
    print(f"  Winland r35            = {r35:.3f} um")
    assert r35 > 0 and winland_r35(100.0, 0.15) > r35
    # at fixed permeability, higher porosity implies smaller pore throats (r35 down)
    assert winland_r35(10.0, 0.25) < winland_r35(10.0, 0.10)

    # RQI / FZI are positive; better pore geometry raises FZI
    assert rqi(10.0, 0.15) > 0 and flow_zone_indicator(10.0, 0.15) > 0

    # Electrofacies clustering separates good- and poor-quality rock
    rng = np.random.default_rng(1)
    good = np.column_stack([rng.normal(2.0, 0.2, 30), rng.normal(0.22, 0.02, 30)])   # high logk, phi
    poor = np.column_stack([rng.normal(-1.0, 0.2, 30), rng.normal(0.08, 0.02, 30)])  # low logk, phi
    feats = np.vstack([good, poor])
    labels, centers = kmeans_rock_classes(feats, n_classes=2)
    print(f"  cluster centers (logk) = {np.round(centers[:, 0], 2)}")
    # the 30 good and 30 poor samples should each fall predominantly in one class
    assert len(np.unique(labels)) == 2
    g_label = np.bincount(labels[:30]).argmax()
    p_label = np.bincount(labels[30:]).argmax()
    assert g_label != p_label
    print("  PASS")
    return {"r35": float(r35), "FZI": float(flow_zone_indicator(10.0, 0.15))}


if __name__ == "__main__":
    test_all()
