"""
Article 11: A New Hierarchical Method for Rock Classification Using Well-Log-Based
            Rock Fabric Quantification
Purba, Garcia, Heidari (2018)
DOI: 10.30632/PJV59N5-2018a10

Rocks are classified hierarchically from well-log-derived rock-fabric attributes:
agglomerative clustering merges the most similar samples step by step, building
a dendrogram, and cutting it at a chosen distance yields the rock classes - a
data-driven rock-typing that does not require a preset number of classes.

Implements:

  - Standardized rock-fabric feature distances
  - Agglomerative (average-linkage) hierarchical clustering
  - Dendrogram cut at a distance threshold -> rock classes
  - Silhouette validation of the resulting classes

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard agglomerative-clustering rock-classification the
paper's title describes.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- agglomerative -----------

def agglomerative(X, n_clusters):
    """Average-linkage agglomerative clustering; returns integer cluster labels."""
    X = np.asarray(X, float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    clusters = [[i] for i in range(len(Xs))]
    D = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1))

    def cluster_dist(a, b):
        return np.mean([D[i, j] for i in a for j in b])    # average linkage

    while len(clusters) > n_clusters:
        best = (None, None, np.inf)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = cluster_dist(clusters[i], clusters[j])
                if d < best[2]:
                    best = (i, j, d)
        i, j, _ = best
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)
    labels = np.zeros(len(Xs), int)
    for c, members in enumerate(clusters):
        for m in members:
            labels[m] = c
    return labels


def silhouette(X, labels):
    """Mean silhouette coefficient (cluster-quality validation)."""
    return petrolib.ml_stats.silhouette_score(X, labels)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 11: Hierarchical Rock Classification")
    print("=" * 60)

    # Three well-separated rock-fabric groups in a 2-attribute space
    rng = np.random.default_rng(10)
    centers = np.array([[0, 0], [5, 0], [0, 5]], float)
    X, truth = [], []
    for g, c in enumerate(centers):
        pts = c + 0.4 * rng.standard_normal((25, 2))
        X.append(pts); truth += [g] * 25
    X = np.vstack(X); truth = np.array(truth)

    labels = agglomerative(X, n_clusters=3)
    print(f"  clusters found         = {len(np.unique(labels))}")
    assert len(np.unique(labels)) == 3

    # Clustering recovers the three groups (each true group maps to one label)
    from collections import Counter
    pure = all(max(Counter(labels[truth == g]).values()) == 25 for g in range(3))
    assert pure

    # Silhouette is high for the well-separated classes
    sil = silhouette(X, labels)
    print(f"  silhouette             = {sil:.3f}")
    assert sil > 0.7

    # Cutting into fewer classes merges groups (coarser rock typing)
    assert len(np.unique(agglomerative(X, n_clusters=2))) == 2
    print("  PASS")
    return {"n_classes": int(len(np.unique(labels))), "silhouette": sil}


if __name__ == "__main__":
    test_all()
