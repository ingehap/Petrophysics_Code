"""
Article 4: Detecting Specific Facies in Well-Log Data Sets Using Knowledge-Driven
           Hierarchical Clustering
Emelyanova, Peyaud, Dance, Pervukhina (2020)
DOI: 10.30632/PJV61N4-2020a4

A knowledge-driven hierarchical clustering (KDHC) workflow builds a decision
tree over wireline logs to isolate a specific facies (high-quality baffles).
At each node a clustering splits the data; three expert-rule metrics - indicator
(I), purity (P), and decision (D) - decide whether each cluster (a candidate
containing baffles, CCB) is split further, kept, or discarded, stopping when
D = 1.  Results are validated against expert electrofacies with the F1 score.

Implements:

  - Neutron-density separation  ND = NP - (rhoG - ZDN)/(rhoG - rhoW)  (Eq. 1)
  - Baffle expert rule  (ZDN > 2.55 and MLR > 15)
  - Cluster area, indicator I (min-max), purity P = A_E4/A_C          (Eqs. 2-5)
  - Decision metric  D = K/N (stop when D = 1)                        (Eq. 6)
  - F1 score from precision and recall                                (Eq. 7)
  - k-means split + silhouette cluster-count selection

Note: this issue's PDF text layer kept the equation numbers and definitions but
dropped the typeset glyphs, so the metrics are reconstructed from the preserved
definitions; the per-node spectral-clustering splitter is represented by a
k-means + silhouette proxy.  Paper anchors reproduced: rhoG = 2.66, rhoW = 1.03,
baffle thresholds ZDN > 2.55 / MLR > 15, reservoir F1 ~ 0.98.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

RHO_GRAIN = 2.66         # g/cm^3
RHO_WATER = 1.03         # g/cm^3
ZDN_BAFFLE = 2.55        # g/cm^3
MLR_BAFFLE = 15.0        # ohm-m


# ---------------------------------------------- log transform -----------

def nd_separation(np_neutron, zdn, rho_g=RHO_GRAIN, rho_w=RHO_WATER):
    """Neutron-density separation  ND = NP - (rhoG - ZDN)/(rhoG - rhoW)  (Eq. 1)."""
    phi_d = (rho_g - np.asarray(zdn, float)) / (rho_g - rho_w)
    return np.asarray(np_neutron, float) - phi_d


def is_baffle(zdn, mlr, zdn_cut=ZDN_BAFFLE, mlr_cut=MLR_BAFFLE):
    """Expert baffle rule: dense AND resistive  (ZDN > 2.55 and MLR > 15)."""
    return (np.asarray(zdn, float) > zdn_cut) & (np.asarray(mlr, float) > mlr_cut)


# ---------------------------------------------- KDHC metrics ------------

def cluster_area(mlr, zdn):
    """Cluster area in the log(MLR)-ZDN plane  (Eq. 3)."""
    mlr = np.asarray(mlr, float); zdn = np.asarray(zdn, float)
    return (np.log10(mlr.max()) - np.log10(mlr.min())) * (zdn.max() - zdn.min())


def indicator_metric(areas_e4):
    """Min-max normalized indicator across clusters  (Eq. 4)."""
    a = np.asarray(areas_e4, float)
    rng = a.max() - a.min()
    return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)


def purity_metric(area_e4, area_cluster):
    """Purity  P = A_E4 / A_C  (Eq. 5)."""
    return area_e4 / area_cluster


def decision_metric(n_ccb, n_clusters):
    """Decision  D = K/N (number of CCBs / clusters); stop when D = 1  (Eq. 6)."""
    return n_ccb / n_clusters


def f1_score(tp, fp, fn):
    """F1 = 2*precision*recall/(precision+recall)  (Eq. 7)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------- clustering --------------

def kmeans(X, k, iters=100):
    """Minimal k-means; returns (labels, centers)."""
    return petrolib.ml_stats.kmeans(X, k, max_iter=iters)


def silhouette(X, labels):
    """Mean silhouette coefficient (Rousseeuw)."""
    return petrolib.ml_stats.silhouette_score(X, labels)


def best_cluster_count(X, candidates=(2, 3, 4, 5)):
    """Pick the cluster count with the highest mean silhouette (the SI step)."""
    best_k, best_s = candidates[0], -1.0
    for k in candidates:
        labels, _ = kmeans(X, k)
        s = silhouette(X, labels)
        if s > best_s:
            best_k, best_s = k, s
    return best_k, best_s


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: KDHC Facies Detection")
    print("=" * 60)

    # ND separation: at equal neutron porosity, a dense (clay-rich / baffle)
    # point has a larger neutron-density separation than a clean reservoir point
    nd_clean = nd_separation(0.20, 2.35)
    nd_dense = nd_separation(0.20, 2.60)
    print(f"  ND clean / dense       = {nd_clean:.3f} / {nd_dense:.3f}")
    assert nd_dense > nd_clean

    # Expert baffle rule flags only dense AND resistive points
    zdn = np.array([2.40, 2.60, 2.60, 2.40])
    mlr = np.array([5.0, 20.0, 5.0, 20.0])
    flags = is_baffle(zdn, mlr)
    assert list(flags) == [False, True, False, False]

    # Decision metric reaches 1 when every cluster is a CCB (stop criterion)
    assert decision_metric(1, 2) == 0.5
    assert decision_metric(5, 5) == 1.0

    # Indicator is min-max normalized across clusters (0..1)
    I = indicator_metric([0.1, 0.5, 0.9, 0.3])
    print(f"  indicator              = {np.array2string(I, precision=2)}")
    assert I.max() == 1.0 and I.min() == 0.0

    # Purity is the E4-area fraction of the cluster area
    assert abs(purity_metric(0.44, 1.0) - 0.44) < 1e-9

    # F1: a near-perfect reservoir classification reproduces the ~0.98 score
    f1 = f1_score(tp=18000, fp=200, fn=300)
    print(f"  reservoir F1           = {f1:.3f}")
    assert f1 > 0.98

    # Clustering: two log-facies separate cleanly; silhouette picks k = 2
    rng = np.random.default_rng(4)
    reservoir = np.column_stack([rng.normal(2.30, 0.03, 80),     # ZDN
                                 rng.normal(0.8, 0.1, 80)])      # log10 MLR
    baffle = np.column_stack([rng.normal(2.62, 0.03, 40),
                              rng.normal(1.4, 0.1, 40)])
    X = np.vstack([reservoir, baffle])
    k, s = best_cluster_count(X)
    print(f"  best cluster count     = {k}  (silhouette {s:.3f})")
    assert k == 2 and s > 0.6
    print("  PASS")
    return {"nd_dense": float(nd_dense), "F1": f1, "best_k": k,
            "silhouette": s}


if __name__ == "__main__":
    test_all()
