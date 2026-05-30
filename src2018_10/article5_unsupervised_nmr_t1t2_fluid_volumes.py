"""
Article 5: An Unsupervised Learning Algorithm to Compute Fluid Volumes From NMR
           T1-T2 Logs in Unconventional Reservoirs
Venkataramanan, Evirgen, Allen, Mutina, Cai, Johnson, Green, Jiang (2018)
DOI: 10.30632/PJV59N5-2018a4

A 2D NMR T1-T2 map of an unconventional reservoir contains several fluid
populations (clay-bound water, capillary water, movable water/light
hydrocarbon, bitumen).  An unsupervised clustering algorithm groups the T1-T2
map's amplitude into these populations - without preset cutoffs - and sums each
cluster's amplitude into a fluid volume.

Implements:

  - T1-T2 map feature space (log T2, log T1/T2)
  - k-means clustering of the amplitude-weighted map
  - Fluid-volume computation per cluster (amplitude sum)
  - Cluster labeling by T1/T2 ratio and T2

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so this is a faithful standard-form reconstruction of the
unsupervised T1-T2 fluid-volume method the paper presents (numpy k-means).
"""

import numpy as np


# ---------------------------------------------- clustering --------------

def kmeans(X, k, weights=None, iters=100, seed=0):
    """Weighted k-means; returns (labels, centers)."""
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


def fluid_volumes(labels, amplitudes, n_clusters):
    """Sum amplitudes within each cluster into fluid volumes (normalized)."""
    a = np.asarray(amplitudes, float)
    vols = np.array([a[labels == c].sum() for c in range(n_clusters)])
    return vols / vols.sum()


def label_cluster(t2, t1t2):
    """Label a cluster centroid by its (T2, T1/T2)."""
    if t1t2 > 4.0:
        return "clay-bound/bitumen"
    if t2 < 10.0:
        return "capillary"
    return "movable"


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Unsupervised NMR T1-T2 Fluid Volumes")
    print("=" * 60)

    rng = np.random.default_rng(4)
    # Three planted fluid populations on the (log T2, log T1/T2) map
    pops = [
        (0.7, 1.2, 0.30),    # clay-bound: short T2, high T1/T2, 30% volume
        (1.3, 0.3, 0.25),    # capillary:  mid T2, low T1/T2,   25%
        (2.4, 0.1, 0.45),    # movable:    long T2, ~1 T1/T2,   45%
    ]
    feats, amps, truth = [], [], []
    for (lt2, lr, vol), n in zip(pops, [40, 35, 60]):
        feats.append(np.column_stack([rng.normal(lt2, 0.08, n),
                                      rng.normal(lr, 0.05, n)]))
        amps.append(np.full(n, vol / n))
        truth.append(vol)
    X = np.vstack(feats); amp = np.concatenate(amps)

    labels, centers = kmeans(X, 3, weights=amp)
    vols = fluid_volumes(labels, amp, 3)
    print(f"  recovered volumes      = {np.array2string(np.sort(vols)[::-1], precision=2)}")
    # the three recovered volumes match the planted set (order-independent)
    assert np.allclose(np.sort(vols), np.sort(truth), atol=0.03)

    # Cluster labeling by centroid (T2 from log T2, ratio from log T1/T2)
    labs = [label_cluster(10 ** c[0], 10 ** c[1]) for c in centers]
    print(f"  cluster labels         = {labs}")
    assert "clay-bound/bitumen" in labs and "movable" in labs
    print("  PASS")
    return {"volumes": np.sort(vols)[::-1].tolist()}


if __name__ == "__main__":
    test_all()
