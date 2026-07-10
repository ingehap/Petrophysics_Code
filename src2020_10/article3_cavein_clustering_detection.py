"""
Article 3: Automatic Detection of Anomalous Density Measurements due to Wellbore
           Cave-in
Sen, Ong, Kainkaryam, Sharma (2020)
DOI: 10.30632/PJV61N5-2020a3

Wellbore cave-ins (washouts) corrupt the bulk-density log where the caliper
enlarges.  An unsupervised time-series clustering of the depth-indexed caliper
and density logs separates "good hole" from "bad hole" (cave-in) intervals; the
rolling coefficient of variation of bulk density and the caliper excess over bit
size are the discriminating features.  A temporal smoothing penalty suppresses
spurious single-sample cluster switches.

Implements:

  - Rolling coefficient of variation  CV = sigma/mean over a 2*wr window  (Eq. 5)
  - Caliper rugosity / excess over bit size                              (Eq. 1)
  - z-score feature standardization
  - k-means (k=2) good-hole / bad-hole clustering (TICC proxy)           (Eqs. 2-4)
  - Temporal (median-filter) smoothing of the cluster path (beta penalty)

Note: this issue's PDF has a text layer; the TICC negative-log-likelihood and
block-Toeplitz formulation (Eqs. 2-4) were image-rendered and are represented
here by a Gaussian k-means proxy with temporal smoothing - the same good/bad
hole partition on the same features.  Eqs. 1 and 5 are implemented directly.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- features ----------------

def rolling_cv(rhob, wr):
    """Rolling coefficient of variation of bulk density over +/- wr  (Eq. 5)."""
    return petrolib.data_qc_io.filt.moving_stat(rhob, 2 * wr + 1, "cv")


def caliper_excess(caliper, bit_size):
    """Caliper enlargement over bit size (rugosity proxy)  (Eq. 1)."""
    return np.maximum(np.asarray(caliper, float) - bit_size, 0.0)


def zscore(x):
    """Standardize to zero mean / unit variance."""
    x = np.asarray(x, float)
    return petrolib.ml_stats.zscore(x) if x.std() > 0 else x - x.mean()


# ---------------------------------------------- k-means -----------------

def kmeans(X, k=2, iters=50, seed=0):
    """Minimal k-means; returns (labels, centers).  X is (n_samples, n_feat)."""
    del seed  # historical, unused: the farthest-point init is deterministic
    return petrolib.ml_stats.kmeans(X, k, max_iter=iters)


def smooth_labels(labels, window=5):
    """Median-filter the cluster path (temporal beta-penalty proxy)."""
    labels = np.asarray(labels, int)
    n = len(labels)
    out = labels.copy()
    h = window // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        out[i] = int(round(np.median(labels[lo:hi])))
    return out


def detect_cavein(caliper, rhob, bit_size, wr=3):
    """Flag cave-in (bad-hole) samples from caliper + density logs.

    Clusters [caliper-excess, CV(rhob)] into two groups and labels the group
    with the larger mean caliper excess as the cave-in cluster, then temporally
    smooths the result.  Returns a boolean bad-hole mask.
    """
    exc = caliper_excess(caliper, bit_size)
    cv = rolling_cv(rhob, wr)
    X = np.column_stack([zscore(exc), zscore(cv)])
    labels, centers = kmeans(X, k=2)
    # the bad-hole cluster has the higher mean caliper excess (feature 0)
    bad_cluster = int(np.argmax(centers[:, 0]))
    return smooth_labels(labels == bad_cluster).astype(bool)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Wellbore Cave-in Detection (Clustering)")
    print("=" * 60)

    rng = np.random.default_rng(3)
    n = 120
    bit = 8.5
    caliper = np.full(n, 8.6) + 0.05 * rng.standard_normal(n)
    rhob = np.full(n, 2.45) + 0.01 * rng.standard_normal(n)
    # plant a cave-in zone: enlarged caliper + erratic, low density
    cave = slice(50, 70)
    caliper[cave] = 12.5 + 0.3 * rng.standard_normal(20)
    rhob[cave] = 2.1 + 0.25 * rng.standard_normal(20)

    cv = rolling_cv(rhob, wr=3)
    print(f"  CV in cave / good      = {cv[cave].mean():.4f} / {cv[:40].mean():.4f}")
    assert cv[cave].mean() > cv[:40].mean()        # density more erratic in cave

    bad = detect_cavein(caliper, rhob, bit)
    flagged = np.where(bad)[0]
    print(f"  flagged samples        = {flagged.min()}..{flagged.max()} "
          f"({bad.sum()} of {n})")
    # the planted cave-in interval is recovered with high overlap
    truth = np.zeros(n, bool); truth[cave] = True
    overlap = (bad & truth).sum() / truth.sum()
    false_rate = (bad & ~truth).sum() / (~truth).sum()
    print(f"  recall / false-flag    = {overlap:.2f} / {false_rate:.3f}")
    assert overlap > 0.8 and false_rate < 0.1
    print("  PASS")
    return {"recall": float(overlap), "false_rate": float(false_rate),
            "n_flagged": int(bad.sum())}


if __name__ == "__main__":
    test_all()
