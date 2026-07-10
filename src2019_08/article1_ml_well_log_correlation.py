"""
Article 1: A Machine-Learning-Based Approach to Assistive Well-Log Correlation
Brazell, Bayeh, Ashby, Burton (2019)
DOI: 10.30632/PJV60N4-2019a1

Stratigraphic correlation ties markers between offset wells whose logs are
shifted and stretched relative to one another.  This module implements the
quantitative core the assistive workflow relies on: cross-correlation and
dynamic time warping to align two wells' logs, a warping path that maps a marker
depth from a reference well to an offset well, and a confidence score for each
tie.

Implements:

  - Pearson correlation and cross-correlation alignment lag
  - Dynamic time warping (DTW) distance and warping path
  - Marker depth mapping between wells via the DTW path
  - Logistic tie-confidence score from feature similarity

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the issue's table of contents and the
numbered formulas are faithful standard-form reconstructions of the
correlation/alignment methods the paper applies.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- similarity --------------

def pearson(a, b):
    """Pearson correlation coefficient."""
    return petrolib.ml_stats.pearson_r(a, b)


def cross_correlation_lag(reference, other, max_lag=80):
    """Integer lag aligning `other` to `reference` by maximum correlation."""
    r = petrolib.depth_matching.xcorr_shift(reference, other, max_lag=max_lag, edge="wrap")
    return r.lag, r.corr


def dtw(a, b, band=None):
    """Dynamic time warping: returns (distance, accumulated-cost matrix)."""
    res = petrolib.depth_matching.dtw(a, b, band=band, root=True)
    return res.distance, res.cost


def warping_path(D):
    """Backtrack the DTW accumulated-cost matrix to the warping path."""
    i, j = D.shape[0] - 1, D.shape[1] - 1
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
        path.append((i - 1, j - 1))
    return path[::-1]


def map_marker(ref_idx, path):
    """Map a reference-well sample index to the offset well via the DTW path."""
    matches = [j for (i, j) in path if i == ref_idx]
    return int(np.mean(matches)) if matches else None


def tie_confidence(feat_ref, feat_other, scale=1.0):
    """Logistic tie-confidence from feature distance (1 = identical features)."""
    d = np.linalg.norm(np.asarray(feat_ref, float) - np.asarray(feat_other, float))
    return float(1.0 / (1.0 + np.exp(scale * d - 2.0)))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: ML-Based Assistive Well-Log Correlation")
    print("=" * 60)

    rng = np.random.default_rng(1)
    n = 240
    # Reference-well gamma ray with marker beds
    ref = np.cumsum(rng.normal(0, 0.5, n)) + 4 * np.sin(np.linspace(0, 10, n))

    # Offset well: same geology shifted down 12 samples (different datum)
    shift = 12
    offset = np.roll(ref, shift) + rng.normal(0, 0.3, n)
    lag, c = cross_correlation_lag(ref, offset)
    print(f"  cross-correlation lag  = {lag}  (corr {c:.3f})")
    assert lag == -shift and c > 0.95

    # DTW aligns a stretched offset well; path maps a marker depth
    idx = np.linspace(0, n - 1, n)
    stretched = np.interp(idx, idx * 0.9, ref)        # 10% stretch
    dist, D = dtw(ref[::3], stretched[::3], band=20)   # decimate for speed
    path = warping_path(D)
    m_ref = 40
    m_off = map_marker(m_ref, path)
    print(f"  marker ref {m_ref} -> offset {m_off}")
    assert m_off is not None and 0 <= m_off < len(ref[::3])

    # Tie confidence is high for similar features, low for dissimilar ones
    conf_same = tie_confidence([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    conf_diff = tie_confidence([1.0, 2.0, 3.0], [5.0, 1.0, 8.0])
    print(f"  tie confidence same/diff = {conf_same:.2f} / {conf_diff:.2f}")
    assert conf_same > 0.8 and conf_diff < conf_same
    print("  PASS")
    return {"lag": lag, "corr": c, "conf_same": conf_same}


if __name__ == "__main__":
    test_all()
