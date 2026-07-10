"""
Article 10: Machine-Learning-Based Automatic Well-Log Depth Matching
Zimmermann, Liang, Zeroug (2018)
DOI: 10.30632/PJV59N6Y2018a9

Logs from different runs/passes are misaligned in depth.  Automatic depth
matching aligns a target log to a reference by finding the depth shift (and local
stretch) that maximizes their similarity; cross-correlation gives the bulk shift
and dynamic time warping handles local stretch, with the Pearson correlation as
the post-alignment quality metric.

Implements:

  - Pearson correlation and cross-correlation alignment lag
  - Dynamic time warping (DTW) distance
  - Windowed local-shift estimation along the log

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard cross-correlation / DTW depth-matching the paper
describes.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


def pearson(a, b):
    """Pearson correlation coefficient."""
    return petrolib.ml_stats.pearson_r(a, b)


def cross_correlation_lag(reference, target, max_lag=60):
    """Integer lag aligning `target` to `reference` by maximum correlation."""
    r = petrolib.depth_matching.xcorr_shift(reference, target, max_lag=max_lag, edge="wrap")
    return r.lag, r.corr


def dtw_distance(a, b, band=None):
    """Dynamic time warping distance (Euclidean local cost)."""
    return petrolib.depth_matching.dtw(a, b, band=band, root=True).distance


def local_shifts(reference, target, window=60, step=30, max_lag=20):
    """Estimate the depth shift in successive windows along the logs.

    Uses a non-wrapping full cross-correlation per window (np.roll would wrap
    short windows and corrupt the lag estimate).
    """
    return petrolib.depth_matching.local_shifts(
        reference, target, window=window, step=step, max_lag=max_lag).astype(int)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 10: ML-Based Automatic Well-Log Depth Matching")
    print("=" * 60)

    rng = np.random.default_rng(9)
    n = 400
    # higher-frequency structure so short correlation windows have a sharp peak
    ref = np.cumsum(rng.normal(0, 0.5, n)) + 4 * np.sin(np.linspace(0, 40, n))

    # Bulk shift of 15 samples recovered by cross-correlation
    shift = 15
    target = np.roll(ref, shift) + rng.normal(0, 0.2, n)
    lag, c = cross_correlation_lag(ref, target)
    print(f"  bulk lag / correlation = {lag} / {c:.3f}")
    assert lag == -shift and c > 0.95

    # Alignment improves the Pearson correlation
    assert pearson(ref, np.roll(target, lag)) > pearson(ref, target)

    # DTW distance smaller for aligned than misaligned
    aligned = np.roll(target, lag)
    assert dtw_distance(ref[30:130], aligned[30:130], band=15) < \
        dtw_distance(ref[30:130], target[30:130], band=15)

    # Windowed local shifts recover the (constant) bulk shift
    ls = local_shifts(ref, target, window=80, step=30, max_lag=25)
    print(f"  local shifts (median)  = {np.median(ls):.0f}")
    assert abs(np.median(ls) + shift) < 2.0
    print("  PASS")
    return {"lag": lag, "corr": c}


if __name__ == "__main__":
    test_all()
