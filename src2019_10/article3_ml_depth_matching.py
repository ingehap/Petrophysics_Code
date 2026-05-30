"""
Article 3: A Machine-Learning Framework for Automating Well-Log Depth Matching
Le, Liang, Zimmermann, Zeroug, Heliot (2019)
DOI: 10.30632/PJV60N5-2019a3

Logs acquired on different passes are depth-misaligned; this paper frames depth
matching as predicting the shift between a reference log and a desynchronized
pass.  The paper publishes no numbered equations; this module implements the
standard similarity metrics it builds on (cross-correlation, dynamic time
warping, Pearson correlation) and a windowed shift estimator.

Implements:

  - Normalized cross-correlation alignment lag
  - Dynamic time warping (DTW) distance
  - Pearson correlation coefficient
  - Windowed depth-shift estimation

Note: this issue's PDF has a text layer but the paper describes its
neural-network pipeline in prose with no numbered equations; these are the
standard alignment metrics the framework relies on (a numpy implementation of
the same similarity measures used to train / evaluate the matcher).
"""

import numpy as np


# ---------------------------------------------- similarity --------------

def pearson(a, b):
    """Pearson product-moment correlation coefficient."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def cross_correlation_lag(reference, shifted, max_lag=50):
    """Lag (samples) that best aligns `shifted` to `reference` by correlation.

    Positive lag means `shifted` must move up (toward smaller index) to align.
    """
    reference = np.asarray(reference, float)
    shifted = np.asarray(shifted, float)
    best_lag, best_c = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        s = np.roll(shifted, lag)
        c = np.corrcoef(reference, s)[0, 1]
        if c > best_c:
            best_c, best_lag = c, lag
    return best_lag, float(best_c)


def dtw_distance(a, b, band=None):
    """Dynamic time warping distance (Euclidean local cost)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n, m = len(a), len(b)
    INF = np.inf
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jlo = 1 if band is None else max(1, i - band)
        jhi = m if band is None else min(m, i + band)
        for j in range(jlo, jhi + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def estimate_shift(reference, shifted, max_lag=50):
    """Estimate the integer depth shift via cross-correlation."""
    return cross_correlation_lag(reference, shifted, max_lag)[0]


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: ML Well-Log Depth Matching")
    print("=" * 60)

    # Synthetic gamma-ray log with structure
    rng = np.random.default_rng(3)
    n = 300
    ref = np.cumsum(rng.normal(0, 1, n))
    ref += 5 * np.sin(np.linspace(0, 12, n))

    # A pass shifted down by 8 samples: cross-correlation recovers the shift
    shift_true = 8
    shifted = np.roll(ref, shift_true) + rng.normal(0, 0.2, n)
    lag, c = cross_correlation_lag(ref, shifted, max_lag=30)
    print(f"  recovered lag / corr   = {lag} / {c:.3f}")
    assert lag == -shift_true                      # undo the +8 roll
    assert c > 0.95

    # After alignment, Pearson correlation is high; before, it is lower
    aligned = np.roll(shifted, lag)
    assert pearson(ref, aligned) > pearson(ref, shifted)

    # DTW distance is smaller for aligned logs than misaligned ones
    d_aligned = dtw_distance(ref[20:120], aligned[20:120], band=15)
    d_mis = dtw_distance(ref[20:120], shifted[20:120], band=15)
    print(f"  DTW aligned / misaligned = {d_aligned:.1f} / {d_mis:.1f}")
    assert d_aligned < d_mis

    # estimate_shift convenience wrapper
    assert estimate_shift(ref, shifted, max_lag=30) == -shift_true
    print("  PASS")
    return {"lag": lag, "corr": c, "dtw_aligned": d_aligned}


if __name__ == "__main__":
    test_all()
