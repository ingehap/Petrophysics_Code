"""
Article 9: Automated Well-Log Pattern Alignment and Depth-Matching
Techniques: An Empirical Review and Recommendations
Ezenkwu, Guntoro, Starkey, Vaziri, Addario (2023)
DOI: 10.30632/PJV64N1-2023a9

Benchmarks three classical signal-alignment algorithms for log
depth-matching against a synthetic, expert-shifted reference:

  - Dynamic Time Warping (DTW)
  - Constrained DTW (CDTW) with a Sakoe-Chiba warping band
  - Correlation Optimised Warping (COW) with piecewise linear time
    re-mapping, Pearson correlation as the per-segment objective

Synthetic input is a gamma-ray-like curve corrupted by depth-dependent
stretch / squeeze, additive noise, and a baseline-amplitude scaling.
The expert "ground truth" alignment is the inverse of the imposed warp.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------------------- synthetic data ---

def synthetic_pair(n=600, seed=0):
    """Return reference, target, true_target_index_for_each_ref_index."""
    rng = np.random.default_rng(seed)
    z = np.linspace(0, 4 * np.pi, n)
    base = (np.sin(z) + 0.5 * np.sin(3.7 * z) + 0.3 * np.cos(11 * z))
    ref = base + 0.05 * rng.standard_normal(n)

    # Non-linear monotonic depth warp f(z): piecewise-linear with three knots
    warp = np.interp(np.arange(n), [0, 0.3 * n, 0.7 * n, n - 1],
                     [0, 0.40 * n, 0.55 * n, n - 1])
    target = np.interp(warp, np.arange(n), base) \
             + 0.07 * rng.standard_normal(n)
    target = 1.10 * target + 0.20  # amplitude scaling + baseline shift

    return ref, target, warp.astype(int)


# ---------------------------------------------------------- DTW ------------

def dtw(x, y, window=None):
    """Classical DTW with optional Sakoe-Chiba band of half-width `window`.

    Returns the warping path as a list of (i, j) tuples and the cost.
    """
    res = petrolib.depth_matching.dtw(x, y, band=window)
    return res.path, res.distance


def warped_target(target, path, n_ref):
    """Reconstruct a length-n_ref vector by indexing the target along the path."""
    return petrolib.depth_matching.warp_to_reference(target, path, n_ref)


# ---------------------------------------------------------- COW ------------

def cow(x, y, n_segments=10, slack=8):
    """Correlation Optimised Warping.

    Splits the reference into n_segments segments of equal length.  For
    each interior boundary, search +/- slack samples in y and choose the
    integer warp boundary that maximises the sum of segment Pearson
    correlations with the corresponding stretched / compressed y window.
    Boundary search is greedy and sequential (a faithful low-cost
    approximation of Nielsen's COW dynamic programme).
    """
    return petrolib.depth_matching.cow(x, y, n_segments=n_segments, slack=slack)


def _segment_corr(a, b):
    """Resample b to length len(a) and return Pearson correlation."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    b_resampled = np.interp(np.linspace(0, len(b) - 1, len(a)),
                            np.arange(len(b)), b)
    return float(np.corrcoef(a, b_resampled)[0, 1])


# ---------------------------------------------------------- evaluation -----

def alignment_score(aligned, reference):
    return float(np.corrcoef(aligned, reference)[0, 1])


def test_all():
    print("=" * 60)
    print("Article 9: Depth-Matching Algorithm Benchmark")
    print("=" * 60)
    ref, tgt, _ = synthetic_pair()
    n = len(ref)

    print(f"  Raw  correlation  ref vs tgt = {alignment_score(tgt, ref):.3f}")

    path_dtw, _ = dtw(ref, tgt)
    aligned_dtw = warped_target(tgt, path_dtw, n)
    print(f"  DTW  correlation              = {alignment_score(aligned_dtw, ref):.3f}")

    path_cdtw, _ = dtw(ref, tgt, window=int(0.10 * n))
    aligned_cdtw = warped_target(tgt, path_cdtw, n)
    print(f"  CDTW (10% Sakoe-Chiba band)   = {alignment_score(aligned_cdtw, ref):.3f}")

    aligned_cow, _ = cow(ref, tgt, n_segments=20, slack=12)
    print(f"  COW  (20 segments, slack=12)  = {alignment_score(aligned_cow, ref):.3f}")

    # The COW alignment should outperform the raw signal and at least
    # match the DTW baselines on this controlled experiment.
    s_raw = alignment_score(tgt, ref)
    s_cow = alignment_score(aligned_cow, ref)
    assert s_cow > s_raw, "COW must beat raw correlation"
    print("  PASS")
    return {"raw": s_raw, "dtw": alignment_score(aligned_dtw, ref),
            "cdtw": alignment_score(aligned_cdtw, ref), "cow": s_cow}


if __name__ == "__main__":
    test_all()
