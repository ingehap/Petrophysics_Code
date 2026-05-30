"""
Article 3: Automated Log Data Analytics Workflow - The Value of Data Access
           and Management to Reduced Turnaround Time for Log Analysis
Torres Caceres, Duffaut, Westad, Stovas, Johansen, Jenssen (2022)
DOI: 10.30632/PJV63N1-2022a3

Two depth-matching engines for synchronizing logs from different passes
(LWD/MWD vs EWL) to a common depth reference, plus the QC metrics used to
score the match:

  - Cross correlation, optionally with stretch/squeeze factor alpha (Eq. 1)
  - Constrained dynamic time warping (DTW), Sakoe-Chiba band      (Eqs. 2-4)
  - Pearson correlation                                           (Eq. A1.1)
  - Trace energy  TE = Sum(x^2)                                   (Eq. A1.2)
  - Residual energy  RE = Sum((x-y)^2)                            (Eq. A1.3)
  - Predictability  P = 1 - RE/TE                                 (Eq. A1.4)
  - Euclidean distance                                            (Eq. A1.5)

The paper builds these on scipy / dtaidistance; here they are implemented
directly in numpy so the module is dependency-light and self-contained.
"""

import numpy as np

NODATA = -999.25             # log no-data sentinel
SAMPLING_FT = 0.5            # standard depth sampling rate (ft)


# ---------------------------------------------- Eq. 1: cross correlation -

def cross_correlation_shift(reference, test, max_shift=40):
    """Bulk depth shift from normalized cross correlation (Eq. 1).

    Returns the integer lag (samples) maximizing the correlation of `test`
    with `reference`.
    """
    r = _standardize(reference)
    t = _standardize(test)
    lags = np.arange(-max_shift, max_shift + 1)
    cc = [float(np.dot(r, np.roll(t, L))) for L in lags]
    return int(lags[int(np.argmax(cc))])


def cross_correlation_alpha(reference, test, alphas=None, max_shift=40):
    """Joint (alpha, shift) search with stretch/squeeze factor alpha (Eq. 1).

    alpha < 1 stretches the test log, alpha > 1 squeezes it.  Returns
    (alpha, shift, correlation).
    """
    if alphas is None:
        alphas = np.round(np.arange(0.75, 1.26, 0.05), 2)
    r = _standardize(reference)
    n = len(reference)
    test = np.asarray(test, float)
    src_idx = np.arange(len(test))
    best = (1.0, 0, -np.inf)
    for a in alphas:
        # stretch/squeeze: reference position i samples test at index i*alpha
        res = np.interp(np.arange(n) * a, src_idx, test)
        t = _standardize(res)
        for L in range(-max_shift, max_shift + 1):
            c = float(np.dot(r, np.roll(t, L)))
            if c > best[2]:
                best = (float(a), L, c)
    return best


# ---------------------------------------------- Eqs. 2-4: DTW -----------

def dtw_distance(x, y, band=50):
    """Constrained dynamic time warping (Eqs. 2-4).

    Local dissimilarity d(i,j) = (x_i - y_j)^2 (Eq. 2); accumulated cost is
    minimized over all admissible warping paths within a Sakoe-Chiba band of
    half-width `band` (Eqs. 3-4).  Returns (distance, path).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        for j in range(j0, j1 + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # backtrace the optimal path
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return float(np.sqrt(D[n, m])), path[::-1]


# ---------------------------------------------- Appendix 1: QC metrics --

def pearson(x, y):
    """Pearson correlation coefficient r (Eq. A1.1)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    d = np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2))
    return float(np.sum(xc * yc) / d) if d > 1e-12 else 0.0


def trace_energy(x):
    """TE = Sum(x_i^2)  (Eq. A1.2)."""
    return float(np.sum(np.asarray(x, float) ** 2))


def residual_energy(x, y):
    """RE = Sum((x_i - y_i)^2)  (Eq. A1.3)."""
    return float(np.sum((np.asarray(x, float) - np.asarray(y, float)) ** 2))


def predictability(x, y):
    """PEP  P = 1 - RE/TE  (Eq. A1.4).  ~1 is a good match, can be negative."""
    te = trace_energy(x)
    return 1.0 - residual_energy(x, y) / te if te > 1e-12 else 0.0


def euclidean(x, y):
    """L2 distance (Eq. A1.5)."""
    return float(np.linalg.norm(np.asarray(x, float) - np.asarray(y, float)))


# ---------------------------------------------- helpers ----------------

def _standardize(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


def first_last_valid(log, nodata=NODATA):
    """Index of first and last samples that are not the no-data sentinel."""
    valid = np.where(np.asarray(log) != nodata)[0]
    return (int(valid[0]), int(valid[-1])) if valid.size else (None, None)


def apply_shift(test, shift):
    """Shift a log by an integer number of samples (positive = deeper)."""
    return np.roll(np.asarray(test, float), shift)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Cross-Correlation / DTW Depth Matching + QC")
    print("=" * 60)

    rng = np.random.default_rng(3)
    n = 400
    ref = np.cumsum(rng.normal(0, 1, n)) + 20 * np.sin(np.linspace(0, 9 * np.pi, n))

    true_shift = -13
    test = apply_shift(ref, true_shift) + rng.normal(0, 1.0, n)

    # no-data handling
    test_nd = test.copy(); test_nd[:5] = NODATA
    i0, i1 = first_last_valid(test_nd)
    print(f"  first/last valid index = {i0}, {i1}")
    assert i0 == 5

    # cross correlation should recover the alignment lag (= -applied shift)
    s = cross_correlation_shift(ref, test, max_shift=40)
    print(f"  cross-corr lag         = {s} samples (applied {true_shift})")
    assert abs(abs(s) - abs(true_shift)) <= 1

    # QC metrics before / after applying the shift
    r0 = pearson(ref, test)
    r1 = pearson(ref, apply_shift(test, s))
    p0 = predictability(ref, test)
    p1 = predictability(ref, apply_shift(test, s))
    d0 = euclidean(ref, test)
    d1 = euclidean(ref, apply_shift(test, s))
    print(f"  Pearson   before/after = {r0:.3f} / {r1:.3f}")
    print(f"  PEP (P)   before/after = {p0:.3f} / {p1:.3f}")
    print(f"  Euclidean before/after = {d0:.1f} / {d1:.1f}")
    assert r1 > r0 and p1 > p0 and d1 < d0

    # alpha (stretch/squeeze) search on a pure bulk shift -> alpha ~ 1
    a, sa, ca = cross_correlation_alpha(ref, test, max_shift=40)
    print(f"  alpha search           = alpha={a:.2f}, shift={sa}")
    assert abs(a - 1.0) <= 0.051, "pure bulk shift should prefer alpha ~ 1"

    # DTW on a short pair
    xa = ref[50:170]
    xb = apply_shift(ref, -4)[50:170] + rng.normal(0, 0.5, 120)
    dist, path = dtw_distance(xa, xb, band=30)
    print(f"  DTW distance           = {dist:.2f} over {len(path)} steps")
    assert dist >= 0 and len(path) >= len(xa)
    print("  PASS")
    return {"shift": s, "pearson_after": r1, "pep_after": p1, "dtw": dist}


if __name__ == "__main__":
    test_all()
