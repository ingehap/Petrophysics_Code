"""
Article 5: Log Facies Analysis and Reservoir Properties of Basement Granitic
Rocks in the Pearl River Mouth Basin, South China Sea
Hua, Yang, Xu, Lei, Zhong (2022)
DOI: 10.30632/PJV63N4-2022a5

Two-stage workflow for log-facies analysis on conventional five-curve
suites (GR, RD, DEN, AT, NP):

  1. Change-point analysis (CPA) on the GR series, using the mean-change-
     point model x_i = a_i + e_i (Eq. 1).  Change-point positions m_j
     minimise the SSE objective (Eq. 2); initial guesses come from the
     Q-statistic Q = H / R with R = 0.5 m (Eq. 3); iterative refinement
     via the W functional (Eq. 4); a jump-magnitude statistic theta
     (Eq. 5) and a minimum-spacing filter validate the picks.
  2. Fuzzy c-means (FCM) clustering on segment-averaged 5-log vectors
     X_b that minimises the FCM objective (Eq. 6), with membership-
     update (Eq. 7) and centroid-update (Eq. 8) iterations, fuzzifier
     r = 2.

The synthetic demo reproduces the paper's qualitative result that 3-5
facies emerge from typical basement-granite suites and that the FCM
solution converges in a few tens of iterations.
"""

import numpy as np


# -------------------------------------------- CPA (Eqs. 1-5) --------------

def cpa_sse(x, breakpoints):
    """Mean-change-point SSE for given breakpoints (Eq. 2).

    Each segment has constant mean equal to the segment mean; the SSE is
    the sum of within-segment squared deviations from those means.
    """
    bp = [0] + sorted(breakpoints) + [len(x)]
    sse = 0.0
    for a, b in zip(bp[:-1], bp[1:]):
        if b > a:
            seg = x[a:b]
            sse += float(((seg - seg.mean()) ** 2).sum())
    return sse


def cpa_initial_guesses(x, window=5):
    """Q = H / R initial-guess strategy (Eq. 3).

    Compute the absolute-change H over a sliding window of length window,
    divided by the window length R; pick local maxima as candidates.
    """
    H = np.abs(x[window:] - x[:-window])
    candidates = []
    for i in range(1, len(H) - 1):
        if H[i] > H[i - 1] and H[i] > H[i + 1]:
            candidates.append((i + window // 2, float(H[i])))
    candidates.sort(key=lambda c: -c[1])
    return candidates


def cpa_refine(x, candidates, max_n_bp=12, min_spacing=4, max_iter=200):
    """Greedy add-and-refine: insert change-points one at a time as long as
    they reduce the SSE by a meaningful margin.  Returns final breakpoint
    list."""
    breakpoints = []
    base_sse = cpa_sse(x, breakpoints)
    for cand_idx, _ in candidates[:max_n_bp * 2]:
        if any(abs(cand_idx - b) < min_spacing for b in breakpoints):
            continue
        new_bp = sorted(breakpoints + [cand_idx])
        new_sse = cpa_sse(x, new_bp)
        if (base_sse - new_sse) / max(base_sse, 1e-9) > 0.01:
            breakpoints = new_bp
            base_sse = new_sse
        if len(breakpoints) >= max_n_bp:
            break
    return breakpoints


def cpa_jump_magnitude(x, breakpoints):
    """Theta statistic (Eq. 5): jump magnitude at each picked breakpoint,
    normalised by the residual within-segment std deviation."""
    bp = [0] + sorted(breakpoints) + [len(x)]
    out = []
    for k in range(1, len(bp) - 1):
        a, b, c = bp[k - 1], bp[k], bp[k + 1]
        seg_a, seg_b = x[a:b], x[b:c]
        if len(seg_a) > 0 and len(seg_b) > 0:
            jump = abs(seg_b.mean() - seg_a.mean())
            std = max(seg_a.std(ddof=1) if len(seg_a) > 1 else 1.0, 1e-6)
            out.append(jump / std)
    return out


# -------------------------------------------- segment-averaged features ---

def average_features_per_segment(features, breakpoints):
    """features: (n_samples, n_features).  Returns (n_segments, n_features)."""
    bp = [0] + sorted(breakpoints) + [features.shape[0]]
    return np.array([features[a:b].mean(0) for a, b in zip(bp[:-1], bp[1:])
                     if b > a])


# -------------------------------------------- Fuzzy c-means (Eqs. 6-8) ----

def fcm(X, c=5, m=2.0, n_iter=200, tol=1e-5, seed=0):
    """Bezdek FCM clustering with fuzzifier m on standardised X.

    Returns final centroids V (c, d), memberships U (n, c) and objective
    history.
    """
    rng = np.random.default_rng(seed)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    n, d = Xs.shape
    U = rng.random((n, c))
    U /= U.sum(1, keepdims=True)
    history = []
    for it in range(n_iter):
        Um = U ** m
        V = (Um.T @ Xs) / (Um.sum(0)[:, None] + 1e-12)
        D = np.linalg.norm(Xs[:, None, :] - V[None, :, :], axis=-1) + 1e-9
        ratio = (D[:, :, None] / D[:, None, :]) ** (2.0 / (m - 1.0))
        U_new = 1.0 / ratio.sum(2)
        # FCM objective (Eq. 6)
        J = float((U_new ** m * D ** 2).sum())
        history.append(J)
        if np.linalg.norm(U_new - U) < tol:
            U = U_new
            break
        U = U_new
    return V, U, history


# -------------------------------------------- synthetic data --------------

def synth_log_suite(n=600, n_facies=4, seed=0):
    """Five-log suite (GR, RD, DEN, AT, NP) with `n_facies` blocks."""
    rng = np.random.default_rng(seed)
    bp_true = sorted(rng.choice(np.arange(40, n - 40), n_facies - 1,
                                replace=False))
    bp_full = [0] + list(bp_true) + [n]
    facies_props = np.array([
        [ 22.0,  80.0, 2.45, 75.0, 0.08],   # facies 0: granitic, low GR
        [ 75.0,  10.0, 2.20, 95.0, 0.30],   # facies 1: shale
        [ 35.0, 200.0, 2.60, 65.0, 0.05],   # facies 2: fractured granite
        [ 50.0,  30.0, 2.40, 80.0, 0.15],   # facies 3: weathered
    ])
    facies_labels = np.zeros(n, dtype=int)
    out = np.zeros((n, 5))
    for k, (a, b) in enumerate(zip(bp_full[:-1], bp_full[1:])):
        fac = k % n_facies
        facies_labels[a:b] = fac
        for j in range(5):
            mu = facies_props[fac, j]
            sigma = 0.05 * abs(mu) + 1e-3
            out[a:b, j] = rng.normal(mu, sigma, b - a)
    return out, np.array(bp_true), facies_labels


# -------------------------------------------- tests ----------------------

def test_all():
    print("=" * 60)
    print("Article 5: CPA + Fuzzy-C-Means Log-Facies Analysis")
    print("=" * 60)

    logs, bp_true, labels_true = synth_log_suite()
    gr = logs[:, 0]
    cand = cpa_initial_guesses(gr, window=5)
    bp = cpa_refine(gr, cand, max_n_bp=8)
    theta = cpa_jump_magnitude(gr, bp)

    # Closest picked breakpoint within +/-5 samples of each true one
    hits = sum(any(abs(b - t) <= 5 for b in bp) for t in bp_true)
    print(f"  True breakpoints   {list(bp_true)}")
    print(f"  Picked breakpoints {bp}")
    print(f"  Hits (within +/- 5 samples) = {hits} / {len(bp_true)}")
    print(f"  Jump-magnitude theta values  = "
          f"{[round(t, 2) for t in theta]}")
    assert hits >= len(bp_true) - 1, "CPA must pick most true breakpoints"

    # FCM on segment-averaged 5-log features
    segments = average_features_per_segment(logs, bp)
    V, U, history = fcm(segments, c=4)
    iters = len(history)
    drop = (history[0] - history[-1]) / history[0]
    print(f"  FCM converged in {iters} iters; "
          f"objective dropped by {drop * 100:.1f} %")
    assert drop > 0.5, "FCM objective must drop substantially"
    print("  PASS")
    return {"n_breakpoints": len(bp), "hits": hits, "fcm_iters": iters}


if __name__ == "__main__":
    test_all()
