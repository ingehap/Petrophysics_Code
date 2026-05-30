"""
Article 7: Learnings From a New Slimhole LWD NMR Technology
Hursan, Silva, Van Steene, Muna (2022)
DOI: 10.30632/PJV63N3-2022a7

Body text was not present in the available PDF extract, so this module
is a *methodology proxy* guided by the Guest Editor's summary:
slimhole LWD NMR tool that acquires T1 and T2 simultaneously across
clastic and carbonate formations.  Factor analysis is applied to the
LWD log to recover pore-fluid components, then validated against an
offset wireline log.  Time-lapse LWD-vs-wireline data give pore-scale
insight into oil-based-mud filtrate invasion.

Implements:

  - Factor analysis (PCA + varimax-like rotation) of a stacked
    NMR-log feature matrix: each row is one depth, each column is a
    bin of the T1 or T2 distribution.
  - Reconstruction of the dominant factors as pore-fluid spectra.
  - LWD-vs-wireline time-lapse difference map for invasion analysis.
"""

import numpy as np


# ----------------------------------------------- factor analysis ---------

def pca_factor_analysis(X, n_factors=4):
    """SVD-based factor analysis.  Returns loadings (n_features, n_factors)
    and scores (n_samples, n_factors)."""
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    loadings = Vt[:n_factors].T * s[:n_factors]
    scores = U[:, :n_factors]
    explained = (s ** 2).cumsum() / (s ** 2).sum()
    return loadings, scores, explained[:n_factors]


def varimax_rotation(loadings, n_iter=100, tol=1e-6):
    """Kaiser varimax rotation for interpretability."""
    p, k = loadings.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(n_iter):
        d_old = d
        Lambda = loadings @ R
        u, s, vh = np.linalg.svd(loadings.T @ (Lambda ** 3
                                               - (1.0 / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))))
        R = u @ vh
        d = float(s.sum())
        if abs(d - d_old) < tol:
            break
    return loadings @ R


# ----------------------------------------------- synthetic LWD log -----

def synth_lwd_nmr_log(n_depths=120, n_bins=32, seed=0):
    """Stack of T2 distributions vs depth - three "fluid factors":
    bound water, oil and free water - with depth-varying weights."""
    rng = np.random.default_rng(seed)
    T2 = np.logspace(-1, 3, n_bins)
    factors = []
    for centre, sigma in [(2.0, 0.3), (40.0, 0.3), (500.0, 0.3)]:
        f = np.exp(-((np.log10(T2) - np.log10(centre)) / sigma) ** 2)
        factors.append(f / f.sum())
    F = np.array(factors).T                 # (n_bins, 3)
    # Independent log-uniform weights (NOT Dirichlet - Dirichlet samples
    # are rank-deficient because they sum to 1, which would prevent PCA
    # from recovering all three sources).
    weights = rng.uniform(0.05, 1.0, size=(n_depths, 3))
    weights = np.array([np.convolve(weights[:, k], np.ones(5) / 5, mode="same")
                        for k in range(3)]).T
    X = weights @ F.T + 0.02 * rng.standard_normal((n_depths, n_bins))
    return X, T2, weights, F


# ----------------------------------------------- LWD-wireline diff ---

def time_lapse_diff(lwd_log, wireline_log):
    """Per-depth absolute difference of T2 distributions, normalised."""
    diff = lwd_log - wireline_log
    return diff / (np.abs(wireline_log).max() + 1e-9)


# ----------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Slimhole LWD NMR + Factor Analysis (proxy)")
    print("=" * 60)

    X, T2, weights, F_true = synth_lwd_nmr_log()
    loadings, scores, explained = pca_factor_analysis(X, n_factors=3)
    loadings_rot = varimax_rotation(loadings)
    print(f"  Cumulative explained variance (top 3 factors) = "
          f"{[round(e, 3) for e in explained]}")
    # Top factor must dominate noise; cumulative top-3 should clear the
    # bulk of the signal variance (the residual is noise + smoothed weights).
    assert explained[0] > 0.10, "Top factor must explain > 10 % of variance"

    # Match each recovered loading to a planted factor by max correlation
    matched_corrs = []
    for k in range(3):
        cs = [abs(np.corrcoef(np.abs(loadings_rot[:, k]),
                              F_true[:, j])[0, 1])
              for j in range(3)]
        matched_corrs.append(max(cs))
    print(f"  Per-factor max correlation with truth   = "
          f"{[round(c, 3) for c in matched_corrs]}")
    assert min(matched_corrs) > 0.70, "Each factor must correspond to a planted source"

    # LWD vs wireline time-lapse "invasion" simulation:
    # wireline taken N depths later -> oil-zone signal shifted by mud filtrate
    wireline = X.copy()
    wireline[60:80] -= 0.05 * F_true[:, 1]      # methane / oil signal weakens
    wireline[60:80] += 0.05 * F_true[:, 0]      # bound water increases
    diff = time_lapse_diff(X, wireline)
    print(f"  Mean LWD-wireline diff inside invaded zone (depth 60-80): "
          f"{abs(diff[60:80]).mean():.3f}")
    print(f"  Mean LWD-wireline diff outside invaded zone:                "
          f"{abs(diff[:60]).mean():.3f}")
    assert abs(diff[60:80]).mean() > 2.0 * abs(diff[:60]).mean()
    print("  PASS")
    return {"explained_top3": float(explained[-1]),
            "min_factor_corr": float(min(matched_corrs))}


if __name__ == "__main__":
    test_all()
