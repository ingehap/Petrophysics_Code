"""
Article 5: An Unsupervised Machine-Learning Workflow for Outlier Detection and
Log Editing With Prediction Uncertainty
Akkurt, Conroy, Psaila, Paxton, Low, and Spaans (2023)
DOI: 10.30632/PJV64N2-2023a5

Implements the five-step workflow:
  1. Data scanning + standardisation
  2. One-Class SVM outlier detection with inflection-point algorithm
  3. Inter-well similarity (Jaccard / Overlap) on per-well "footprints"
     + Multidimensional Scaling (MDS) for clustering
  4. Regression to reconstruct logs (k-NN ensemble for uncertainty)
  5. Uncertainty-based QC flag
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


# ------------------------------------------------------- outlier detection --

def ocsvm_footprint(X, nu=0.05, gamma="scale"):
    """Fit a One-Class SVM and return the model + scores."""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma).fit(Xs)
    scores = model.decision_function(Xs)
    return model, scaler, scores


def inflection_outlier_fraction(X, fractions=None):
    """
    Scan outlier fraction values and find the elbow of the fraction-vs-
    SVM-score curve.  Mirrors the inflection-point approach from the paper:
    increasing the outlier fraction beyond the elbow yields slowly-changing
    scores.

    We use the maximum-distance-from-chord (kneedle) criterion.
    """
    fractions = fractions if fractions is not None else np.arange(0.005, 0.30, 0.005)
    score_threshold = []
    Xs = StandardScaler().fit_transform(X)
    for nu in fractions:
        nu_eff = float(np.clip(nu, 1e-3, 0.5))
        m = OneClassSVM(kernel="rbf", nu=nu_eff, gamma="scale").fit(Xs)
        s = m.decision_function(Xs)
        thr = np.quantile(s, nu_eff)
        score_threshold.append(thr)
    score_threshold = np.array(score_threshold)
    fx = (fractions - fractions.min()) / (fractions.max() - fractions.min() + 1e-9)
    fy = (score_threshold - score_threshold.min()) / \
         (score_threshold.max() - score_threshold.min() + 1e-9)
    # the curve rises (score threshold goes up with more outliers).  The
    # 'knee' is the point with maximum vertical distance from the chord
    # joining (0,0) -> (1,1).  Because the curve is concave we look for the
    # most positive (fy - fx).
    distances = fy - fx
    idx = int(np.argmax(distances))
    return fractions[idx], score_threshold


def detect_outliers(X, nu=None):
    """End-to-end outlier flag (1 = outlier, 0 = inlier)."""
    if nu is None:
        nu, _ = inflection_outlier_fraction(X)
    nu = float(np.clip(nu, 1e-3, 0.5))
    model, scaler, scores = ocsvm_footprint(X, nu=nu)
    pred = model.predict(scaler.transform(X))   # +1 inlier, -1 outlier
    return (pred == -1).astype(int), scores, nu


# ------------------------------------------------------- well similarity --

def well_footprint_grid(X, bins=10):
    """Bin a well's logs into a 2D occupancy grid (using first two logs)."""
    a, b = X[:, 0], X[:, 1]
    H, _, _ = np.histogram2d(a, b, bins=bins)
    return (H > 0).astype(int)  # binary footprint


def jaccard(A, B):
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return inter / max(union, 1)


def overlap(A, B):
    inter = np.logical_and(A, B).sum()
    return inter / max(min(A.sum(), B.sum()), 1)


def well_similarity(footprints, metric="jaccard"):
    n = len(footprints)
    M = np.zeros((n, n))
    fn = jaccard if metric == "jaccard" else overlap
    for i in range(n):
        for j in range(n):
            M[i, j] = fn(footprints[i], footprints[j])
    return M


def mds_embedding(sim, n_components=2, seed=0):
    """Convert similarity to dissimilarity then run classical MDS."""
    diss = 1.0 - sim
    np.fill_diagonal(diss, 0.0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        mds = MDS(n_components=n_components, dissimilarity="precomputed",
                  random_state=seed, normalized_stress="auto", n_init=4)
        return mds.fit_transform(diss)


# ------------------------------------------------------- reconstruction --

def reconstruct_with_uncertainty(X_train, y_train, X_query, k=5):
    """
    Use a kNN ensemble to predict y at query points and return mean +
    predictive standard deviation.
    """
    knn = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
    # build predictive distribution from neighbour values
    distances, indices = knn.kneighbors(X_query)
    yk = y_train[indices]
    mean = yk.mean(axis=1)
    std = yk.std(axis=1)
    return mean, std


# ----------------------------------------------------------- testing ---

def synthetic_well(seed, n=300, contamination=0.05):
    rng = np.random.default_rng(seed)
    z = np.arange(n)
    DENSITY = 2.4 + 0.1 * np.sin(z * 0.05) + rng.normal(0, 0.02, n)
    DTC = 80 - 5 * np.sin(z * 0.05) + rng.normal(0, 1.5, n)
    DTS = 1.6 * DTC + rng.normal(0, 2, n)
    X = np.column_stack([DENSITY, DTC, DTS])
    # inject washouts
    n_out = int(contamination * n)
    out_idx = rng.choice(n, size=n_out, replace=False)
    X[out_idx, 0] = rng.uniform(1.6, 2.0, n_out)   # too low density
    X[out_idx, 1] = rng.uniform(120, 180, n_out)   # huge DTC
    return X, out_idx


def test_all():
    print("=" * 60)
    print("Article 5: Outlier Detection and Log Editing")
    print("=" * 60)
    X, true_outliers = synthetic_well(seed=0, n=400, contamination=0.06)
    flag, scores, nu = detect_outliers(X)
    detected = np.where(flag == 1)[0]
    tp = len(set(detected) & set(true_outliers))
    precision = tp / max(len(detected), 1)
    recall = tp / max(len(true_outliers), 1)
    print(f"  nu chosen by inflection = {nu:.3f}")
    print(f"  Outliers detected: {len(detected)}  true: {len(true_outliers)}")
    print(f"  Precision={precision:.2f}  Recall={recall:.2f}")

    # similarity / MDS
    wells = [synthetic_well(seed=s, n=300)[0] for s in range(8)]
    fps = [well_footprint_grid(w, bins=12) for w in wells]
    sim = well_similarity(fps, metric="jaccard")
    emb = mds_embedding(sim)
    print(f"  Mean inter-well Jaccard = {sim[np.triu_indices_from(sim, k=1)].mean():.3f}")
    print(f"  MDS embedding shape: {emb.shape}")

    # reconstruction with uncertainty
    inliers = flag == 0
    Xtr = X[inliers][:, 1:]   # use DTC+DTS
    ytr = X[inliers][:, 0]    # predict density
    mean, std = reconstruct_with_uncertainty(Xtr, ytr, X[:, 1:], k=5)
    rmse = float(np.sqrt(np.mean((mean - X[:, 0]) ** 2)))
    print(f"  Reconstruction RMSE (density)={rmse:.3f}  mean uncertainty={std.mean():.3f}")

    assert recall > 0.3
    print("  PASS")
    return {"precision": precision, "recall": recall, "rmse": rmse, "nu": nu}


if __name__ == "__main__":
    test_all()
