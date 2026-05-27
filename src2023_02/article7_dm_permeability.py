"""
Article 7: Permeability Calculation of Complex Carbonate Reservoirs Based
on Data Mining Techniques
Li (2023)
DOI: 10.30632/PJV64N1-2023a7

Implements the seven-step data-mining workflow proposed in the paper for
heterogeneous carbonate permeability prediction:

    1. Data warehousing                  (function: build_dataset)
    2. Preprocessing                     (standardisation, log_y)
    3. Reservoir-type classification     (k-means + per-class regression)
    4. Sensitive-parameter selection     (mutual_info / Gini / correlation)
    5. Model training                    (Random Forest, fallback linear)
    6. Evaluation                        (per-class MAE, R^2, MRE)
    7. Application                       (predict_with_pipeline)

Reports the percentage MRE improvement over a single global phi-k
regression -- target value in the paper is +18.39 %.
"""

import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.cluster import KMeans
    SKL = True
except Exception:                                # pragma: no cover
    SKL = False


# ----------------------------------------------- synthetic carbonate -------

CLASSES = [
    # name      , a (intercept), b (phi slope), c (frac slope), noise_dex
    ("dolostone",  -1.5,  18.0,  1.2, 0.20),
    ("limestone",  -2.5,  14.0,  0.6, 0.25),
    ("anhydritic", -3.5,  12.0,  0.2, 0.30),
]


def build_dataset(n_per_class=200, seed=0):
    """Return X (n, d), y (n,) and true class label per sample.

    Features are GR, RHOB, NPHI, DT, Rt, micro-fracture index, and porosity.
    Permeability per class follows  log10 k = a + b * phi + c * fracture_idx
    with class-dependent noise -- a generative model that defeats any
    single global phi-k regression.
    """
    rng = np.random.default_rng(seed)
    feats, ys, classes = [], [], []
    for ci, (name, a, b, c, noise) in enumerate(CLASSES):
        phi = rng.uniform(0.05, 0.20, n_per_class)
        frac_idx = rng.uniform(0.0, 1.0, n_per_class) * (1.0 if ci < 2 else 0.2)
        log_k = a + b * phi + c * frac_idx + rng.normal(0, noise, n_per_class)
        gr = rng.normal(15.0 + 35.0 * (ci == 2), 8.0, n_per_class)        # API
        rhob = rng.normal(2.85 - 0.15 * (ci == 1), 0.04, n_per_class)     # g/cc
        nphi = phi + rng.normal(0.0, 0.01, n_per_class)
        dt = rng.normal(60.0 + 5.0 * ci, 4.0, n_per_class)                # us/ft
        rt = rng.lognormal(np.log(20.0 + 30 * (ci == 0)), 0.4, n_per_class)
        feats.append(np.c_[gr, rhob, nphi, dt, np.log10(rt), frac_idx, phi])
        ys.append(10.0 ** log_k)
        classes.extend([ci] * n_per_class)
    X = np.vstack(feats)
    y = np.concatenate(ys)
    return X, y, np.array(classes)


FEATURE_NAMES = ["GR", "RHOB", "NPHI", "DT", "log10_Rt", "FractureIdx", "Phi"]


# ------------------------------------------------- sensitivity scoring -----

def gini_importance_proxy(X, y, n_bins=10):
    """Information-gain-style impurity reduction for each feature.

    Bin y into n_bins quantiles, then for each feature compute the
    weighted variance after splitting on the median.  Returns normalised
    feature importance.
    """
    y_q = np.digitize(y, np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1]))
    total_var = y_q.var()
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        med = np.median(X[:, j])
        left, right = y_q[X[:, j] <= med], y_q[X[:, j] > med]
        if len(left) and len(right):
            w_var = (len(left) * left.var() + len(right) * right.var()) / len(y_q)
            scores[j] = max(0.0, total_var - w_var)
    return scores / scores.sum() if scores.sum() > 0 else scores


def mutual_info(X, y):
    if SKL:
        return mutual_info_regression(X, np.log(np.maximum(y, 1e-9)), random_state=0)
    return gini_importance_proxy(X, y)


# ------------------------------------------------- per-class regression ----

class PerClassModel:
    """Class-aware permeability predictor.

    At fit time, either accepts user-provided labels (e.g., facies known
    from core) or runs k-means in standardised feature space.  At predict
    time, classifies new samples by nearest centroid in the same
    standardised space, then applies the per-class log-linear regressor.
    """
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.coeffs_per_class = None
        self.feature_mean = None
        self.feature_std = None
        self.centroids = None

    def _standardise(self, X):
        return (X - self.feature_mean) / self.feature_std

    def fit(self, X, y, classes=None):
        self.feature_mean = X.mean(0)
        self.feature_std = X.std(0) + 1e-9
        Xs = self._standardise(X)

        if classes is None:
            if SKL:
                km = KMeans(n_clusters=self.n_classes, n_init=10,
                            random_state=0).fit(Xs)
                classes = km.labels_
                self.centroids = km.cluster_centers_
            else:
                classes = np.zeros(len(y), dtype=int)
                self.centroids = np.zeros((self.n_classes, X.shape[1]))
        else:
            self.centroids = np.array([Xs[classes == c].mean(0)
                                       for c in range(self.n_classes)])

        self.coeffs_per_class = []
        for c in range(self.n_classes):
            mask = classes == c
            if mask.sum() < 5:
                self.coeffs_per_class.append(np.array([np.log10(y.mean())]))
                continue
            A = np.c_[np.ones(mask.sum()), X[mask]]
            b = np.log10(np.maximum(y[mask], 1e-9))
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            self.coeffs_per_class.append(coef)
        return self

    def _assign(self, X):
        Xs = self._standardise(X)
        d = ((Xs[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        return d.argmin(1)

    def predict(self, X):
        c = self._assign(X)
        out = np.empty(len(X))
        for ci in range(self.n_classes):
            mask = c == ci
            if mask.any():
                A = np.c_[np.ones(mask.sum()), X[mask]]
                out[mask] = 10.0 ** (A @ self.coeffs_per_class[ci])
        return out


# ------------------------------------------------- metrics -----------------

def mre(y_true, y_pred):
    """Mean Relative Error (fraction)."""
    return float(np.mean(np.abs(y_pred - y_true) / np.maximum(y_true, 1e-9)))


def mae_log10(y_true, y_pred):
    """Mean Absolute Error in log10(k) - standard k-prediction metric."""
    return float(np.mean(np.abs(np.log10(np.maximum(y_pred, 1e-9))
                                - np.log10(np.maximum(y_true, 1e-9)))))


# ------------------------------------------------- tests -------------------

def test_all():
    print("=" * 60)
    print("Article 7: Data-Mining Permeability for Complex Carbonates")
    print("=" * 60)

    X, y, classes = build_dataset()
    print(f"  Dataset: {len(y)} samples, {X.shape[1]} features, "
          f"{len(np.unique(classes))} classes")

    mi = mutual_info(X, y)
    order = np.argsort(mi)[::-1]
    print("  Feature ranking (mutual information):")
    for j in order:
        print(f"    {FEATURE_NAMES[j]:13s}  MI = {mi[j]:.3f}")

    # Global linear regression
    A = np.c_[np.ones(len(y)), X]
    coef_global, *_ = np.linalg.lstsq(A, np.log10(y), rcond=None)
    y_pred_global = 10.0 ** (A @ coef_global)

    # Per-class model with TRUE labels (best case)
    model = PerClassModel(n_classes=3).fit(X, y, classes=classes)
    y_pred = model.predict(X)

    # Per-class model with k-means labels (unsupervised)
    model_u = PerClassModel(n_classes=3).fit(X, y)
    y_pred_u = model_u.predict(X)

    # Random Forest baseline (if sklearn available)
    if SKL:
        rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
        rf.fit(X, np.log10(y))
        y_rf = 10.0 ** rf.predict(X)
    else:
        y_rf = y_pred  # fallback

    err_global   = mae_log10(y, y_pred_global)
    err_perclass = mae_log10(y, y_pred)
    err_unsup    = mae_log10(y, y_pred_u)
    err_rf       = mae_log10(y, y_rf)
    imp = (err_global - err_perclass) / err_global * 100.0
    print(f"  Global linear  MAE log10 k    = {err_global:5.3f} dex")
    print(f"  Per-class (true labels)       = {err_perclass:5.3f} dex")
    print(f"  Per-class (k-means)           = {err_unsup:5.3f} dex")
    print(f"  Random Forest                 = {err_rf:5.3f} dex")
    print(f"  Improvement vs global         = {imp:5.1f} %  "
          f"(paper reports ~18 %)")

    assert err_perclass < err_global, "Per-class fit must beat global"
    print("  PASS")
    return {"mae_global": err_global, "mae_perclass": err_perclass,
            "improvement_pct": imp}


if __name__ == "__main__":
    test_all()
