"""
Article 2: Data-Driven Algorithms for Image-Based Rock Classification and
Formation Evaluation in Formations With Rapid Spatial Variation in Rock Fabric
Gonzalez, Heidari, and Lopez (2023)
DOI: 10.30632/PJV64N2-2023a2

Implements:
  - Feature extraction from CT-scan-like grayscale images:
      * mean, variance, skewness, kurtosis (Eqs. 1-4)
      * GLCM-based contrast, energy, correlation (Eqs. 6-8)
      * HSV mean from RGB core photos (Eq. 5)
  - Supervised rock classification (Random Forest + SVM)
  - Unsupervised rock classification (k-means)
  - Class-based permeability-porosity model
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops


# --------------------------------------------------------------- features ---

def grayscale_stats(image_row):
    """Compute mean, variance, skewness, kurtosis (Eqs. 1-4) over a row."""
    x = image_row.astype(float)
    n = len(x)
    mu = np.mean(x)
    var = np.var(x)
    sd = np.sqrt(var) + 1e-12
    skew = np.mean(((x - mu) / sd) ** 3)
    kurt = np.mean(((x - mu) / sd) ** 4) - 3.0
    return mu, var, skew, kurt


def hsv_mean(rgb_row):
    """Mean H, S, V per depth (Eq. 5)."""
    from skimage.color import rgb2hsv
    img = rgb_row.reshape(1, -1, 3) / 255.0
    hsv = rgb2hsv(img)[0]
    return hsv.mean(axis=0)


def glcm_features(patch, distances=(1,), angles=(0,), levels=16):
    """Contrast, energy, correlation from GLCM (Eqs. 6-8)."""
    p = (patch.astype(float) / patch.max() * (levels - 1)).astype(np.uint8) if patch.max() > 0 else patch.astype(np.uint8)
    glcm = graycomatrix(p, distances=distances, angles=angles, levels=levels,
                        symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    return contrast, energy, correlation


def extract_features(ct_image, photo_image=None, window=5):
    """
    Slide a window down the depth axis (axis=0) and compute features per depth row.
    Returns matrix (T, n_features).
    """
    T = ct_image.shape[0]
    feats = []
    pad = window // 2
    for t in range(T):
        row = ct_image[t]
        gs = grayscale_stats(row)
        lo = max(0, t - pad)
        hi = min(T, t + pad + 1)
        patch = ct_image[lo:hi]
        if patch.shape[0] >= 2 and patch.shape[1] >= 2:
            gl = glcm_features(patch)
        else:
            gl = (0.0, 0.0, 0.0)
        f = list(gs) + list(gl)
        if photo_image is not None:
            f.extend(hsv_mean(photo_image[t]))
        feats.append(f)
    return np.array(feats)


# ---------------------------------------------------------- classification ---

def train_random_forest(X_train, y_train, n_estimators=100, cv=5):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    scores = cross_val_score(rf, X_train, y_train, cv=cv)
    rf.fit(X_train, y_train)
    return rf, scores.mean()


def train_svm(X_train, y_train, cv=5):
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0)
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    scores = cross_val_score(svm, Xs, y_train, cv=cv)
    svm.fit(Xs, y_train)
    return svm, scaler, scores.mean()


def kmeans_unsupervised(X, n_clusters=3, random_state=0):
    """Unsupervised baseline."""
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(scaler.transform(X))
    return km, scaler


def class_based_perm_porosity(porosity, permeability, classes):
    """Per-class log(k) = a + b * phi linear fit."""
    models = {}
    for c in np.unique(classes):
        mask = classes == c
        if mask.sum() < 3:
            continue
        phi = porosity[mask]
        k = np.log10(permeability[mask] + 1e-6)
        A = np.vstack([np.ones_like(phi), phi]).T
        coef, *_ = np.linalg.lstsq(A, k, rcond=None)
        models[int(c)] = coef
    return models


def predict_perm(porosity, classes, models):
    """Predict permeability from porosity using per-class model."""
    pred = np.zeros_like(porosity)
    for c, coef in models.items():
        mask = classes == c
        pred[mask] = 10 ** (coef[0] + coef[1] * porosity[mask])
    return pred


# --------------------------------------------------------------- testing ---

def synthetic_data(seed=0):
    """Synthetic CT-like images for 3 lithofacies."""
    rng = np.random.default_rng(seed)
    H, W = 300, 64       # depth x width
    image = np.zeros((H, W), dtype=np.uint8)
    labels = np.zeros(H, dtype=int)
    porosity = np.zeros(H)
    permeability = np.zeros(H)
    for t in range(H):
        c = t // 100
        labels[t] = c
        if c == 0:        # bright + smooth (carbonate)
            image[t] = rng.normal(200, 8, W).clip(0, 255).astype(np.uint8)
            porosity[t] = rng.uniform(0.05, 0.10)
            permeability[t] = 10 ** (rng.normal(0.5 + 5 * porosity[t], 0.2))
        elif c == 1:      # mid-tone + textured (sandstone)
            base = rng.normal(140, 25, W)
            image[t] = base.clip(0, 255).astype(np.uint8)
            porosity[t] = rng.uniform(0.15, 0.25)
            permeability[t] = 10 ** (rng.normal(1.0 + 8 * porosity[t], 0.3))
        else:             # dark + heterogeneous (shale)
            image[t] = rng.normal(70, 35, W).clip(0, 255).astype(np.uint8)
            porosity[t] = rng.uniform(0.02, 0.08)
            permeability[t] = 10 ** (rng.normal(-1.0 + 3 * porosity[t], 0.4))
    return image, labels, porosity, permeability


def test_all():
    print("=" * 60)
    print("Article 2: Image-Based Rock Classification")
    print("=" * 60)
    img, y_true, phi, k = synthetic_data()
    X = extract_features(img)
    print(f"  Feature matrix: {X.shape}")

    # 60/40 split
    n = X.shape[0]
    perm = np.random.default_rng(0).permutation(n)
    split = int(n * 0.6)
    tr, te = perm[:split], perm[split:]

    rf, rf_cv = train_random_forest(X[tr], y_true[tr])
    rf_acc = accuracy_score(y_true[te], rf.predict(X[te]))

    svm, scaler, svm_cv = train_svm(X[tr], y_true[tr])
    svm_acc = accuracy_score(y_true[te], svm.predict(scaler.transform(X[te])))

    km, kscaler = kmeans_unsupervised(X, n_clusters=3)
    km_labels = km.predict(kscaler.transform(X))
    # k-means labels are arbitrary -> compute best alignment
    from itertools import permutations
    def best_acc(yt, yp, k):
        return max(accuracy_score(yt, np.array([p[c] for c in yp]))
                   for p in permutations(range(k)))
    km_acc = best_acc(y_true, km_labels, 3)

    perm_models = class_based_perm_porosity(phi[tr], k[tr], y_true[tr])
    k_pred = predict_perm(phi[te], y_true[te], perm_models)
    rel_err = np.mean(np.abs(k_pred - k[te]) / (k[te] + 1e-6))

    print(f"  RF  cv-acc={rf_cv:.3f}  test={rf_acc:.3f}")
    print(f"  SVM cv-acc={svm_cv:.3f}  test={svm_acc:.3f}")
    print(f"  KMeans best-aligned acc = {km_acc:.3f}")
    print(f"  Class-based perm relative error = {rel_err:.3f}")
    assert rf_acc > 0.6
    print("  PASS")
    return {"rf_acc": rf_acc, "svm_acc": svm_acc, "km_acc": km_acc, "perm_err": rel_err}


if __name__ == "__main__":
    test_all()
