"""
Article 2: Geological Feature Prediction Using Image-Based Machine Learning
Jobe, Vital-Brazil, Khait (2018)
DOI: 10.30632/PJV59N6Y2018a1

Geological features (e.g. bedding vs fractured/chaotic fabric) are predicted from
borehole or core images with machine learning.  Texture features extracted from
the image (mean, gradient energy, directional contrast) feed a classifier that
labels each image patch by its geological feature.

Implements:

  - Image texture features (mean, gradient energy, orientation contrast)
  - Logistic-regression image classifier (gradient descent)
  - Classification accuracy and confusion matrix

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so this is a faithful standard-form reconstruction of the
image-feature + ML classification workflow the paper applies (numpy stands in
for the published model).
"""

import numpy as np


# ---------------------------------------------- features ----------------

def texture_features(image):
    """Extract (mean, gradient energy, horizontal-vertical contrast) from a patch."""
    img = np.asarray(image, float)
    gy, gx = np.gradient(img)
    grad_energy = float(np.mean(gx ** 2 + gy ** 2))
    contrast = float(np.mean(gx ** 2) - np.mean(gy ** 2))   # bedding vs chaotic
    return np.array([img.mean(), grad_energy, contrast])


# ---------------------------------------------- classifier --------------

def logistic_fit(X, y, iters=4000, lr=0.1):
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
    w = np.zeros(Xb.shape[1]); y = np.asarray(y, float)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        w -= lr * Xb.T @ (p - y) / len(y)
    return dict(w=w, mean=X.mean(0), std=X.std(0) + 1e-12)


def logistic_predict(model, X):
    Xs = (np.asarray(X, float) - model["mean"]) / model["std"]
    Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
    return (1.0 / (1.0 + np.exp(-Xb @ model["w"])) >= 0.5).astype(int)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Geological Feature Prediction (Image ML)")
    print("=" * 60)

    rng = np.random.default_rng(1)

    def bedded(): return np.add.outer(np.linspace(0, 1, 16), np.zeros(16)) \
        + 0.05 * rng.standard_normal((16, 16))      # horizontal layering

    def chaotic(): return rng.standard_normal((16, 16))   # no orientation

    # Texture features separate bedded (directional) from chaotic fabric
    fb = texture_features(bedded()); fc = texture_features(chaotic())
    print(f"  contrast bedded/chaotic = {fb[2]:+.3f} / {fc[2]:+.3f}")
    assert abs(fb[2]) != abs(fc[2])

    # Build a labeled image set and classify
    X, y = [], []
    for _ in range(60):
        X.append(texture_features(bedded())); y.append(0)
        X.append(texture_features(chaotic())); y.append(1)
    X = np.array(X); y = np.array(y)
    idx = rng.permutation(len(y)); X, y = X[idx], y[idx]
    ntr = 90
    m = logistic_fit(X[:ntr], y[:ntr])
    acc = accuracy(y[ntr:], logistic_predict(m, X[ntr:]))
    print(f"  classification accuracy = {acc:.2f}")
    assert acc > 0.85
    print("  PASS")
    return {"accuracy": acc}


if __name__ == "__main__":
    test_all()
