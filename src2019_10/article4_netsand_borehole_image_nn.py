"""
Article 4: Estimating Net Sand From Borehole Images in Laminated Deepwater
           Reservoirs With a Neural Network
Gong, Keele, Toumelin, Clinch (2019)
DOI: 10.30632/PJV60N5-2019a4

Counting net sand from oil-based-mud (OBM) borehole images with a fixed
resistivity/brightness cutoff fails because the apparent-to-true response is
nonlinear.  A neural network instead regresses the sand fraction directly from
the image, learning features rather than relying on a hand-picked threshold.
The paper publishes no numbered equations; this module implements that
image-to-sand-fraction regression and contrasts it with a fixed cutoff.

Implements:

  - Image brightness-histogram features
  - Fixed-cutoff net-sand estimate (the baseline that fails under OBM)
  - Neural-network sand-fraction regressor (MSE loss)
  - RMSE / correlation evaluation

Note: this issue's PDF has a text layer but the paper describes a deep-learning
regression with no numbered equations; this is a faithful numpy implementation
of the same image-to-sand-fraction method, with the OBM nonlinearity that
defeats a fixed cutoff.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- features ----------------

def histogram_features(image, bins=8):
    """Normalized brightness histogram of a (distorted) borehole-image crop."""
    h, _ = np.histogram(np.asarray(image, float), bins=bins, range=(0.0, 1.0))
    return h / h.sum()


def cutoff_netsand(image, threshold=0.5):
    """Fixed-cutoff net-sand fraction = fraction of pixels above the threshold."""
    return float(np.mean(np.asarray(image, float) >= threshold))


# ---------------------------------------------- NN regressor ------------

def train_nn(X, y, hidden=16, iters=4000, lr=0.1, seed=0):
    """Single-hidden-layer tanh NN regressor for the sand fraction."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float); y = np.asarray(y, float)
    ym, ys = y.mean(), y.std() + 1e-12
    yn = (y - ym) / ys
    d = X.shape[1]
    W1 = rng.normal(0, 0.5, (d, hidden)); b1 = np.zeros(hidden)
    w2 = rng.normal(0, 0.3, hidden); b2 = 0.0
    N = len(y)
    for _ in range(iters):
        H = np.tanh(X @ W1 + b1)
        err = (H @ w2 + b2) - yn
        gw2 = H.T @ err / N; gb2 = err.mean()
        dH = (np.outer(err, w2) * (1 - H ** 2)) / N
        W1 -= lr * (X.T @ dH); b1 -= lr * dH.sum(0)
        w2 -= lr * gw2; b2 -= lr * gb2
    return dict(W1=W1, b1=b1, w2=w2, b2=b2, ym=ym, ys=ys)


def predict_nn(p, X):
    H = np.tanh(np.asarray(X, float) @ p["W1"] + p["b1"])
    return np.clip((H @ p["w2"] + p["b2"]) * p["ys"] + p["ym"], 0.0, 1.0)


def rmse(y, yhat):
    return petrolib.ml_stats.rmse(y, yhat)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Net Sand From Borehole Images (NN)")
    print("=" * 60)

    rng = np.random.default_rng(4)
    n_img, n_pix = 400, 200

    def make_image(f):
        n_sand = int(round(f * n_pix))
        sand = rng.normal(0.7, 0.12, n_sand)
        shale = rng.normal(0.3, 0.12, n_pix - n_sand)
        b = np.clip(np.concatenate([sand, shale]), 0.0, 1.0)
        return b ** 2                              # OBM nonlinear distortion

    fracs = rng.uniform(0.0, 1.0, n_img)
    images = [make_image(f) for f in fracs]
    X = np.array([histogram_features(im) for im in images])

    ntr = 280
    p = train_nn(X[:ntr], fracs[:ntr], hidden=16, iters=4000, lr=0.1, seed=1)
    nn_pred = predict_nn(p, X[ntr:])
    cutoff_pred = np.array([cutoff_netsand(im, 0.5) for im in images[ntr:]])

    rmse_nn = rmse(fracs[ntr:], nn_pred)
    rmse_cut = rmse(fracs[ntr:], cutoff_pred)
    print(f"  net-sand RMSE NN / cutoff = {rmse_nn:.3f} / {rmse_cut:.3f}")
    # the NN learns the true sand fraction; the fixed cutoff is badly biased by
    # the OBM nonlinearity
    assert rmse_nn < rmse_cut
    assert rmse_nn < 0.1
    R = float(np.corrcoef(fracs[ntr:], nn_pred)[0, 1])
    print(f"  NN correlation R       = {R:.3f}")
    assert R > 0.9
    print("  PASS")
    return {"rmse_nn": rmse_nn, "rmse_cutoff": rmse_cut, "R": R}


if __name__ == "__main__":
    test_all()
