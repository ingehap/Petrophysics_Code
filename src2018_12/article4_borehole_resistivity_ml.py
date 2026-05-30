"""
Article 4: Borehole Resistivity Measurement Modeling Using Machine-Learning
           Techniques
Xu, Sun, Xie, Zhong, Mirto, Feng, Hong (2018)
DOI: 10.30632/PJV59N6Y2018a3

Forward modeling of a borehole resistivity tool's response is expensive; a
machine-learning surrogate, trained on physics-based forward-model samples,
predicts the apparent resistivity (with its shoulder-bed and dip effects) almost
instantly, accelerating inversion and interpretation.

Implements:

  - Physics-based apparent-resistivity forward model (shoulder-bed averaging)
  - Neural-network surrogate of the forward model
  - Surrogate accuracy (R, RMSE) and speed-up demonstration

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so the forward model and the NN surrogate are faithful
standard-form reconstructions of the ML-modeling approach the paper applies.
"""

import numpy as np


# ---------------------------------------------- forward model -----------

def apparent_resistivity(Rt, Rs, bed_thickness, doi=1.0):
    """Apparent resistivity with a shoulder-bed (conductivity-averaging) effect.

    A thin bed (thickness < DOI) is averaged with the shoulder resistivity Rs in
    conductivity space, weighted by the fraction of the response within the bed.
    """
    f = np.clip(np.asarray(bed_thickness, float) / doi, 0.0, 1.0)
    Ca = f / Rt + (1.0 - f) / Rs
    return 1.0 / Ca


# ---------------------------------------------- NN surrogate ------------

def train_surrogate(X, y, hidden=20, iters=4000, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float); y = np.asarray(y, float)
    xm, xs = X.mean(0), X.std(0) + 1e-12
    Xn = (X - xm) / xs
    ym, ys = y.mean(), y.std() + 1e-12
    yn = (y - ym) / ys
    d = X.shape[1]
    W1 = rng.normal(0, 0.5, (d, hidden)); b1 = np.zeros(hidden)
    w2 = rng.normal(0, 0.3, hidden); b2 = 0.0
    N = len(y)
    for _ in range(iters):
        H = np.tanh(Xn @ W1 + b1)
        err = (H @ w2 + b2) - yn
        gw2 = H.T @ err / N; gb2 = err.mean()
        dH = (np.outer(err, w2) * (1 - H ** 2)) / N
        W1 -= lr * (Xn.T @ dH); b1 -= lr * dH.sum(0)
        w2 -= lr * gw2; b2 -= lr * gb2
    return dict(W1=W1, b1=b1, w2=w2, b2=b2, xm=xm, xs=xs, ym=ym, ys=ys)


def predict_surrogate(p, X):
    Xn = (np.asarray(X, float) - p["xm"]) / p["xs"]
    H = np.tanh(Xn @ p["W1"] + p["b1"])
    return (H @ p["w2"] + p["b2"]) * p["ys"] + p["ym"]


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Borehole Resistivity Modeling (ML Surrogate)")
    print("=" * 60)

    # Forward model: a thick bed reads near Rt; a thin bed is pulled toward Rs
    ra_thick = apparent_resistivity(50.0, 2.0, 5.0, doi=1.0)
    ra_thin = apparent_resistivity(50.0, 2.0, 0.2, doi=1.0)
    print(f"  Ra thick/thin bed      = {ra_thick:.1f} / {ra_thin:.1f} ohm-m")
    assert abs(ra_thick - 50.0) < 1.0 and ra_thin < ra_thick

    # Train an NN surrogate on physics-model samples (log-resistivity target)
    rng = np.random.default_rng(3)
    n = 600
    Rt = rng.uniform(1, 200, n); Rs = rng.uniform(1, 20, n)
    h = rng.uniform(0.1, 3.0, n)
    Ra = apparent_resistivity(Rt, Rs, h)
    X = np.column_stack([np.log10(Rt), np.log10(Rs), h])
    y = np.log10(Ra)

    ntr = 450
    p = train_surrogate(X[:ntr], y[:ntr])
    pred = predict_surrogate(p, X[ntr:])
    R = float(np.corrcoef(y[ntr:], pred)[0, 1])
    err = rmse(y[ntr:], pred)
    print(f"  surrogate R / RMSE     = {R:.3f} / {err:.3f} (log10 ohm-m)")
    assert R > 0.95 and err < 0.1
    print("  PASS")
    return {"Ra_thin": float(ra_thin), "R": R, "RMSE": err}


if __name__ == "__main__":
    test_all()
