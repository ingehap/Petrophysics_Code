"""
Article 7: New Robust Model to Estimate Formation Tops in Real Time Using
           Artificial Neural Networks (ANN)
Elkatatny, Al-AbdulJabbar, Mahmoud (2019)
DOI: 10.30632/PJV60N6-2019a7

A feed-forward neural network estimates formation tops (a depth marker) in real
time from surface drilling parameters, so boundaries can be picked while
drilling without waiting for wireline logs.

Implements:

  - Min-max feature normalization
  - Single-hidden-layer tanh ANN regressor
  - Correlation coefficient R, RMSE, and average absolute percent error (AAPE)

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard ANN-from-drilling-parameters regression the
paper's title describes.  Inputs: WOB, RPM, ROP, torque, SPP, flow rate.
"""

import numpy as np

DRILLING_INPUTS = ["WOB", "RPM", "ROP", "Torque", "SPP", "GPM"]


# ---------------------------------------------- metrics -----------------

def correlation_coefficient(y, yhat):
    """Pearson correlation coefficient R."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.corrcoef(y, yhat)[0, 1])


def rmse(y, yhat):
    """Root-mean-square error."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def aape(y, yhat):
    """Average absolute percentage error (%)."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.mean(np.abs((y - yhat) / y)) * 100.0)


# ---------------------------------------------- ANN ---------------------

def train_ann(X, y, hidden=20, iters=5000, lr=0.05, seed=0):
    """Single-hidden-layer tanh ANN regressor (min-max inputs, standardized y)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float); y = np.asarray(y, float)
    xmin, xmax = X.min(0), X.max(0)
    Xn = (X - xmin) / (xmax - xmin + 1e-12)
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
    return dict(W1=W1, b1=b1, w2=w2, b2=b2, xmin=xmin, xmax=xmax, ym=ym, ys=ys)


def predict_ann(p, X):
    Xn = (np.asarray(X, float) - p["xmin"]) / (p["xmax"] - p["xmin"] + 1e-12)
    H = np.tanh(Xn @ p["W1"] + p["b1"])
    return (H @ p["w2"] + p["b2"]) * p["ys"] + p["ym"]


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: ANN Formation Tops From Drilling Parameters")
    print("=" * 60)

    rng = np.random.default_rng(2)
    n = 700
    X = np.column_stack([rng.uniform(5, 40, n), rng.uniform(60, 180, n),
                         rng.uniform(5, 60, n), rng.uniform(2, 12, n),
                         rng.uniform(1500, 3500, n), rng.uniform(300, 700, n)])
    Xs = (X - X.mean(0)) / X.std(0)
    # formation-top depth marker (a smooth function of drilling response) + noise
    top = 8000.0 + 120 * Xs[:, 0] - 80 * Xs[:, 2] + 50 * np.tanh(Xs[:, 4]) \
        + rng.normal(0, 10, n)

    ntr = 490
    p = train_ann(X[:ntr], top[:ntr], hidden=20, iters=5000, lr=0.05, seed=1)
    pred = predict_ann(p, X[ntr:])
    R = correlation_coefficient(top[ntr:], pred)
    err = rmse(top[ntr:], pred)
    ae = aape(top[ntr:], pred)
    print(f"  test R / RMSE / AAPE   = {R:.3f} / {err:.1f} / {ae:.3f}%")
    assert R > 0.9 and ae < 1.0

    # Metric sanity
    assert abs(correlation_coefficient([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    assert abs(rmse([1, 2, 3], [1, 2, 3])) < 1e-12
    print("  PASS")
    return {"R": R, "RMSE": err, "AAPE": ae}


if __name__ == "__main__":
    test_all()
