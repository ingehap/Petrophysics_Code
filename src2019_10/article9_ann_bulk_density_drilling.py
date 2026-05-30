"""
Article 9: Application of Artificial Neural Network to Predict Formation Bulk
           Density While Drilling
Gowida, Elkatatny, Abdulraheem (2019)
DOI: 10.30632/PJV60N5-2019a9

A feed-forward neural network predicts formation bulk density in real time from
surface drilling parameters, providing a density estimate where no density log
is available while drilling.

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
    return float(np.corrcoef(np.asarray(y, float), np.asarray(yhat, float))[0, 1])


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
    print("Article 9: ANN Bulk Density While Drilling")
    print("=" * 60)

    rng = np.random.default_rng(9)
    n = 700
    X = np.column_stack([rng.uniform(5, 40, n), rng.uniform(60, 180, n),
                         rng.uniform(5, 60, n), rng.uniform(2, 12, n),
                         rng.uniform(1500, 3500, n), rng.uniform(300, 700, n)])
    Xs = (X - X.mean(0)) / X.std(0)
    # bulk density ~ 2.2-2.7 g/cc, smooth response with a mild nonlinearity
    rhob = 2.45 + 0.10 * Xs[:, 0] - 0.07 * Xs[:, 2] + 0.04 * np.tanh(Xs[:, 4]) \
        + rng.normal(0, 0.02, n)

    ntr = 490
    p = train_ann(X[:ntr], rhob[:ntr], hidden=20, iters=5000, lr=0.05, seed=1)
    pred = predict_ann(p, X[ntr:])
    R = correlation_coefficient(rhob[ntr:], pred)
    err = rmse(rhob[ntr:], pred)
    ae = aape(rhob[ntr:], pred)
    print(f"  test R / RMSE / AAPE   = {R:.3f} / {err:.3f} / {ae:.3f}%")
    assert R > 0.9 and ae < 2.0

    # Metric sanity
    assert abs(correlation_coefficient([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9
    assert abs(aape([2.0, 2.0], [2.0, 2.0])) < 1e-12
    print("  PASS")
    return {"R": R, "RMSE": err, "AAPE": ae}


if __name__ == "__main__":
    test_all()
