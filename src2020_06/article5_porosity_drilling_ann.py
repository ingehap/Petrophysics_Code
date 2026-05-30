"""
Article 5: Estimation of Reservoir Porosity From Drilling Parameters Using
           Artificial Neural Networks
Al-AbdulJabbar, Al-Azani, Elkatatny (2020)
DOI: 10.30632/PJV61N3-2020a5

A feed-forward neural network predicts reservoir porosity from six surface
drilling parameters - ROP, WOB, RPM, torque, pumping rate, and standpipe
pressure - trained on one well and validated on an unseen well.  The paper
reports no numbered equations; the correlation coefficient R and the RMSE are
the reported performance metrics.

Implements:

  - Min-max feature normalization
  - Feed-forward ANN (tan-sigmoid hidden layer) porosity regressor
  - Correlation coefficient R and RMSE

Note: this issue's PDF has a text layer but the paper publishes no numbered
equations; R and RMSE are the standard metric definitions.  The published
network used two hidden layers of 30 neurons (Levenberg-Marquardt, tan-sigmoid)
and reported R = 0.96/0.94 (train/test) and RMSE ~ 0.018-0.035; this compact
numpy net reaches comparable accuracy on synthetic data.
"""

import numpy as np

DRILLING_INPUTS = ["ROP", "WOB", "RPM", "Torque", "GPM", "SPP"]


# ---------------------------------------------- metrics -----------------

def correlation_coefficient(y, yhat):
    """Pearson correlation coefficient R between actual and predicted."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.corrcoef(y, yhat)[0, 1])


def rmse(y, yhat):
    """Root-mean-square error  sqrt(mean((y-yhat)^2))."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


# ---------------------------------------------- ANN ---------------------

def train_ann(X, y, hidden=24, iters=5000, lr=0.05, seed=0):
    """Single-hidden-layer tanh ANN porosity regressor (min-max inputs)."""
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
        pred = H @ w2 + b2
        err = pred - yn
        gw2 = H.T @ err / N; gb2 = err.mean()
        dH = (np.outer(err, w2) * (1 - H ** 2)) / N
        gW1 = Xn.T @ dH; gb1 = dH.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1
        w2 -= lr * gw2; b2 -= lr * gb2
    return dict(W1=W1, b1=b1, w2=w2, b2=b2, xmin=xmin, xmax=xmax, ym=ym, ys=ys)


def predict_ann(p, X):
    """Predict porosity for inputs X."""
    Xn = (np.asarray(X, float) - p["xmin"]) / (p["xmax"] - p["xmin"] + 1e-12)
    H = np.tanh(Xn @ p["W1"] + p["b1"])
    return (H @ p["w2"] + p["b2"]) * p["ys"] + p["ym"]


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Porosity From Drilling Parameters (ANN)")
    print("=" * 60)

    # Synthetic field set: porosity from six drilling parameters + mild noise
    rng = np.random.default_rng(3)
    n = 800
    ROP = rng.uniform(5, 60, n); WOB = rng.uniform(5, 40, n)
    RPM = rng.uniform(60, 180, n); TOR = rng.uniform(2, 12, n)
    GPM = rng.uniform(300, 700, n); SPP = rng.uniform(1500, 3500, n)
    X = np.column_stack([ROP, WOB, RPM, TOR, GPM, SPP])
    Xs = (X - X.mean(0)) / X.std(0)
    # porosity ~ 0.02-0.30, smooth response with a mild nonlinearity
    phi = (0.16 + 0.04 * Xs[:, 0] - 0.03 * Xs[:, 1] + 0.02 * np.tanh(Xs[:, 5])
           + 0.02 * Xs[:, 2] + rng.normal(0, 0.01, n))

    ntr = int(0.7 * n)
    p = train_ann(X[:ntr], phi[:ntr], hidden=24, iters=5000, lr=0.05, seed=1)

    pred_tr = predict_ann(p, X[:ntr])
    pred_te = predict_ann(p, X[ntr:])
    R_tr, R_te = correlation_coefficient(phi[:ntr], pred_tr), correlation_coefficient(phi[ntr:], pred_te)
    rmse_tr, rmse_te = rmse(phi[:ntr], pred_tr), rmse(phi[ntr:], pred_te)
    print(f"  train R / RMSE         = {R_tr:.3f} / {rmse_tr:.3f}")
    print(f"  test  R / RMSE         = {R_te:.3f} / {rmse_te:.3f}")
    assert R_tr > 0.9 and R_te > 0.9
    assert rmse_te < 0.04                          # comparable to the paper

    # Validation on an unseen distribution (within training input ranges)
    m = 200
    Xv = np.column_stack([rng.uniform(5, 60, m), rng.uniform(5, 40, m),
                          rng.uniform(60, 180, m), rng.uniform(2, 12, m),
                          rng.uniform(300, 700, m), rng.uniform(1500, 3500, m)])
    pred_v = predict_ann(p, Xv)
    assert np.all(np.isfinite(pred_v))
    print(f"  validation porosity range = {pred_v.min():.2f}..{pred_v.max():.2f}")
    print("  PASS")
    return {"R_train": R_tr, "R_test": R_te, "RMSE_test": rmse_te}


if __name__ == "__main__":
    test_all()
