"""
Article 2: Total Organic Carbon Characterization Using Neural-Network Analysis
           of XRF Data
Lawal, Mahmoud, Alade, Abdulraheem (2019)
DOI: 10.30632/PJV60N4-2019a2

Total organic carbon (TOC) is predicted from X-ray-fluorescence (XRF) elemental
concentrations with a neural network.  Redox-sensitive and detrital elements
(e.g. Mo, S, Ni, V vs Si, Al, Ca) carry a nonlinear signature of organic
enrichment that a network learns better than a single-element proxy; the
classic Schmoker density and Passey delta-log-R methods are the baselines.

Implements:

  - Neural-network TOC regression from XRF elements (R, RMSE)
  - Schmoker TOC from bulk density (baseline)
  - Passey delta-log-R TOC (baseline)

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the issue's table of contents and these are
faithful standard-form reconstructions of the NN-from-XRF method and the
classic TOC baselines the paper compares against.
"""

import numpy as np

XRF_ELEMENTS = ["Si", "Al", "Ca", "Fe", "S", "Mo", "Ni", "V"]


# ---------------------------------------------- baselines ---------------

def schmoker_toc(rho_b, rho_min=2.69, rho_org=1.10, factor=1.0):
    """Schmoker (1979) TOC from bulk density (wt%).

        TOC = factor * (rho_min - rho_b)/(rho_min - rho_org) * 100 * organic_C_frac
    Simplified: organic carbon content rises as bulk density falls.
    """
    return np.clip((rho_min - np.asarray(rho_b, float)) / (rho_min - rho_org)
                   * 100.0 * 0.8, 0.0, None)


def passey_dlogr(resistivity, dt_sonic, r_baseline, dt_baseline, lom=10.0):
    """Passey delta-log-R TOC (wt%).

        dlogR = log10(R/R_base) + 0.02*(dt - dt_base)
        TOC = dlogR * 10^(2.297 - 0.1688*LOM)
    """
    dlogr = np.log10(np.asarray(resistivity, float) / r_baseline) \
        + 0.02 * (np.asarray(dt_sonic, float) - dt_baseline)
    return np.clip(dlogr * 10 ** (2.297 - 0.1688 * lom), 0.0, None)


# ---------------------------------------------- NN ----------------------

def train_nn(X, y, hidden=16, iters=5000, lr=0.05, seed=0):
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


def predict_nn(p, X):
    Xn = (np.asarray(X, float) - p["xm"]) / p["xs"]
    H = np.tanh(Xn @ p["W1"] + p["b1"])
    return np.clip((H @ p["w2"] + p["b2"]) * p["ys"] + p["ym"], 0.0, None)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: TOC From XRF via Neural Network")
    print("=" * 60)

    # Schmoker: lower bulk density -> higher TOC
    assert schmoker_toc(2.20) > schmoker_toc(2.60)
    print(f"  Schmoker TOC @2.3 g/cc = {schmoker_toc(2.30):.2f} wt%")

    # Passey: higher resistivity / sonic separation -> higher TOC
    assert passey_dlogr(50.0, 90.0, 5.0, 70.0) > passey_dlogr(8.0, 75.0, 5.0, 70.0)

    # NN TOC from XRF: redox elements (Mo, S, V) drive TOC nonlinearly
    rng = np.random.default_rng(2)
    n = 600
    Si = rng.uniform(10, 40, n); Al = rng.uniform(2, 12, n)
    Ca = rng.uniform(1, 30, n); Fe = rng.uniform(1, 6, n)
    S = rng.uniform(0.1, 5, n); Mo = rng.uniform(1, 80, n)
    Ni = rng.uniform(10, 120, n); V = rng.uniform(50, 600, n)
    X = np.column_stack([Si, Al, Ca, Fe, S, Mo, Ni, V])
    Xs = (X - X.mean(0)) / X.std(0)
    # TOC driven by redox proxies + mild nonlinearity, anticorrelated with Ca
    toc = np.clip(4.0 + 2.0 * np.tanh(Xs[:, 5]) + 1.5 * Xs[:, 4]
                  + 1.0 * Xs[:, 7] - 1.0 * Xs[:, 2] + rng.normal(0, 0.3, n), 0.2, None)

    ntr = 420
    p = train_nn(X[:ntr], toc[:ntr], hidden=16, iters=5000, lr=0.05, seed=1)
    pred = predict_nn(p, X[ntr:])
    R = float(np.corrcoef(toc[ntr:], pred)[0, 1])
    err = rmse(toc[ntr:], pred)
    print(f"  NN TOC test R / RMSE   = {R:.3f} / {err:.3f} wt%")
    assert R > 0.9 and err < 1.0
    print("  PASS")
    return {"R": R, "RMSE": err, "schmoker_2.3": float(schmoker_toc(2.30))}


if __name__ == "__main__":
    test_all()
