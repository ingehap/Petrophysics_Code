"""
Article 9: Dual Neural Network Architecture for Determining Permeability and
           Associated Uncertainty
Kausik, Prado, Gkortsas, Venkataramanan, Datir, Johansen (2021)
DOI: 10.30632/PJV62N1-2021a8

Permeability is predicted from NMR / log inputs by a network with two output
heads: a mean head that regresses log-permeability and a variance head that
regresses its predictive uncertainty.  Training minimizes the heteroscedastic
Gaussian negative-log-likelihood, so the network learns to report larger
uncertainty where the inputs are less informative (here, low-porosity rock).

Implements:

  - Two-head MLP (shared tanh hidden layer; mean + log-variance heads)
  - Heteroscedastic Gaussian NLL  L = 0.5 * mean[ (y-mu)^2 e^-s + s ]
  - Full-batch backpropagation training (numpy only)
  - Calibrated mean + sigma prediction of log-permeability

Note: this issue's source PDF has no usable text layer, so the dual-head
architecture and the heteroscedastic NLL are faithful standard-form
reconstructions of the method the paper describes (a compact numpy network
stands in for the published deep network).  log10-permeability in log10(mD).
"""

import numpy as np


# ---------------------------------------------- network -----------------

def train_dual_network(X, y, hidden=16, iters=4000, lr=0.05, seed=0):
    """Train a two-head MLP by minimizing the heteroscedastic Gaussian NLL.

    Returns (params, history) where history is the per-iteration NLL.
    Features and target are standardized internally for stable training.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, d = X.shape

    xm, xs = X.mean(0), X.std(0) + 1e-9
    ym, ys = y.mean(), y.std() + 1e-9
    Xn = (X - xm) / xs
    yn = (y - ym) / ys

    # Mean head starts small; the variance head starts at zero output (s = 0,
    # unit variance) so the mean regression settles before the heteroscedastic
    # term sharpens the variance - this keeps the NLL training stable.
    W1 = rng.normal(0, 0.5, (d, hidden)); b1 = np.zeros(hidden)
    w_mu = rng.normal(0, 0.1, hidden); b_mu = 0.0
    w_s = np.zeros(hidden); b_s = 0.0

    history = []
    for _ in range(iters):
        H = np.tanh(Xn @ W1 + b1)                  # (N, hidden)
        mu = H @ w_mu + b_mu                        # (N,)
        s = np.clip(H @ w_s + b_s, -6.0, 6.0)      # (N,) log-variance
        inv = np.exp(-s)
        resid = mu - yn
        loss = 0.5 * np.mean(resid ** 2 * inv + s)
        history.append(loss)

        dmu = (resid * inv) / N                     # dL/dmu
        ds = 0.5 * (1.0 - resid ** 2 * inv) / N     # dL/ds
        gw_mu = H.T @ dmu; gb_mu = dmu.sum()
        gw_s = H.T @ ds; gb_s = ds.sum()
        dH = (np.outer(dmu, w_mu) + np.outer(ds, w_s)) * (1.0 - H ** 2)
        gW1 = Xn.T @ dH; gb1 = dH.sum(0)

        W1 -= lr * gW1; b1 -= lr * gb1
        w_mu -= lr * gw_mu; b_mu -= lr * gb_mu
        w_s -= lr * gw_s; b_s -= lr * gb_s

    params = dict(W1=W1, b1=b1, w_mu=w_mu, b_mu=b_mu, w_s=w_s, b_s=b_s,
                  xm=xm, xs=xs, ym=ym, ys=ys)
    return params, history


def predict(params, X):
    """Predict (mean, sigma) of log-permeability for inputs X (de-standardized)."""
    p = params
    Xn = (np.asarray(X, float) - p["xm"]) / p["xs"]
    H = np.tanh(Xn @ p["W1"] + p["b1"])
    mu = H @ p["w_mu"] + p["b_mu"]
    s = np.clip(H @ p["w_s"] + p["b_s"], -6.0, 6.0)
    mean = mu * p["ys"] + p["ym"]
    sigma = np.sqrt(np.exp(s)) * p["ys"]
    return mean, sigma


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: Dual NN Permeability + Uncertainty")
    print("=" * 60)

    rng = np.random.default_rng(0)
    N = 500
    phi = rng.uniform(0.05, 0.30, N)
    t2lm = rng.uniform(40.0, 300.0, N)            # ms

    # True log-permeability follows an SDR-like law; measurement noise is
    # heteroscedastic - larger in low-porosity (less informative) rock.
    logk_true = 0.60 + 4.0 * np.log10(phi) + 2.0 * np.log10(t2lm)
    noise_std = 0.05 + 0.8 * (0.30 - phi)         # bigger at low phi
    y = logk_true + rng.normal(0.0, 1.0, N) * noise_std

    X = np.column_stack([phi, np.log10(t2lm)])
    params, hist = train_dual_network(X, y, iters=8000, lr=0.02, seed=1)

    # 1) Training reduces the NLL
    print(f"  NLL start / end        = {hist[0]:.3f} / {hist[-1]:.3f}")
    assert hist[-1] < hist[0]

    # 2) Mean head recovers the underlying (noise-free) permeability law
    mean, sigma = predict(params, X)
    rmse = float(np.sqrt(np.mean((mean - logk_true) ** 2)))
    print(f"  mean-head RMSE vs truth = {rmse:.3f} log10(mD)")
    assert rmse < 0.5

    # 3) Variance head reports larger uncertainty in the noisy (low-phi) rock
    low = phi < 0.12
    high = phi > 0.22
    sig_low = float(sigma[low].mean())
    sig_high = float(sigma[high].mean())
    print(f"  mean sigma low-phi / high-phi = {sig_low:.3f} / {sig_high:.3f}")
    assert np.all(sigma > 0)
    assert sig_low > sig_high                      # learned heteroscedasticity
    print("  PASS")
    return {"rmse": rmse, "sigma_lowphi": sig_low, "sigma_highphi": sig_high,
            "nll_end": hist[-1]}


if __name__ == "__main__":
    test_all()
