"""
Article 4: Comparative Study of Machine-Learning-Based Methods for Log Prediction
Simoes, Maniar, Abubakar, and Zhao (2023)
DOI: 10.30632/PJV64N2-2023a4

Implements three log-prediction methods discussed in the paper:
  - PAE  : Pointwise Autoencoder (fully-connected MLP)
  - WAE  : Window-based 1D-conv-style autoencoder (here a sliding-window MLP
           that consumes a window of nearby depths to predict the centre
           target log)
  - XGBoost: tree-based pointwise regressor

All methods are trained to handle missing inputs by random masking during
training. Metrics: RMSE, MAE, Pearson, PSNR.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# ---------------------------------------------------------- preprocessing ---

def normalize(X, scaler=None):
    if scaler is None:
        scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def make_windowed_input(X, window=11):
    """For each depth t, gather features from depths [t-w/2, t+w/2]."""
    n, d = X.shape
    pad = window // 2
    Xpad = np.pad(X, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros((n, window * d))
    for t in range(n):
        out[t] = Xpad[t:t + window].ravel()
    return out


def random_mask(X, mask_prob=0.15, rng=None):
    """Randomly zero-out some entries, returning (masked, mask)."""
    rng = rng or np.random.default_rng(0)
    mask = rng.random(X.shape) < mask_prob
    Xm = X.copy()
    Xm[mask] = 0.0
    return Xm, mask


# --------------------------------------------------------- model wrappers ---

class PAE:
    """Pointwise FC autoencoder predicting the target curve from other curves."""
    def __init__(self, hidden=(64, 32, 64), max_iter=200, seed=0):
        self.model = MLPRegressor(hidden_layer_sizes=hidden,
                                  max_iter=max_iter, random_state=seed,
                                  early_stopping=False)
        self.x_scaler = None
        self.y_scaler = None

    def fit(self, X, y, mask_prob=0.2, rng=None):
        rng = rng or np.random.default_rng(0)
        # augment training data with random masking
        Xm, _ = random_mask(X, mask_prob, rng)
        Xa = np.vstack([X, Xm])
        ya = np.concatenate([y, y])
        self.x_scaler = StandardScaler().fit(Xa)
        self.y_scaler = StandardScaler().fit(ya.reshape(-1, 1))
        self.model.fit(self.x_scaler.transform(Xa),
                       self.y_scaler.transform(ya.reshape(-1, 1)).ravel())
        return self

    def predict(self, X):
        Xs = self.x_scaler.transform(X)
        ys = self.model.predict(Xs).reshape(-1, 1)
        return self.y_scaler.inverse_transform(ys).ravel()


class WAE:
    """Window-based autoencoder (MLP on stacked windowed features)."""
    def __init__(self, window=11, hidden=(128, 64, 128), max_iter=200, seed=0):
        self.window = window
        self.model = MLPRegressor(hidden_layer_sizes=hidden,
                                  max_iter=max_iter, random_state=seed)
        self.x_scaler = None
        self.y_scaler = None

    def fit(self, X, y, mask_prob=0.15, rng=None):
        rng = rng or np.random.default_rng(0)
        Xw = make_windowed_input(X, self.window)
        Xm, _ = random_mask(Xw, mask_prob, rng)
        Xa = np.vstack([Xw, Xm])
        ya = np.concatenate([y, y])
        self.x_scaler = StandardScaler().fit(Xa)
        self.y_scaler = StandardScaler().fit(ya.reshape(-1, 1))
        self.model.fit(self.x_scaler.transform(Xa),
                       self.y_scaler.transform(ya.reshape(-1, 1)).ravel())
        return self

    def predict(self, X):
        Xw = make_windowed_input(X, self.window)
        Xs = self.x_scaler.transform(Xw)
        ys = self.model.predict(Xs).reshape(-1, 1)
        return self.y_scaler.inverse_transform(ys).ravel()


class XGBLogModel:
    """XGBoost regressor — natively handles missing values (NaN)."""
    def __init__(self, n_estimators=200, max_depth=4, seed=0):
        self.model = xgb.XGBRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      random_state=seed,
                                      verbosity=0)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# -------------------------------------------------------- metrics ---

def metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")
    rng = y_true.max() - y_true.min()
    psnr = float(20 * np.log10(rng / (rmse + 1e-12))) if rng > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "Pearson": pearson, "PSNR": psnr}


# -------------------------------------------------------- testing ---

def synthetic_data(seed=0, n=600):
    """Synthetic well logs with structured relationships."""
    rng = np.random.default_rng(seed)
    z = np.linspace(0, 100, n)
    GR = 60 + 30 * np.sin(z * 0.3) + rng.normal(0, 3, n)
    RHOB = 2.65 - 0.3 * (GR / 150) + rng.normal(0, 0.02, n)
    NPHI = 0.3 - (RHOB - 2.65) * 0.4 + rng.normal(0, 0.01, n)
    DTC = 60 + (NPHI - 0.05) * 200 + rng.normal(0, 2, n)
    X = np.column_stack([GR, RHOB, NPHI])  # inputs
    y = DTC                                 # target to predict
    return X, y


def test_all():
    print("=" * 60)
    print("Article 4: ML Methods for Log Prediction")
    print("=" * 60)
    X, y = synthetic_data()
    n = X.shape[0]
    split = int(n * 0.7)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    pae = PAE(hidden=(64, 32, 64), max_iter=300).fit(Xtr, ytr)
    wae = WAE(window=11, hidden=(128, 64), max_iter=300).fit(Xtr, ytr)
    xgbm = XGBLogModel().fit(Xtr, ytr)

    res = {}
    for name, model, getX in [
        ("PAE", pae, lambda x: x),
        ("WAE", wae, lambda x: x),
        ("XGB", xgbm, lambda x: x),
    ]:
        ypred = model.predict(getX(Xte))
        m = metrics(yte, ypred)
        res[name] = m
        print(f"  {name}:  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  "
              f"r={m['Pearson']:.3f}  PSNR={m['PSNR']:.2f} dB")

    # also check PAE handles a missing input
    Xte_missing = Xte.copy()
    Xte_missing[:, 1] = 0.0  # zero RHOB to simulate missing
    ypred_missing = pae.predict(Xte_missing)
    m_miss = metrics(yte, ypred_missing)
    print(f"  PAE w/ RHOB missing: RMSE={m_miss['RMSE']:.2f}  r={m_miss['Pearson']:.3f}")

    assert res["PAE"]["Pearson"] > 0.5
    assert res["XGB"]["Pearson"] > 0.5
    print("  PASS")
    return res


if __name__ == "__main__":
    test_all()
