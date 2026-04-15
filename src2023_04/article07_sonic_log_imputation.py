"""
Article 7: Sonic Well-Log Imputation Through Machine-Learning-Based
Uncertainty Models
Maldonado-Cruz, Foster, and Pyrcz (2023)
DOI: 10.30632/PJV64N2-2023a7

Implements:
  - Mutual-information feature selection
  - GBDT ensemble (multiple seeds) producing a non-parametric predictive CDF
  - Goodness metric (Maldonado-Cruz & Pyrcz 2021) for uncertainty model
    accuracy & precision (Eq. 4)
  - Hyperparameter scan that maximises goodness rather than just MSE
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler


# ----------------------------------------------- feature selection ---

def select_features_by_mi(X, y, top_k=5):
    """Mutual information based feature ranking."""
    mi = mutual_info_regression(X, y, random_state=0)
    order = np.argsort(mi)[::-1]
    return order[:top_k], mi


# ------------------------------------------------- ensemble GBDT ---

class GBDTEnsemble:
    """An ensemble of gradient-boosted regressors with different seeds."""
    def __init__(self, n_models=10, n_estimators=200, learning_rate=0.05,
                 max_depth=3, base_seed=0):
        self.n_models = n_models
        self.models = [
            GradientBoostingRegressor(n_estimators=n_estimators,
                                      learning_rate=learning_rate,
                                      max_depth=max_depth,
                                      subsample=0.7,
                                      random_state=base_seed + i)
            for i in range(n_models)
        ]

    def fit(self, X, y):
        # subsample rows differently for each member
        rng = np.random.default_rng(0)
        n = X.shape[0]
        for m in self.models:
            idx = rng.choice(n, int(n * 0.8), replace=True)
            m.fit(X[idx], y[idx])
        return self

    def predict_ensemble(self, X):
        """Returns array (n_samples, n_models) of predictions."""
        return np.column_stack([m.predict(X) for m in self.models])

    def predict_mean_std(self, X):
        P = self.predict_ensemble(X)
        return P.mean(axis=1), P.std(axis=1)

    def cdf_at(self, X, y_true):
        """Nonparametric CDF F_y(u_i; y(u_i)) at each test point."""
        P = self.predict_ensemble(X)
        # rank of true value among the predictions
        return np.mean(P <= y_true.reshape(-1, 1), axis=1)


# ---------------------------------------------- goodness metric ---

def indicator_xi(cdf_values, p_low, p_high):
    """xi(u_i; p) = 1 if true value falls inside symmetric p-interval."""
    return ((cdf_values >= p_low) & (cdf_values <= p_high)).astype(float)


def goodness_score(cdf_values, n_bins=20):
    """
    Compute the goodness metric (Maldonado-Cruz & Pyrcz 2021).
    Returns the mean over symmetric probability intervals of:
        a(p) = 1 - 2 * |xi(p) - p|
    """
    ps = np.linspace(0.05, 0.95, n_bins)
    a = []
    for p in ps:
        lo = (1 - p) / 2
        hi = 1 - lo
        xi = indicator_xi(cdf_values, lo, hi).mean()
        a.append(1.0 - 2.0 * abs(xi - p))
    return float(np.mean(a))


def loss_function(y_true, y_pred_mean, cdf_values, alpha=0.5):
    """
    Combined loss: alpha * (1 - normalised_MAE) + (1 - alpha) * goodness.
    Higher is better — bounded in [0, 1].  Mirrors Eq. 4 of the paper.
    """
    mae = np.mean(np.abs(y_true - y_pred_mean))
    norm_mae = mae / (y_true.std() + 1e-9)
    accuracy_term = max(0.0, 1.0 - norm_mae)
    g = goodness_score(cdf_values)
    return alpha * accuracy_term + (1 - alpha) * g


def hyperparameter_scan(X_tr, y_tr, X_te, y_te,
                        learning_rates=(0.01, 0.05, 0.1),
                        n_estimators_list=(100, 200, 400),
                        n_models=8):
    results = []
    for lr in learning_rates:
        for ne in n_estimators_list:
            ens = GBDTEnsemble(n_models=n_models, n_estimators=ne,
                               learning_rate=lr).fit(X_tr, y_tr)
            mean, _ = ens.predict_mean_std(X_te)
            cdf = ens.cdf_at(X_te, y_te)
            score = loss_function(y_te, mean, cdf)
            mse = float(np.mean((y_te - mean) ** 2))
            g = goodness_score(cdf)
            results.append(dict(lr=lr, ne=ne, score=score, mse=mse, goodness=g))
    return sorted(results, key=lambda r: -r["score"])


# --------------------------------------------------------- testing ---

def synthetic_data(seed=0, n=600):
    """Conventional logs + DTC/DTS targets with structured noise."""
    rng = np.random.default_rng(seed)
    GR = rng.uniform(20, 150, n)
    NPHI = 0.15 + 0.0015 * GR + rng.normal(0, 0.02, n)
    RHOB = 2.65 - 1.5 * NPHI + rng.normal(0, 0.05, n)
    PEF = 2.0 + 0.005 * GR + rng.normal(0, 0.3, n)
    RT = 10 ** (1.0 + 0.01 * GR + rng.normal(0, 0.3, n))
    CALI = 8.5 + rng.normal(0, 0.2, n)
    DTC = 60 + 200 * NPHI + rng.normal(0, 3, n)
    X = np.column_stack([GR, NPHI, RHOB, PEF, np.log10(RT), CALI])
    return X, DTC


def test_all():
    print("=" * 60)
    print("Article 7: Sonic Log Imputation with Uncertainty")
    print("=" * 60)
    X, y = synthetic_data()

    # MI feature selection
    top, mi = select_features_by_mi(X, y, top_k=4)
    names = ["GR", "NPHI", "RHOB", "PEF", "logRT", "CALI"]
    print(f"  Top MI features: {[names[i] for i in top]}  (MI={mi[top].round(2)})")
    Xs = X[:, top]
    Xs = MinMaxScaler().fit_transform(Xs)

    n = Xs.shape[0]
    split = int(n * 0.7)
    Xtr, Xte, ytr, yte = Xs[:split], Xs[split:], y[:split], y[split:]

    # build ensemble
    ens = GBDTEnsemble(n_models=12).fit(Xtr, ytr)
    mean, std = ens.predict_mean_std(Xte)
    cdf = ens.cdf_at(Xte, yte)
    g = goodness_score(cdf)
    rmse = float(np.sqrt(np.mean((yte - mean) ** 2)))
    print(f"  Ensemble RMSE = {rmse:.2f}  goodness = {g:.3f}  mean unc = {std.mean():.2f}")

    # hyperparameter scan
    scan = hyperparameter_scan(Xtr, ytr, Xte, yte,
                               learning_rates=(0.05, 0.1),
                               n_estimators_list=(100, 200),
                               n_models=6)
    best = scan[0]
    print(f"  Best by goodness-aware loss: lr={best['lr']} ne={best['ne']} "
          f"score={best['score']:.3f}")

    assert g > 0.3, "goodness too low"
    print("  PASS")
    return {"rmse": rmse, "goodness": g, "best": best}


if __name__ == "__main__":
    test_all()
