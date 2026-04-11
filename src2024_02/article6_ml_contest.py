"""
Article 6: Well-Log-Based Reservoir Property Estimation With Machine Learning:
A Contest Summary
Fu, Yu, Xu, Ashby, McDonald, Pan, Deng, Szabó, Hanzelik, Kalmár, Alatwah, Lee
(Petrophysics, Vol. 65, No. 1, Feb 2024, pp. 108-127)

The 2023 SPWLA PDDA contest asked teams to predict reservoir properties
(porosity, water saturation, permeability) from standard well logs
(GR, RHOB, NPHI, DT, RT). This module shows the contest baseline
workflow: feature scaling + gradient-boosted regression, with the same
RMSE evaluation metric reported in the paper.
"""
import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def train_predict(X_train, y_train, X_test):
    """Use sklearn GBR if available, else fall back to ridge regression."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X_train)
        m = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=0)
        m.fit(sc.transform(X_train), y_train)
        return m.predict(sc.transform(X_test))
    except Exception:
        # Closed-form ridge: beta = (X'X + lI)^-1 X'y
        Xtr = np.column_stack([np.ones(len(X_train)), X_train])
        Xte = np.column_stack([np.ones(len(X_test)), X_test])
        l = 1.0 * np.eye(Xtr.shape[1]); l[0, 0] = 0.0
        beta = np.linalg.solve(Xtr.T @ Xtr + l, Xtr.T @ y_train)
        return Xte @ beta


def synthetic_well(n=600, seed=0):
    rng = np.random.default_rng(seed)
    GR = rng.uniform(20, 150, n)
    RHOB = 2.71 - 1.0 * rng.uniform(0.05, 0.30, n) + rng.normal(0, 0.02, n)
    NPHI = (2.71 - RHOB) / 1.0 + rng.normal(0, 0.01, n)
    DT = 55 + 200 * (2.71 - RHOB) + rng.normal(0, 3, n)
    RT = 10 ** rng.uniform(0, 2.5, n)
    phi = np.clip((2.71 - RHOB) / 1.0, 0, 0.4)
    Sw = np.clip(1.0 / np.sqrt(np.maximum(RT, 1e-3)) * 0.6, 0.05, 1.0)
    X = np.column_stack([GR, RHOB, NPHI, DT, np.log10(RT)])
    return X, phi, Sw


def test_all():
    X, phi, Sw = synthetic_well(800)
    s = 600
    phi_hat = train_predict(X[:s], phi[:s], X[s:])
    sw_hat = train_predict(X[:s], Sw[:s], X[s:])
    e_phi = rmse(phi[s:], phi_hat); e_sw = rmse(Sw[s:], sw_hat)
    assert e_phi < 0.05 and e_sw < 0.20
    print(f"article6 OK | RMSE phi={e_phi:.4f}  Sw={e_sw:.4f}")


if __name__ == "__main__":
    test_all()
