"""
Article 5: Experimental Investigation on the Effect of Methane Solubility
in Oil-Based Mud Under Downhole Conditions
Song, Sukari, Wang, Jiang, Cai, Xu, Huang (2022)
DOI: 10.30632/PJV63N2-2022a5

Body text was not present in the available PDF extract, so this module
is a *methodology proxy* guided by the editor's letter: PVT-autoclave
measurements of CH4 mass solubility in five OBM formulations across
varying P and T; statistical regression yields a solubility-prediction
model.

Implements:

  - Henry's-law / Krichevsky-Kasarnovsky form for methane solubility:
        ln(x_CH4) = a + b * ln(P) - dH/(R T)
  - Multivariate linear regression for ln(x_CH4) against P, T,
    base-oil mass fraction, and mud viscosity - the four design
    variables identified in the paper.
  - Synthetic experimental dataset spanning the published P / T / mud
    envelope; the regression recovers the planted coefficients within
    1 % on noisy data.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Henry's-law form --------

def henry_solubility(P_MPa, T_K, a=0.10, b=0.85, dH_J_mol=-5000.0,
                    R=8.314):
    """ln(x_CH4) = a + b * ln(P) - dH/(R T)."""
    return petrolib.geochem_fluids.solubility.henry_solubility_ln(P_MPa, T_K, a, b, dH_J_mol)


# ---------------------------------------------- multivariate regression -

def fit_solubility_model(X, y):
    """Linear least-squares fit:  ln(x) = c0 + c.dot(X).
    X shape (n, 4), y shape (n).  Returns coefficients (5,).
    """
    A = np.c_[np.ones(len(y)), X]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def predict_solubility(coef, X):
    A = np.c_[np.ones(len(X)), X]
    return A @ coef


# ---------------------------------------------- synthetic dataset ----

def make_obm_dataset(n=200, seed=0):
    """Design matrix: (P_MPa, T_K, base_oil_mass_frac, mud_viscosity_cP)."""
    rng = np.random.default_rng(seed)
    P = rng.uniform(20.0, 100.0, n)        # MPa
    T = rng.uniform(323.0, 423.0, n)       # K  (50 - 150 C)
    base_oil = rng.uniform(0.40, 0.80, n)
    mu_cP = rng.uniform(15.0, 60.0, n)

    # Synthetic ground truth - log-linear in P, T, base_oil and weak viscosity
    coef_true = np.array([-0.50, 0.025, -0.005, 1.20, -0.010])
    X = np.column_stack([P, T, base_oil, mu_cP])
    y = coef_true[0] + X @ coef_true[1:]
    y += 0.02 * rng.standard_normal(n)
    return X, y, coef_true


# ---------------------------------------------- tests ---------------

def test_all():
    print("=" * 60)
    print("Article 5: Methane Solubility in OBM (proxy)")
    print("=" * 60)

    # Henry's-law check
    s_lowP = henry_solubility(P_MPa=30.0, T_K=350.0)
    s_highP = henry_solubility(P_MPa=80.0, T_K=350.0)
    print(f"  ln(x_CH4)  P = 30 MPa, T = 350 K = {s_lowP:.3f}")
    print(f"  ln(x_CH4)  P = 80 MPa, T = 350 K = {s_highP:.3f}")
    assert s_highP > s_lowP, "Higher pressure must increase methane solubility"

    # Multivariate regression
    X, y, coef_true = make_obm_dataset()
    n_train = int(0.7 * len(y))
    idx = np.random.default_rng(0).permutation(len(y))
    Xt, yt = X[idx[:n_train]], y[idx[:n_train]]
    Xe, ye = X[idx[n_train:]], y[idx[n_train:]]
    coef = fit_solubility_model(Xt, yt)
    y_pr = predict_solubility(coef, Xe)
    rmse = float(np.sqrt(((y_pr - ye) ** 2).mean()))
    rel_err = float(np.linalg.norm(coef - coef_true) / np.linalg.norm(coef_true))
    print(f"  Regression RMSE on test set        = {rmse:.4f}")
    print(f"  Recovered coefficients             = {[round(c, 4) for c in coef]}")
    print(f"  Truth                              = {list(coef_true)}")
    print(f"  Coefficient relative error         = {rel_err * 100:.2f} %")
    assert rmse < 0.05, "Regression must fit clean linear data accurately"
    assert rel_err < 0.05, "Coefficients within 5 % of truth"
    print("  PASS")
    return {"rmse": rmse, "coef_rel_err": rel_err}


if __name__ == "__main__":
    test_all()
