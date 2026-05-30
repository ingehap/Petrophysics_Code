"""
Article 3: A Rigorous Data-Driven Approach to Predict Poisson's Ratio of
           Carbonate Rocks Using a Functional Network
Tariq, Abdulraheem, Mahmoud, Ahmed (2018)
DOI: 10.30632/PJV59N6Y2018a2

A functional network predicts the (dynamic) Poisson's ratio of carbonate rocks
from conventional logs.  A functional network expands the inputs in a set of
basis functions and fits the combining coefficients by least squares - a
transparent, rigorous alternative to a black-box neural network.

Implements:

  - Dynamic Poisson's ratio from Vp, Vs (the physical reference)
  - Functional-network basis expansion (polynomial + interaction terms)
  - Least-squares coefficient fit and prediction
  - Correlation coefficient R and RMSE

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so this is a faithful standard-form reconstruction of the
functional-network regression the paper applies.
"""

import numpy as np


# ---------------------------------------------- reference ---------------

def dynamic_poisson(vp, vs):
    """Dynamic Poisson's ratio  nu = (Vp^2 - 2Vs^2)/(2(Vp^2 - Vs^2))."""
    vp2, vs2 = np.asarray(vp, float) ** 2, np.asarray(vs, float) ** 2
    return (vp2 - 2.0 * vs2) / (2.0 * (vp2 - vs2))


# ---------------------------------------------- functional network ------

def basis_expansion(X):
    """Functional-network basis: [1, xi, xi^2, xi*xj] for the input columns."""
    X = np.asarray(X, float)
    cols = [np.ones(len(X))]
    cols += [X[:, i] for i in range(X.shape[1])]
    cols += [X[:, i] ** 2 for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            cols.append(X[:, i] * X[:, j])
    return np.column_stack(cols)


def fit_functional_network(X, y):
    """Least-squares fit of the basis-expansion coefficients."""
    Phi = basis_expansion(X)
    coef, *_ = np.linalg.lstsq(Phi, np.asarray(y, float), rcond=None)
    return coef


def predict_functional_network(coef, X):
    return basis_expansion(X) @ coef


def correlation(y, yhat):
    return float(np.corrcoef(np.asarray(y, float), np.asarray(yhat, float))[0, 1])


def rmse(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Poisson's Ratio via Functional Network")
    print("=" * 60)

    # Physical reference: Poisson's ratio is in (0, 0.5) for typical rock
    nu = dynamic_poisson(5.5, 3.0)
    print(f"  dynamic nu (Vp=5.5,Vs=3.0) = {nu:.3f}")
    assert 0.0 < nu < 0.5

    # Synthetic carbonate log set; target = dynamic Poisson's ratio
    rng = np.random.default_rng(2)
    n = 400
    vp = rng.uniform(4.5, 6.5, n)
    vs = vp / rng.uniform(1.7, 2.0, n)            # Vp/Vs ratio
    rho = rng.uniform(2.4, 2.75, n)
    nu_true = dynamic_poisson(vp, vs) + rng.normal(0, 0.005, n)
    X = np.column_stack([vp, vs, rho])

    ntr = 280
    coef = fit_functional_network(X[:ntr], nu_true[:ntr])
    pred = predict_functional_network(coef, X[ntr:])
    R = correlation(nu_true[ntr:], pred)
    err = rmse(nu_true[ntr:], pred)
    print(f"  functional-net R / RMSE = {R:.3f} / {err:.4f}")
    assert R > 0.95 and err < 0.02
    print("  PASS")
    return {"nu_example": float(nu), "R": R, "RMSE": err}


if __name__ == "__main__":
    test_all()
