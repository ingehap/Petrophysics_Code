"""
Cely et al. (2023), Petrophysics 64(6): 919-930.
Reservoir oil viscosity estimation in the Breidablikk Field from advanced mud-gas
ratios + cuttings geochemistry, compared against PVT.

This module implements (a) gas-ratio derived viscosity proxies (wetness, balance,
character) and (b) a simple multivariate regressor to predict log10(viscosity).
"""
import numpy as np
from numpy.linalg import lstsq


def gas_ratios(c1, c2, c3, ic4, nc4, ic5, nc5):
    """Pixler/Haworth-style mud-gas ratios."""
    light = c1 + c2 + c3 + ic4 + nc4 + ic5 + nc5
    wetness = 100 * (c2 + c3 + ic4 + nc4 + ic5 + nc5) / light
    balance = (c1 + c2) / (c3 + ic4 + nc4 + ic5 + nc5 + 1e-9)
    character = (ic4 + nc4 + ic5 + nc5) / (c3 + 1e-9)
    return dict(wetness=wetness, balance=balance, character=character)


def fit_viscosity_model(features, log_visc):
    """Linear regression: log10(visc_cP) = X @ beta."""
    X = np.column_stack([np.ones(len(features)), features])
    beta, *_ = lstsq(X, log_visc, rcond=None)
    return beta


def predict_viscosity(features, beta):
    X = np.column_stack([np.ones(len(features)), features])
    return 10 ** (X @ beta)


def test_all():
    rng = np.random.default_rng(2)
    n = 80
    c1 = rng.uniform(40, 90, n)
    c2 = rng.uniform(2, 15, n); c3 = rng.uniform(1, 10, n)
    ic4 = rng.uniform(0.1, 3, n); nc4 = rng.uniform(0.1, 3, n)
    ic5 = rng.uniform(0.05, 1.5, n); nc5 = rng.uniform(0.05, 1.5, n)
    r = gas_ratios(c1, c2, c3, ic4, nc4, ic5, nc5)
    feats = np.column_stack([r["wetness"], np.log(r["balance"]), r["character"]])
    true_beta = np.array([0.5, 0.04, -0.3, 0.1])
    log_v = (np.column_stack([np.ones(n), feats]) @ true_beta
             + rng.normal(0, 0.05, n))
    beta = fit_viscosity_model(feats, log_v)
    pred = predict_viscosity(feats, beta)
    err = np.mean(np.abs(np.log10(pred) - log_v))
    print("Cely et al. viscosity model:")
    print(f"  beta: {beta.round(3)}")
    print(f"  mean |log10 error|: {err:.3f}")
    assert err < 0.1
    print("  PASS")


if __name__ == "__main__":
    test_all()
