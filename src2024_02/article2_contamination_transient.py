"""
Article 2: Fluid Contamination Transient Analysis
Gelvez and Torres-Verdín (Petrophysics, Vol. 65, No. 1, Feb 2024, pp. 32-50)

During formation-tester sampling, the produced fluid is a mixture of
oil-base mud (OBM) filtrate and native reservoir fluid. Contamination
typically decays as a power-law of pumped volume:
    eta(V) = eta_inf + A * V^(-b)
This module fits and predicts the cleanup transient.
"""
import numpy as np
from scipy.optimize import curve_fit


def contamination_model(V, eta_inf, A, b):
    return eta_inf + A * np.power(np.maximum(V, 1e-9), -b)


def fit_transient(V, eta):
    p0 = [float(np.min(eta)), float(np.max(eta) - np.min(eta)) + 1e-3, 0.5]
    popt, _ = curve_fit(contamination_model, V, eta, p0=p0, maxfev=10000)
    return {"eta_inf": popt[0], "A": popt[1], "b": popt[2]}


def predict_volume_to_threshold(params, target_eta):
    """Solve eta_inf + A V^-b = target -> V."""
    delta = target_eta - params["eta_inf"]
    if delta <= 0:
        return np.inf
    return (params["A"] / delta) ** (1.0 / params["b"])


def test_all():
    rng = np.random.default_rng(0)
    V = np.linspace(1, 200, 80)
    true = dict(eta_inf=0.02, A=0.6, b=0.55)
    eta = contamination_model(V, **true) + rng.normal(0, 0.005, V.size)
    fit = fit_transient(V, eta)
    for k in true:
        assert abs(fit[k] - true[k]) < 0.05, (k, fit[k], true[k])
    v_to_5pct = predict_volume_to_threshold(fit, 0.05)
    assert np.isfinite(v_to_5pct) and v_to_5pct > 0
    print("article2 OK | fit:", {k: round(v, 4) for k, v in fit.items()},
          "| V@5%:", round(v_to_5pct, 1))


if __name__ == "__main__":
    test_all()
