"""
Article 6: Prediction of Sonic Wave Transit Times From Drilling Parameters While
           Horizontal Drilling in Carbonate Rocks Using Neural Networks
Gowida, Elkatatny (2020)
DOI: 10.30632/PJV61N5-2020a6

Two artificial neural networks predict the compressional and shear transit times
(DTC, DTS) from six surface drilling parameters - weight on bit, rotary speed,
rate of penetration, torque, standpipe pressure, and mud flow rate - so sonic
logs can be estimated where none were run.  The predicted slownesses then feed
standard geomechanics relations for the dynamic elastic moduli.

Implements:

  - Min-max feature normalization
  - Compact single-hidden-layer ANN (tan-sigmoid) regressor for DTC/DTS
  - Correlation coefficient R and average absolute percentage error AAPE
  - Dynamic Poisson's ratio and Young's modulus from Vp, Vs, rho   (Eqs. 1-2)

Note: this issue's PDF text layer kept the equation numbers / variable
definitions but dropped the typeset glyphs, so the dynamic-moduli closed forms
are the standard Fjaer et al. expressions and R/AAPE are the standard metric
definitions.  The paper's network used ~20-23 neurons and reported R ~0.94 and
AAPE ~1-1.9%; this numpy net reaches comparable accuracy on synthetic data.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

DRILLING_INPUTS = ["WOB", "RPM", "ROP", "Torque", "SPP", "GPM"]


# ---------------------------------------------- metrics -----------------

def correlation_coefficient(y, yhat):
    """Pearson correlation coefficient R between actual and predicted."""
    return petrolib.ml_stats.pearson_r(y, yhat)


def aape(y, yhat):
    """Average absolute percentage error (%)  mean(|y-yhat|/|y|)*100."""
    return petrolib.ml_stats.mape(y, yhat)


# ---------------------------------------------- ANN ---------------------

def train_ann(X, y, hidden=20, iters=4000, lr=0.05, seed=0):
    """Single-hidden-layer tanh ANN regressor (min-max in, standardized target)."""
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
    """Predict DTC/DTS for inputs X (de-standardized)."""
    Xn = (np.asarray(X, float) - p["xmin"]) / (p["xmax"] - p["xmin"] + 1e-12)
    H = np.tanh(Xn @ p["W1"] + p["b1"])
    return (H @ p["w2"] + p["b2"]) * p["ys"] + p["ym"]


# ---------------------------------------------- dynamic moduli ----------

def dynamic_poisson(vp, vs):
    """Dynamic Poisson's ratio  nu = (Vp^2 - 2Vs^2)/(2(Vp^2 - Vs^2))  (Eq. 1)."""
    return petrolib.acoustic_geomech.poisson_from_velocity(vp, vs)


def dynamic_youngs(rho, vp, vs):
    """Dynamic Young's modulus  E = rho*Vs^2*(3Vp^2-4Vs^2)/(Vp^2-Vs^2)  (Eq. 2).

    rho in g/cm^3, V in km/s -> E in GPa.
    """
    return petrolib.acoustic_geomech.youngs_poisson_dynamic(vp, vs, rho)[0]


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Sonic Transit Times From Drilling Params (ANN)")
    print("=" * 60)

    # Synthetic field set: DTC from six drilling parameters + mild noise
    rng = np.random.default_rng(6)
    n = 600
    WOB = rng.uniform(5, 40, n); RPM = rng.uniform(60, 180, n)
    ROP = rng.uniform(5, 60, n); TOR = rng.uniform(2, 12, n)
    SPP = rng.uniform(1500, 3500, n); GPM = rng.uniform(300, 700, n)
    X = np.column_stack([WOB, RPM, ROP, TOR, SPP, GPM])
    Xs = (X - X.mean(0)) / X.std(0)
    # a smooth response with a mild nonlinearity (DTC ~ 45-95 us/ft)
    dtc = (70.0 + 6 * Xs[:, 2] - 4 * Xs[:, 0] + 3 * Xs[:, 4]
           + 2 * np.tanh(Xs[:, 1]) + rng.normal(0, 1.0, n))

    ntr = 450
    p = train_ann(X[:ntr], dtc[:ntr], hidden=20, iters=4000, lr=0.05, seed=1)
    pred = predict_ann(p, X[ntr:])
    R = correlation_coefficient(dtc[ntr:], pred)
    err = aape(dtc[ntr:], pred)
    print(f"  test R / AAPE          = {R:.3f} / {err:.2f} %")
    assert R > 0.9 and err < 5.0

    # Dynamic moduli from typical carbonate velocities
    vp, vs, rho = 6.0, 3.2, 2.6        # km/s, km/s, g/cc
    nu = dynamic_poisson(vp, vs)
    E = dynamic_youngs(rho, vp, vs)
    print(f"  dynamic nu / E         = {nu:.3f} / {E:.1f} GPa")
    assert 0.0 < nu < 0.5 and 30.0 < E < 120.0
    # softer rock (lower velocities) gives a lower modulus
    assert dynamic_youngs(2.4, 4.5, 2.4) < E
    print("  PASS")
    return {"R": R, "AAPE": err, "nu": float(nu), "E_GPa": float(E)}


if __name__ == "__main__":
    test_all()
