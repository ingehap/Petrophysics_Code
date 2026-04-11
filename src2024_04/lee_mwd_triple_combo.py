"""
lee_mwd_triple_combo.py
Implementation of ideas from:
Lee et al., "Machine-Learning Prediction of Triple-Combo Logs from Drilling
Dynamics with a Physics-Based Joint Inversion",
Petrophysics, Vol. 65, No. 2 (April 2024), pp. 215-232.

Two stage workflow:
  1. ML regression: drilling dynamics (WOB, RPM, ROP, torque, MSE) -> triple
     combo logs (gamma ray, bulk density, neutron porosity, resistivity).
  2. Physics-based joint inversion: combine the predicted logs to estimate
     porosity, water saturation and shale volume.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def make_drilling_to_log_model():
    return make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=80, random_state=0))


def joint_inversion(gr, rhob, nphi, rt, rho_ma=2.65, rho_fl=1.0, gr_clean=20, gr_shale=120, Rw=0.05, m=2, n=2):
    """Density porosity, Vsh from GR, and Archie water saturation."""
    phi_d = (rho_ma - rhob) / (rho_ma - rho_fl)
    Vsh = np.clip((gr - gr_clean) / (gr_shale - gr_clean), 0, 1)
    phi = np.clip(0.5 * (phi_d + nphi) * (1 - 0.5 * Vsh), 1e-3, 0.5)
    Sw = np.clip((Rw / (phi ** m * rt)) ** (1.0 / n), 0, 1)
    return phi, Vsh, Sw


def synthetic_drilling_and_logs(n=400, rng=None):
    rng = rng or np.random.default_rng(3)
    WOB = rng.uniform(5, 25, n)
    RPM = rng.uniform(60, 180, n)
    ROP = rng.uniform(5, 60, n)
    TRQ = rng.uniform(1, 8, n)
    MSE = 13.6 * (WOB / 2.0 + 120 * np.pi * RPM * TRQ / (ROP + 1))
    X = np.column_stack([WOB, RPM, ROP, TRQ, MSE])
    GR = 30 + 0.4 * MSE / 100 + rng.normal(0, 5, n)
    RHOB = 2.7 - 0.002 * ROP + rng.normal(0, 0.02, n)
    NPHI = 0.05 + 0.003 * ROP + rng.normal(0, 0.01, n)
    RT = np.exp(2 + 0.02 * (200 - GR) + rng.normal(0, 0.2, n))
    Y = np.column_stack([GR, RHOB, NPHI, RT])
    return X, Y


def test_all():
    X, Y = synthetic_drilling_and_logs(500)
    n_train = 400
    model = make_drilling_to_log_model()
    model.fit(X[:n_train], Y[:n_train])
    Yhat = model.predict(X[n_train:])
    err = np.abs(Yhat - Y[n_train:]).mean(0)
    assert (err < [20, 0.2, 0.1, 50]).all(), err
    phi, Vsh, Sw = joint_inversion(Yhat[:, 0], Yhat[:, 1], Yhat[:, 2], Yhat[:, 3])
    assert (0 < phi).all() and (phi < 0.5).all()
    assert (0 <= Vsh).all() and (Vsh <= 1).all()
    assert (0 <= Sw).all() and (Sw <= 1).all()
    print("lee_mwd_triple_combo OK  mean(phi)=%.3f mean(Sw)=%.3f" % (phi.mean(), Sw.mean()))


if __name__ == "__main__":
    test_all()
