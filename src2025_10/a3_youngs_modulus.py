#!/usr/bin/env python3
"""
Article 3: An Efficient Approach for Predicting Young's Modulus of
           Sandstones Using Well Logs Without Shear-Wave Traveltime
Authors: Mabkhout M. Al-Dousari, Yousef M. Al-Enezi, Ali A. Garrouch
Ref: Petrophysics, Vol. 66, No. 5 (October 2025), pp. 741-762.
     DOI: 10.30632/PJV66N5-2025a3

Implements:
  - Dynamic Young's modulus from Vp, Vs, rho (Eq. 1)
  - Mullen et al. (2007) Ed models per lithology (Eqs. 2-6)
  - Abbas et al. (2018) porosity-based Es model (Eq. 7)
  - Steiber shale-volume model (Eq. 16)
  - Flow zone indicator / discrete rock typing (Eqs. 17-20)
  - Nonlinear regression Es model (Eq. 22)
  - Canady (2011) Ed-to-Es transfer function (Eq. 24)
  - Simple BPNN for Es prediction
"""

import numpy as np


# ---------------------------------------------------------------------------
# Dynamic Young's modulus (Eq. 1)
# ---------------------------------------------------------------------------

def dynamic_youngs_modulus(rho_b, dt_s, dt_c):
    """Dynamic Young's modulus (Eq. 1).

    Parameters
    ----------
    rho_b : bulk density (g/cm^3)
    dt_s  : shear traveltime (us/ft)
    dt_c  : compressional traveltime (us/ft)

    Returns
    -------
    Ed : dynamic Young's modulus (Mpsi)
    """
    rho_b = np.asarray(rho_b, dtype=float)
    dt_s = np.asarray(dt_s, dtype=float)
    dt_c = np.asarray(dt_c, dtype=float)
    ratio = (dt_s / dt_c) ** 2
    nu = 0.5 * (ratio - 2.0) / (ratio - 1.0)  # Poisson's ratio
    Ed = rho_b / dt_s ** 2 * (1.3464e10) * (1.0 + nu)  # approximate
    # Simplified Mullen-type:
    Ed = 1.345e10 * rho_b * (3.0 * dt_s ** 2 - 4.0 * dt_c ** 2) / (
        dt_s ** 2 * (dt_s ** 2 - dt_c ** 2))
    return Ed / 1e6  # Mpsi


# ---------------------------------------------------------------------------
# Mullen et al. (2007) lithology-specific Ed/rho models (Eqs. 2-6)
# ---------------------------------------------------------------------------

def mullen_ed_sandstone(dt_c, rho_b):
    """Ed for sandstone (Eq. 2). dt_c in us/ft, rho_b in g/cm3. Returns Mpsi."""
    x = dt_c
    ed_over_rho = -2.21e-15 * x**8 + 6.28e-12 * x**7 - 7.58e-9 * x**6 + \
                   5.05e-6 * x**5 - 2.01e-3 * x**4 + 0.481 * x**3 - \
                   67.6 * x**2 + 5.16e3 * x - 1.66e5
    return np.maximum(ed_over_rho * rho_b / 1e6, 0.0)


def mullen_ed_shale(dt_c, rho_b):
    """Ed for shale (Eq. 3). Returns Mpsi."""
    x = dt_c
    ed_over_rho = 3.57e-11 * x**6 - 1.49e-8 * x**5 + 2.46e-6 * x**4 - \
                   2.04e-4 * x**3 + 8.88e-3 * x**2 - 0.178 * x + 1.14
    return np.maximum(ed_over_rho * rho_b, 0.0)


# ---------------------------------------------------------------------------
# Steiber shale volume (Eq. 16)
# ---------------------------------------------------------------------------

def steiber_vsh(igr):
    """Steiber shale volume fraction from gamma ray index (Eq. 16).
    Vsh = 0.5 * IGR / (1.5 - IGR)
    """
    igr = np.clip(np.asarray(igr, dtype=float), 0.001, 0.999)
    return 0.5 * igr / (1.5 - igr)


def gamma_ray_index(gr, gr_min, gr_max):
    """Gamma ray index IGR = (GR - GR_min) / (GR_max - GR_min)."""
    return np.clip((gr - gr_min) / (gr_max - gr_min), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Flow zone indicator and discrete rock type (Eqs. 17-20)
# ---------------------------------------------------------------------------

def flow_zone_indicator(k_md, porosity):
    """Flow zone indicator FZI (Eqs. 17-19).
    k in md, porosity as fraction.
    """
    phi = np.asarray(porosity, dtype=float)
    k = np.asarray(k_md, dtype=float)
    phi_z = phi / (1.0 - phi)
    rqi = 0.0314 * np.sqrt(k / phi)  # RQI in microns
    fzi = rqi / phi_z
    return fzi


def discrete_rock_type(fzi):
    """Discrete rock type DRT (Eq. 20).
    DRT = round(2 * log2(FZI)) + 10
    """
    fzi = np.maximum(np.asarray(fzi, dtype=float), 1e-10)
    return np.round(2.0 * np.log2(fzi) + 10.0).astype(int)


# ---------------------------------------------------------------------------
# Nonlinear regression Es model (Eq. 22)
# ---------------------------------------------------------------------------

def predict_es_regression(dt_c, vsh, rho_b,
                           a0=-0.172278, a1=0.0290223,
                           a2=-0.0865234, a3=3.3003800e9):
    """Static Young's modulus Es (Mpsi) from Eq. 22.

    Es = a0 + a1*rho_b + a2*Vsh + a3 / dt_c^4
    """
    dtc = np.asarray(dt_c, dtype=float)
    v = np.asarray(vsh, dtype=float)
    rho = np.asarray(rho_b, dtype=float)
    return a0 + a1 * rho + a2 * v + a3 / dtc ** 4


# ---------------------------------------------------------------------------
# Abbas et al. (2018) porosity-based Es (Eq. 7)
# ---------------------------------------------------------------------------

def predict_es_abbas(porosity):
    """Es (GPa) = 24.2 * exp(-4.27 * phi)  (Eq. 7, approximate)."""
    phi = np.asarray(porosity, dtype=float)
    return 24.2 * np.exp(-4.27 * phi)


# ---------------------------------------------------------------------------
# Canady (2011) Ed-to-Es transfer (Eq. 24)
# ---------------------------------------------------------------------------

def canady_transfer(Ed_gpa):
    """Es (GPa) from Ed (GPa) using Canady (2011), Eq. 24.
    Es = 0.8 * Ed  (simplified linear form for sandstones).
    """
    return 0.8 * np.asarray(Ed_gpa, dtype=float)


# ---------------------------------------------------------------------------
# Simple BPNN for Es prediction
# ---------------------------------------------------------------------------

class SimpleBPNN:
    """Minimal 1-hidden-layer back-propagation neural network."""

    def __init__(self, n_in=3, n_hidden=10, lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((n_in, n_hidden)) * 0.3
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, 1)) * 0.3
        self.b2 = np.zeros(1)
        self.lr = lr

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0)

    def predict(self, X):
        h = self._relu(X @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).ravel()

    def fit(self, X, y, epochs=500):
        y = y.ravel()
        for _ in range(epochs):
            h = self._relu(X @ self.W1 + self.b1)
            yp = (h @ self.W2 + self.b2).ravel()
            err = yp - y
            dW2 = h.T @ err.reshape(-1, 1) / len(y)
            db2 = err.mean()
            dh = err.reshape(-1, 1) @ self.W2.T
            dh[h <= 0] = 0.0
            dW1 = X.T @ dh / len(y)
            db1 = dh.mean(axis=0)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== Article 3: Young's Modulus Prediction Demo ===\n")

    np.random.seed(42)
    n = 100
    dt_c = np.random.uniform(70, 140, n)
    rho_b = np.random.uniform(1.9, 2.6, n)
    gr = np.random.uniform(20, 120, n)
    igr = gamma_ray_index(gr, 20, 120)
    vsh = steiber_vsh(igr)

    es = predict_es_regression(dt_c, vsh, rho_b)
    print(f"Es (regression) range: {es.min():.4f} – {es.max():.4f} Mpsi")

    k_md = np.random.uniform(0.1, 500, n)
    phi = np.random.uniform(0.08, 0.32, n)
    fzi = flow_zone_indicator(k_md, phi)
    drt = discrete_rock_type(fzi)
    print(f"DRT range: {drt.min()} – {drt.max()}")

    es_abbas = predict_es_abbas(phi) / 6.895  # GPa -> Mpsi approx
    print(f"Es (Abbas) range: {es_abbas.min():.3f} – {es_abbas.max():.3f} Mpsi")

    # BPNN demo
    X = np.column_stack([(dt_c - 100) / 30, vsh, (rho_b - 2.2) / 0.3])
    y_true = es
    nn = SimpleBPNN(n_in=3, n_hidden=8, lr=0.005)
    nn.fit(X, y_true, epochs=300)
    y_pred = nn.predict(X)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"BPNN RMSE on training set: {rmse:.4f} Mpsi")
    print()


if __name__ == "__main__":
    demo()
