"""
Article 1: Through-Tubing Casing Deformation and Tubing Eccentricity Image
Tool for Well-Integrity Monitoring and Plug Abandonment
(Best of 2021 SPWLA Symposium, Paper 0045)
Yang, Qin, Olson, Rourke (2022)
DOI: 10.30632/PJV63N2-2022a1

Implements the deformation-and-eccentricity (DEC) tool interpretation
pipeline:

  - Magnetostatic transfer function for the per-azimuth Hall-probe
    measurement                                              (Eq. 1)
  - Casing / tubing magnetic-flux density ratio              (Eqs. 2-3)
  - Eccentricity ratio  Ecc = Delta_e / (IR_casing - OR_tubing)   (Eq. 4)
  - Deformation factor  Def = R_A / R_B                       (Eq. 5)
  - Forward model d_obs = T(P)                                (Eq. 6)
  - Bayesian inversion via Gaussian Process Regression with a
    Matern-5/2 covariance kernel                              (Eqs. 7-9)
"""

import numpy as np


# ---------------------------------------------- magnetostatic forward ----

def dec_transfer(delta_mu, delta_r, delta_t, k_mu=1.0, k_r=2.0, k_t=0.5):
    """Linearised transfer function dBr = k_mu*dmu + k_r*dr + k_t*dt (Eq. 1)."""
    return k_mu * delta_mu + k_r * delta_r + k_t * delta_t


def flux_density_ratio(phi_C, phi_T):
    """r_flux = Phi_C / Phi_T  (Eqs. 2-3)."""
    return phi_C / max(phi_T, 1e-12)


# ---------------------------------------------- geometric parameters ---

def eccentricity_ratio(delta_e_m, IR_casing_m, OR_tubing_m):
    """Ecc = Delta_e / (IR_casing - OR_tubing)  (Eq. 4)."""
    return delta_e_m / (IR_casing_m - OR_tubing_m)


def deformation_factor(R_A_m, R_B_m):
    """Def = R_A / R_B  (Eq. 5)."""
    return R_A_m / R_B_m


# ---------------------------------------------- forward model -----------

def synth_24_hall_response(Ecc, theta_deg, Def, gamma_deg,
                          n_probes=24, noise=0.01, seed=0):
    """Synthesise a 24-Hall-probe azimuthal magnetic response.

    Ecc shifts the DC level and adds a cos(theta - theta_0) component;
    Def adds a 2*theta deformation harmonic.
    """
    rng = np.random.default_rng(seed)
    phi = np.linspace(0, 2 * np.pi, n_probes, endpoint=False)
    theta = np.deg2rad(theta_deg)
    gamma = np.deg2rad(gamma_deg)
    d = (1.0
         + Ecc * np.cos(phi - theta)
         + 0.5 * (Def - 1.0) * np.cos(2.0 * (phi - gamma)))
    d += noise * rng.standard_normal(len(d))
    return d


def forward_T(P, n_probes=24, noise=0.0, seed=0):
    """T(P) - the forward operator of Eq. 6.  P = (Ecc, theta, Def, gamma)."""
    Ecc, theta_deg, Def, gamma_deg = P
    return synth_24_hall_response(Ecc, theta_deg, Def, gamma_deg,
                                 n_probes=n_probes, noise=noise, seed=seed)


# ---------------------------------------------- Matern-5/2 GPR ---------

def matern_52(X1, X2, length=1.0):
    """Matern-5/2 covariance kernel (Eq. 8).

        k(r) = (1 + sqrt(5) r / l + 5/3 (r/l)^2) * exp(-sqrt(5) r / l)
    """
    r = np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=-1)
    s5 = np.sqrt(5.0) * r / length
    return (1.0 + s5 + (5.0 / 3.0) * (r / length) ** 2) * np.exp(-s5)


class GPRegressor:
    """Bayesian GP with Matern-5/2 covariance for the inverse problem."""
    def __init__(self, length=1.0, sigma_n=0.05):
        self.length = length
        self.sigma_n = sigma_n
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        K = matern_52(X, X, self.length) + self.sigma_n ** 2 * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)
        return self

    def predict(self, X_new):
        K_s = matern_52(X_new, self.X_train, self.length)
        return K_s @ self.K_inv @ self.y_train


# ---------------------------------------------- Bayesian inversion -----

def bayesian_inversion_GPR(d_obs, n_train=700, n_test=300, n_probes=24,
                           seed=0):
    """Train per-target GPR on synthetic (P -> d) pairs, then predict P
    from the observed measurement d_obs.

    Cyclic-angle handling: instead of predicting the raw angles theta
    and gamma (which wrap at 360 deg and create discontinuities), we
    predict the four Cartesian projections

        Ex = Ecc * cos(theta)     Ey = Ecc * sin(theta)
        Dx = (Def - 1) * cos(2*gamma)     Dy = (Def - 1) * sin(2*gamma)

    and recover (Ecc, theta, Def, gamma) by polar conversion.  The
    factor 2 on gamma reflects the half-period symmetry of the
    deformation harmonic in Eq. 5.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    Ecc_arr = rng.uniform(0.0, 0.8, n)
    theta_arr = rng.uniform(0.0, 360.0, n)
    Def_arr = rng.uniform(0.95, 1.15, n)
    gamma_arr = rng.uniform(0.0, 360.0, n)
    P_data = np.column_stack([Ecc_arr, theta_arr, Def_arr, gamma_arr])
    D_data = np.array([forward_T(P_data[i], n_probes=n_probes,
                                 noise=0.005, seed=i)
                       for i in range(n)])
    Xtr, Xte = D_data[:n_train], D_data[n_train:]
    # Cartesian targets
    t_rad = np.deg2rad(theta_arr)
    g_rad = np.deg2rad(gamma_arr)
    Y = np.column_stack([
        Ecc_arr * np.cos(t_rad),
        Ecc_arr * np.sin(t_rad),
        (Def_arr - 1.0) * np.cos(2.0 * g_rad),
        (Def_arr - 1.0) * np.sin(2.0 * g_rad),
    ])
    Ytr, Yte = Y[:n_train], Y[n_train:]
    preds_test = np.zeros_like(Yte)
    pred_obs_cart = np.zeros(4)
    for j in range(4):
        gpr = GPRegressor(length=1.0, sigma_n=0.05).fit(Xtr, Ytr[:, j])
        preds_test[:, j] = gpr.predict(Xte)
        pred_obs_cart[j] = float(gpr.predict(d_obs[None, :])[0])
    rmse = float(np.sqrt(((preds_test - Yte) ** 2).mean(0)).mean())
    # Polar conversion of pred_obs
    Ex, Ey, Dx, Dy = pred_obs_cart
    Ecc_hat = float(np.sqrt(Ex ** 2 + Ey ** 2))
    theta_hat = float(np.rad2deg(np.arctan2(Ey, Ex)) % 360.0)
    Def_hat = 1.0 + float(np.sqrt(Dx ** 2 + Dy ** 2))
    gamma_hat = float(0.5 * np.rad2deg(np.arctan2(Dy, Dx)) % 360.0)
    return np.array([Ecc_hat, theta_hat, Def_hat, gamma_hat]), rmse


# ---------------------------------------------- tests ----------------

def test_all():
    print("=" * 60)
    print("Article 1: DEC Tool Bayesian-GPR Inversion")
    print("=" * 60)

    # Eq. 1 sanity
    dBr = dec_transfer(0.05, 0.10, 0.20)
    print(f"  Transfer-function output (Eq. 1) = {dBr:.3f}")
    assert dBr > 0

    # Geometric params
    Ecc = eccentricity_ratio(0.003, 0.062, 0.044)
    Def = deformation_factor(0.062, 0.058)
    print(f"  Eccentricity ratio (Eq. 4)       = {Ecc:.3f}")
    print(f"  Deformation factor (Eq. 5)       = {Def:.3f}")
    assert 0.05 < Ecc < 1.0 and 1.0 <= Def < 1.10

    # Synthetic field log: tubing-buckling event at X158 ft
    true_params = (0.40, 60.0, 1.05, 30.0)
    d_obs = forward_T(true_params, n_probes=24, noise=0.01, seed=42)
    pred, rmse = bayesian_inversion_GPR(d_obs, n_train=400, n_test=100, seed=0)
    print(f"  GPR test RMSE (averaged across 4 parameters) = {rmse:.3f}")
    print(f"  Recovered (Ecc, theta, Def, gamma) = "
          f"({pred[0]:.3f}, {pred[1]:.1f}, {pred[2]:.3f}, {pred[3]:.1f})")
    print(f"  True       (Ecc, theta, Def, gamma) = "
          f"({true_params[0]:.3f}, {true_params[1]:.1f}, {true_params[2]:.3f}, {true_params[3]:.1f})")

    assert abs(pred[0] - true_params[0]) < 0.10, "Ecc recovered within 0.10"
    assert abs(pred[2] - true_params[2]) < 0.05, "Def recovered within 0.05"
    print("  PASS")
    return {"rmse_gpr": rmse, "Ecc_recovered": float(pred[0]),
            "Def_recovered": float(pred[2])}


if __name__ == "__main__":
    test_all()
