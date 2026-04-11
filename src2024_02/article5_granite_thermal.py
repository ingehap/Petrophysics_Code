"""
Article 5: Experimental Research on the Permeability of Granite Under
Different Temperature Cycles
Yu, Li, Wu, Wang, Zhang, Zhao (Petrophysics, Vol. 65, No. 1, Feb 2024, pp. 95-107)

The authors observe that granite permeability k increases with peak
thermal-cycle temperature T (microcracking) and with the number of
cycles N. We model:
    k(T, N) = k0 * exp(alpha*(T - T0)) * (1 + beta*log(1 + N))
"""
import numpy as np


def permeability_thermal(T_peak, N_cycles, k0=1e-19, T0=25.0,
                         alpha=0.012, beta=0.25):
    T_peak = np.asarray(T_peak, dtype=float)
    N_cycles = np.asarray(N_cycles, dtype=float)
    return k0 * np.exp(alpha * (T_peak - T0)) * (1.0 + beta * np.log1p(N_cycles))


def fit_thermal_model(T_peak, N_cycles, k_obs):
    """Linearize: ln(k) = ln(k0) + alpha*(T-T0) + ln(1 + beta*ln(1+N)).
    For simplicity treat the log term separately via two-step regression."""
    from numpy.linalg import lstsq
    T_peak = np.asarray(T_peak); N_cycles = np.asarray(N_cycles)
    y = np.log(k_obs)
    X = np.column_stack([np.ones_like(T_peak), T_peak - 25.0, np.log1p(N_cycles)])
    coeffs, *_ = lstsq(X, y, rcond=None)
    ln_k0, alpha, gamma = coeffs
    # gamma ~ beta when beta*ln(1+N) << 1; first-order approx
    return {"k0": float(np.exp(ln_k0)), "alpha": float(alpha), "beta_approx": float(gamma)}


def test_all():
    Ts = np.array([25, 100, 200, 400, 600], dtype=float)
    Ns = np.array([0, 1, 3, 5, 10], dtype=float)
    TT, NN = np.meshgrid(Ts, Ns)
    k = permeability_thermal(TT, NN)
    # k must increase along both axes
    assert np.all(np.diff(k, axis=1) > 0), "k should grow with T"
    assert np.all(np.diff(k, axis=0) > 0), "k should grow with N"
    fit = fit_thermal_model(TT.ravel(), NN.ravel(), k.ravel())
    assert abs(fit["alpha"] - 0.012) < 1e-3
    print("article5 OK | k(600C, N=10) =", f"{k[-1, -1]:.2e}", "| fit:", fit)


if __name__ == "__main__":
    test_all()
