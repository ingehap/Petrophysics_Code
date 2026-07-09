"""
Article 2: Unlocking the Full Potential of NMR Using Machine Learning -
A Case Study of a Gas Field With an Oil Problem
Cuddy (2022)
DOI: 10.30632/PJV63N3-2022a2

Seven Heads gas field, Celtic Sea - thin-bedded sands carrying high-
viscosity residual oil that must be avoided at perforation.  The paper
combines a Dual-Echo-Time (DTE) NMR-while-drilling acquisition with an
ML stack:

  - Fuzzy logic (FL) for grayscale lithotype probability
  - Genetic algorithms (GA) to evolve a continuous functional map from
    logs to T2
  - k-nearest-neighbours (kNN) classification calibrated to Dean-Stark
    cores and verified by MDT-LFA

This module implements:

  - DTE diffusion-decay model T2_eff(D, TE) used to push WBM filtrate
    to short T2.
  - Triangular-membership fuzzy classifier for the (gas / oil / water)
    saturation problem.
  - Lightweight GA that evolves polynomial coefficients mapping a
    log-feature vector to predicted saturation.
  - kNN regressor used as the benchmark of the ML chain.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- DTE diffusion (Table 1) ---

def t2_effective_dte(T2_intrinsic, D_m2_s, gamma_rad_s_T=2.675e8,
                    G_T_m=20.0, TE_s=1.2e-3):
    """Effective T2 from CPMG with field gradient G and echo spacing TE.

        1/T2_eff = 1/T2_int + (1/12) * (gamma G TE)^2 * D
    """
    return petrolib.nmr.t2_apparent(
        t2_bulk=T2_intrinsic, D=D_m2_s, G=G_T_m, TE=TE_s, gamma=gamma_rad_s_T)


# ---------------------------------------------- fuzzy classifier ---------

def triangular(x, lo, mid, hi):
    """Triangular membership function."""
    if x <= lo or x >= hi:
        return 0.0
    if x <= mid:
        return (x - lo) / (mid - lo)
    return (hi - x) / (hi - mid)


def fuzzy_fluid(t2_log_ms, density_g_cm3, resistivity_Ohm_m):
    """Returns three fuzzy memberships (gas, oil, water) in [0, 1]."""
    # T2 fuzzy bins (ms): gas long T2, oil intermediate, water variable
    mu_T2_gas = triangular(t2_log_ms, 30, 200, 1000)
    mu_T2_oil = triangular(t2_log_ms, 3, 30, 300)
    mu_T2_water = triangular(t2_log_ms, 0.5, 15, 300)
    # Density fuzzy: gas low, oil mid, water high
    mu_d_gas = triangular(density_g_cm3, 0.0, 0.2, 0.6)
    mu_d_oil = triangular(density_g_cm3, 0.6, 0.8, 0.95)
    mu_d_water = triangular(density_g_cm3, 0.9, 1.0, 1.15)
    # Resistivity fuzzy
    mu_r_gas = triangular(np.log10(resistivity_Ohm_m), 1.5, 2.5, 3.5)
    mu_r_oil = triangular(np.log10(resistivity_Ohm_m), 1.0, 2.0, 3.0)
    mu_r_water = triangular(np.log10(resistivity_Ohm_m), -1.0, 0.5, 1.5)
    # AND-of-features then normalise
    g = mu_T2_gas * mu_d_gas * mu_r_gas
    o = mu_T2_oil * mu_d_oil * mu_r_oil
    w = mu_T2_water * mu_d_water * mu_r_water
    s = g + o + w + 1e-12
    return g / s, o / s, w / s


# ---------------------------------------------- GA functional map -------

def fitness(coeffs, X, y):
    """MSE of the polynomial mapping X @ coeffs - y."""
    pred = X @ coeffs
    return float(np.mean((pred - y) ** 2))


def genetic_polynomial_fit(X, y, n_pop=80, n_gen=200, mut=0.10, seed=0):
    """Real-valued GA: tournament selection, blended crossover, Gaussian mutation."""
    rng = np.random.default_rng(seed)
    n_d = X.shape[1]
    pop = rng.standard_normal((n_pop, n_d))
    for gen in range(n_gen):
        fits = np.array([fitness(p, X, y) for p in pop])
        order = np.argsort(fits)
        survivors = pop[order[: n_pop // 2]]
        # Children via blended crossover
        children = []
        for _ in range(n_pop // 2):
            a, b = survivors[rng.integers(0, len(survivors))], \
                   survivors[rng.integers(0, len(survivors))]
            alpha = rng.random()
            children.append(alpha * a + (1 - alpha) * b)
        children = np.array(children)
        # Mutation
        children += mut * rng.standard_normal(children.shape)
        pop = np.vstack([survivors, children])
    fits = np.array([fitness(p, X, y) for p in pop])
    return pop[np.argmin(fits)], float(fits.min())


# ---------------------------------------------- kNN regressor ---------

def knn_predict(X_train, y_train, X_query, k=5):
    """Distance-weighted k-NN regressor."""
    out = np.zeros(len(X_query))
    for i, x in enumerate(X_query):
        d = np.linalg.norm(X_train - x, axis=1)
        idx = np.argsort(d)[:k]
        w = 1.0 / (d[idx] + 1e-9)
        out[i] = (w * y_train[idx]).sum() / w.sum()
    return out


# ---------------------------------------------- synthetic dataset -----

def make_seven_heads_dataset(n=400, seed=0):
    """Synthetic (T2, RHOB, Rt) feature triples with continuous gas saturation y."""
    rng = np.random.default_rng(seed)
    T2 = 10.0 ** rng.uniform(0.0, 3.0, n)
    rhob = rng.uniform(0.1, 1.10, n)
    rt = 10.0 ** rng.uniform(0.0, 3.0, n)
    # Ground-truth gas saturation: high gas <-> low rhob and long T2
    y = np.clip((np.log10(T2) / 3.0) * (1.0 - rhob / 1.10), 0.0, 1.0)
    y += 0.05 * rng.standard_normal(n)
    return np.c_[np.log10(T2), rhob, np.log10(rt)], np.clip(y, 0.0, 1.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Fuzzy + GA + kNN NMR-WD Fluid Typing")
    print("=" * 60)

    # DTE diffusion check
    t2_short = t2_effective_dte(T2_intrinsic=500e-3, D_m2_s=2.5e-9, TE_s=1.2e-3)
    t2_long = t2_effective_dte(T2_intrinsic=500e-3, D_m2_s=2.5e-9, TE_s=4.8e-3)
    print(f"  Water T2_eff at TE=1.2 ms = {t2_short * 1e3:7.2f} ms")
    print(f"  Water T2_eff at TE=4.8 ms = {t2_long * 1e3:7.2f} ms (should be << short)")
    assert t2_long < t2_short, "Longer TE must shorten effective T2 (diffusion)"

    # Fuzzy classifier sanity
    g, o, w = fuzzy_fluid(t2_log_ms=300.0, density_g_cm3=0.20,
                          resistivity_Ohm_m=200.0)
    print(f"  Fuzzy (gas, oil, water) for gas-like sample = "
          f"({g:.2f}, {o:.2f}, {w:.2f})")
    assert g > o and g > w, "Gas-like sample must be classified as gas"

    # Train / test split
    X, y = make_seven_heads_dataset()
    n_train = int(0.7 * len(y))
    idx = np.random.default_rng(0).permutation(len(y))
    Xt, yt, Xe, ye = X[idx[:n_train]], y[idx[:n_train]], \
                     X[idx[n_train:]], y[idx[n_train:]]

    # GA polynomial fit
    coeffs, mse = genetic_polynomial_fit(Xt, yt, n_pop=60, n_gen=100)
    y_ga = Xe @ coeffs
    rmse_ga = float(np.sqrt(((y_ga - ye) ** 2).mean()))

    # kNN baseline
    y_knn = knn_predict(Xt, yt, Xe, k=7)
    rmse_knn = float(np.sqrt(((y_knn - ye) ** 2).mean()))

    print(f"  GA  3-coef polynomial    test RMSE = {rmse_ga:.3f}")
    print(f"  k-NN (k=7)               test RMSE = {rmse_knn:.3f}")
    assert rmse_knn < 0.20 and rmse_ga < 0.30
    print("  PASS")
    return {"rmse_ga": rmse_ga, "rmse_knn": rmse_knn,
            "ga_coeffs": list(coeffs)}


if __name__ == "__main__":
    test_all()
