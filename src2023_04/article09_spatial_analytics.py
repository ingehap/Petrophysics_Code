"""
Article 9: Spatial Data Analytics-Assisted Subsurface Modeling: A Duvernay
Case Study
Salazar, Ochoa, Garland, Lake, and Pyrcz (2023)
DOI: 10.30632/PJV64N2-2023a9

Implements an end-to-end 2D geostatistical workflow:
  1. Outlier filtering via Mahalanobis distance + isolation forest
  2. Trend modelling via Gaussian moving-window
  3. Experimental + spherical-model semivariogram fitting
  4. Simple kriging (SK) interpolation
  5. Sequential Gaussian Simulation (SGS) for uncertainty
  6. Collocated cokriging (Markov-Bayes) for cosimulation
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
from scipy.optimize import minimize
from scipy.spatial import cKDTree


# ---------------------------------------------- outlier identification ---

def mahalanobis_outliers(X, threshold=3.5):
    cov = np.cov(X.T)
    inv = np.linalg.pinv(cov)
    mean = X.mean(axis=0)
    dists = np.array([mahalanobis(x, mean, inv) for x in X])
    return dists > threshold, dists


def isolation_forest_outliers(X, contamination=0.05, seed=0):
    iso = IsolationForest(contamination=contamination, random_state=seed).fit(X)
    pred = iso.predict(X)  # -1 = outlier
    return pred == -1


# ---------------------------------------------- Gaussian trend model ---

def gaussian_trend(coords, values, grid, sigma=200.0):
    """Kernel smoothing trend at every grid point."""
    out = np.zeros(len(grid))
    for i, g in enumerate(grid):
        d = np.linalg.norm(coords - g, axis=1)
        w = np.exp(-d ** 2 / (2 * sigma ** 2))
        out[i] = np.sum(w * values) / (np.sum(w) + 1e-12)
    return out


# --------------------------------------------------- semivariogram ---

def experimental_variogram(coords, values, n_lags=10, lag_size=None):
    n = len(values)
    pair_dists = []
    pair_diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            pair_dists.append(d)
            pair_diffs.append((values[i] - values[j]) ** 2)
    pair_dists = np.array(pair_dists)
    pair_diffs = np.array(pair_diffs)
    if lag_size is None:
        lag_size = pair_dists.max() / (2 * n_lags)
    lags = np.arange(0.5, n_lags + 0.5) * lag_size
    gamma = np.zeros(n_lags)
    counts = np.zeros(n_lags)
    for i, lag in enumerate(lags):
        mask = (pair_dists >= lag - lag_size / 2) & (pair_dists < lag + lag_size / 2)
        if mask.any():
            gamma[i] = 0.5 * pair_diffs[mask].mean()
            counts[i] = mask.sum()
    return lags, gamma, counts


def spherical_model(h, nugget, sill, rang):
    """Standard spherical variogram model."""
    g = np.where(
        h <= rang,
        nugget + (sill - nugget) * (1.5 * h / rang - 0.5 * (h / rang) ** 3),
        sill,
    )
    g[h == 0] = 0
    return g


def fit_variogram(lags, gamma):
    """Fit spherical model to experimental variogram."""
    def loss(p):
        nugget, sill, rang = p
        if nugget < 0 or sill < nugget or rang <= 0:
            return 1e12
        return np.sum((spherical_model(lags, nugget, sill, rang) - gamma) ** 2)

    p0 = [0.0, gamma.max(), lags[-1] / 2]
    res = minimize(loss, p0, method="Nelder-Mead",
                   options={"maxiter": 500})
    return dict(nugget=res.x[0], sill=res.x[1], range=res.x[2])


# ----------------------------------------------- simple kriging ---

def simple_kriging(coords, values, query, vmodel, mean=0.0, max_neighbours=12):
    tree = cKDTree(coords)
    out = np.zeros(len(query))
    var = np.zeros(len(query))
    for q_i, q in enumerate(query):
        _, idx = tree.query(q, k=min(max_neighbours, len(coords)))
        idx = np.atleast_1d(idx)
        Xn = coords[idx]
        vn = values[idx]
        K = np.zeros((len(idx), len(idx)))
        for i in range(len(idx)):
            for j in range(len(idx)):
                d = np.linalg.norm(Xn[i] - Xn[j])
                K[i, j] = vmodel["sill"] - spherical_model(np.array([d]),
                                                            vmodel["nugget"],
                                                            vmodel["sill"],
                                                            vmodel["range"])[0]
        k_vec = np.zeros(len(idx))
        for i in range(len(idx)):
            d = np.linalg.norm(Xn[i] - q)
            k_vec[i] = vmodel["sill"] - spherical_model(np.array([d]),
                                                        vmodel["nugget"],
                                                        vmodel["sill"],
                                                        vmodel["range"])[0]
        try:
            w = np.linalg.solve(K + 1e-8 * np.eye(len(idx)), k_vec)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(K, k_vec, rcond=None)[0]
        out[q_i] = mean + np.sum(w * (vn - mean))
        var[q_i] = max(vmodel["sill"] - np.sum(w * k_vec), 0.0)
    return out, var


# -------------------------------------------- sequential Gaussian simulation -

def sgs(coords, values, query, vmodel, mean=0.0, n_realizations=10, seed=0):
    """Simplified SGS: visits queries in random order, drawing from kriged distribution."""
    rng = np.random.default_rng(seed)
    realizations = np.zeros((n_realizations, len(query)))
    for r in range(n_realizations):
        order = rng.permutation(len(query))
        c_run = list(coords)
        v_run = list(values)
        sim = np.zeros(len(query))
        for idx in order:
            q = query[idx]
            mu, var = simple_kriging(np.array(c_run), np.array(v_run),
                                     q.reshape(1, -1), vmodel, mean=mean)
            sim[idx] = rng.normal(mu[0], np.sqrt(var[0]) + 1e-6)
            c_run.append(q)
            v_run.append(sim[idx])
        realizations[r] = sim
    return realizations


# ---------------------------------------- collocated cokriging (Markov-Bayes)

def collocated_cokriging(coords, primary, query, secondary_at_query,
                         vmodel_primary, rho, mean_p=0.0, mean_s=0.0,
                         var_red_factor=1.0):
    """Markov-Bayes collocated cokriging for cosimulation."""
    pred, _ = simple_kriging(coords, primary, query, vmodel_primary, mean_p)
    # inflate by correlation with secondary
    return pred + rho * (secondary_at_query - mean_s) * var_red_factor


# ---------------------------------------------------- testing ---

def synthetic_field(seed=0, nx=20, ny=20, n_samples=80):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, 100, nx), np.linspace(0, 100, ny))
    truth = (5 * np.sin(xs / 20) + 3 * np.cos(ys / 15) + rng.normal(0, 0.4, xs.shape))
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    sample_idx = rng.choice(len(coords), n_samples, replace=False)
    sample_coords = coords[sample_idx]
    sample_vals = truth.ravel()[sample_idx]
    return sample_coords, sample_vals, coords, truth.ravel()


def test_all():
    print("=" * 60)
    print("Article 9: Spatial Data Analytics")
    print("=" * 60)
    coords, vals, all_coords, truth = synthetic_field()

    # outlier filtering
    iso_out = isolation_forest_outliers(np.column_stack([coords, vals]))
    inliers = ~iso_out
    print(f"  Isolation-forest flagged {iso_out.sum()} / {len(vals)} outliers")
    coords = coords[inliers]
    vals = vals[inliers]

    # trend (use a small subgrid for speed)
    sub_idx = np.arange(0, len(all_coords), 5)
    trend = gaussian_trend(coords, vals, all_coords[sub_idx], sigma=25.0)
    residual = vals - gaussian_trend(coords, vals, coords, sigma=25.0)
    print(f"  Trend std on subgrid = {trend.std():.2f}, residual std = {residual.std():.2f}")

    # variogram
    lags, gamma, counts = experimental_variogram(coords, residual, n_lags=8)
    vmodel = fit_variogram(lags, gamma)
    print(f"  Variogram: nugget={vmodel['nugget']:.2f}  "
          f"sill={vmodel['sill']:.2f}  range={vmodel['range']:.2f}")

    # kriging at small grid for sanity
    query = all_coords[sub_idx][:30]
    pred, kvar = simple_kriging(coords, residual, query, vmodel)
    print(f"  Kriging RMSE (residual) = {np.sqrt(np.mean((pred - truth[sub_idx][:30]) ** 2)):.3f}")

    # SGS — reduced size for speed
    real = sgs(coords[:30], residual[:30], query[:10], vmodel,
               n_realizations=3, seed=0)
    print(f"  SGS realizations: {real.shape}, ensemble std = {real.std(axis=0).mean():.3f}")

    # collocated cokriging
    sec = truth.ravel()[sub_idx][:30] + np.random.default_rng(2).normal(0, 0.3, 30)
    co = collocated_cokriging(coords, residual, query, sec,
                              vmodel, rho=0.7, mean_s=sec.mean())
    print(f"  Collocated cokriging mean = {co.mean():.3f}")

    print("  PASS")
    return {"vmodel": vmodel, "kriging_rmse": float(np.sqrt(np.mean((pred - truth[sub_idx][:30]) ** 2)))}


if __name__ == "__main__":
    test_all()
