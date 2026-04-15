"""
Article 1: Unsupervised Electrofacies Clustering Based on Parameterization
of Petrophysical Properties: A Dynamic Programming Approach
Sinnathamby, Hou, Gkortsas, Venkataramanan, Datir, Kollien, and Fleuret (2023)
DOI: 10.30632/PJV64N2-2023a1

Implements a DP-based clustering algorithm that respects:
  - max number of clusters C
  - max number of transitions N
  - minimal block size MinPhi
The cluster characterization fits a parameterized physical model
(Waxman-Smits resistivity equation by default) per cluster.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------- physics ---

def waxman_smits_conductivity(phi_t, sw, qv, m_star, n_star, sigma_w, temp_c):
    """Waxman-Smits conductivity (Eq. 13 of the paper)."""
    B = (-1.28 + 0.225 * temp_c - 0.0004059 * temp_c ** 2) / (
        1.0 + sigma_w ** -1.23 * (0.045 * temp_c - 0.27)
    )
    sigma_o = (phi_t ** m_star) * (sw ** n_star) * (sigma_w + B * qv / sw)
    return sigma_o


def fit_waxman_smits(phi_t, sw, sigma_xo, fclay, temp_c):
    """Fit (m*, n*, sigma_w, qv_scale) to a cluster of measurements via least squares."""
    from scipy.optimize import minimize

    qv = fclay * 0.1  # simple proxy

    def loss(params):
        m, n, sw_cond = params
        if m <= 0 or n <= 0 or sw_cond <= 0:
            return 1e10
        pred = waxman_smits_conductivity(phi_t, sw, qv, m, n, sw_cond, temp_c)
        return np.mean((np.log(pred + 1e-10) - np.log(sigma_xo + 1e-10)) ** 2)

    res = minimize(loss, x0=[2.0, 2.0, 5.0], method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 1e-3})
    return np.array([res.x[0], res.x[1], res.x[2]])


def ws_cost(point, weights, temp_c):
    """Cost of a single point under WS model with given weights."""
    phi_t, sw, sigma_xo, fclay = point
    m, n, sigma_w = weights
    qv = fclay * 0.1
    pred = waxman_smits_conductivity(phi_t, sw, qv, m, n, sigma_w, temp_c)
    return (np.log(pred + 1e-10) - np.log(sigma_xo + 1e-10)) ** 2


# ---------------------------------------------------- generic DP clustering --

def random_assignment(C, T, N, min_phi, rng):
    """Randomly assign cluster labels of length T using <= N transitions and min block size."""
    n_blocks = rng.integers(1, N + 2)  # number of blocks in [1, N+1]
    if n_blocks * min_phi > T:
        n_blocks = max(1, T // min_phi)
    # split T points into n_blocks blocks each >= min_phi
    extra = T - n_blocks * min_phi
    block_sizes = [min_phi] * n_blocks
    for _ in range(extra):
        block_sizes[rng.integers(0, n_blocks)] += 1
    labels = []
    used = list(range(C))
    rng.shuffle(used)
    last = -1
    for i, sz in enumerate(block_sizes):
        choices = [c for c in range(C) if c != last]
        c = choices[rng.integers(0, len(choices))]
        labels.extend([c] * sz)
        last = c
    return np.array(labels[:T])


def dp_path_finder(weights, X, C, N, min_phi, cost_fn):
    """
    Dynamic-programming optimal cluster assignment.
    weights: array shape (C, dw) — current cluster characterization
    X: array shape (T, d) — multivariate series
    Returns: optimal labels of length T, total cost.
    """
    T = X.shape[0]
    INF = 1e15
    # omega[t, n, c] = optimal cumulative cost up to step t, n transitions used, point t in cluster c
    omega = np.full((T, N + 1, C), INF)
    parent = np.full((T, N + 1, C, 2), -1, dtype=int)  # (prev_n, prev_c)

    # initialization: first min_phi points must be in same cluster, n=0
    for c in range(C):
        s = sum(cost_fn(X[i], weights[c]) for i in range(min_phi))
        omega[min_phi - 1, 0, c] = s

    # fill the table
    for t in range(min_phi, T):
        for c in range(C):
            cost_pt = cost_fn(X[t], weights[c])
            for n in range(N + 1):
                # option A: continue the same cluster (no new transition)
                if omega[t - 1, n, c] < INF:
                    val = omega[t - 1, n, c] + cost_pt
                    if val < omega[t, n, c]:
                        omega[t, n, c] = val
                        parent[t, n, c] = (n, c)
                # option B: a transition occurred at t - min_phi + 1
                if n >= 1 and t >= 2 * min_phi - 1:
                    block_cost = sum(cost_fn(X[t - k], weights[c]) for k in range(min_phi))
                    for cp in range(C):
                        if cp == c:
                            continue
                        prev = omega[t - min_phi, n - 1, cp]
                        if prev < INF:
                            val = prev + block_cost
                            if val < omega[t, n, c]:
                                omega[t, n, c] = val
                                parent[t, n, c] = (n - 1, cp)

    # backtrack
    final = omega[T - 1]
    n_opt, c_opt = np.unravel_index(np.argmin(final), final.shape)
    best_cost = final[n_opt, c_opt]
    labels = np.empty(T, dtype=int)
    labels[T - 1] = c_opt
    t = T - 1
    n, c = n_opt, c_opt
    while t > 0:
        pn, pc = parent[t, n, c]
        if pn == -1:
            # initialization region
            labels[:t] = c
            break
        if pc != c:
            # transition: previous min_phi-1 points share cluster c
            for k in range(1, min_phi):
                labels[t - k] = c
            t -= min_phi
            n, c = pn, pc
            labels[t] = c
        else:
            t -= 1
            labels[t] = c
            n = pn
    return labels, best_cost


def cluster_characterize(X, labels, C, fit_fn, dw):
    """Fit per-cluster weights using the provided fit_fn."""
    weights = np.zeros((C, dw))
    for c in range(C):
        mask = labels == c
        if mask.sum() < 2:
            weights[c] = np.array([2.0, 2.0, 5.0])[:dw]
            continue
        phi = X[mask, 0]
        sw = X[mask, 1]
        sig = X[mask, 2]
        fcl = X[mask, 3]
        weights[c] = fit_fn(phi, sw, sig, fcl, 80.0)
    return weights


def dp_cluster(X, C=3, N=2, min_phi=3, n_init=5, max_iter=10, seed=0,
               fit_fn=fit_waxman_smits, cost_fn=None, dw=3, temp_c=80.0):
    """
    Run the DP electrofacies algorithm with random initializations.
    Returns the most common assignment among all inits using ARI.
    """
    if cost_fn is None:
        def cost_fn(pt, w):
            return ws_cost(pt, w, temp_c)

    rng = np.random.default_rng(seed)
    T = X.shape[0]
    all_labels = []
    all_costs = []
    for init in range(n_init):
        labels = random_assignment(C, T, N, min_phi, rng)
        prev_cost = np.inf
        for _ in range(max_iter):
            weights = cluster_characterize(X, labels, C, fit_fn, dw)
            labels, cost = dp_path_finder(weights, X, C, N, min_phi, cost_fn)
            if abs(prev_cost - cost) < 1e-4:
                break
            prev_cost = cost
        all_labels.append(labels)
        all_costs.append(cost)
    # pick the labeling with the highest mean ARI to the others
    best_idx = 0
    best_mean = -2
    for i in range(n_init):
        m = np.mean([adjusted_rand_score(all_labels[i], all_labels[j])
                     for j in range(n_init) if j != i] or [1.0])
        if m > best_mean:
            best_mean = m
            best_idx = i
    return all_labels[best_idx], all_costs[best_idx]


# ---------------------------------------------------------------- testing ---

def synthetic_data(seed=42):
    """Generate a 3-cluster synthetic series with WS-consistent responses."""
    rng = np.random.default_rng(seed)
    T = 60
    labels_true = np.concatenate([
        np.zeros(20, dtype=int),
        np.ones(20, dtype=int),
        np.full(20, 2, dtype=int),
    ])
    cluster_params = [
        dict(m=1.8, n=1.9, sigma_w=4.0),
        dict(m=2.2, n=2.0, sigma_w=6.0),
        dict(m=2.0, n=2.1, sigma_w=2.0),
    ]
    X = np.zeros((T, 4))
    for t in range(T):
        c = labels_true[t]
        p = cluster_params[c]
        phi = rng.uniform(0.10, 0.30)
        sw = rng.uniform(0.30, 0.95)
        fclay = rng.uniform(0.0, 0.3)
        sig = waxman_smits_conductivity(
            phi, sw, fclay * 0.1, p["m"], p["n"], p["sigma_w"], 80.0
        )
        sig *= rng.lognormal(0.0, 0.05)  # noise
        X[t] = [phi, sw, sig, fclay]
    return X, labels_true


def test_all():
    print("=" * 60)
    print("Article 1: DP Electrofacies Clustering")
    print("=" * 60)
    X, labels_true = synthetic_data()
    labels, cost = dp_cluster(X, C=3, N=2, min_phi=4, n_init=3, max_iter=5)
    ari = adjusted_rand_score(labels_true, labels)
    print(f"  Total cost  = {cost:.4f}")
    print(f"  ARI vs true = {ari:.3f}  (1.0 = perfect)")
    print(f"  Predicted blocks: {np.unique(labels, return_counts=True)}")
    assert ari > 0.3, "ARI too low — clustering failed"
    print("  PASS")
    return {"ari": ari, "cost": cost}


if __name__ == "__main__":
    test_all()
