"""
Article 6: Enhancing the Detectability of Deep-Sensing Borehole EM
Instruments by Joint Inversion of Multiple Logs Within a Probabilistic
Geosteering Workflow
Jahani, Alyaev, Ambia, Fossum, Suter, Torres-Verdín (2023)
DOI: 10.30632/PJV64N1-2023a6

Implements an LM-EnRML (Levenberg-Marquardt Ensemble Randomised Maximum
Likelihood) inversion for a layered geosteering scenario.  The state
vector m holds per-layer (porosity, water saturation) plus the two
bed-boundary positions; observations d are stacked shallow-propagation
resistivity, extra-deep symmetric EM, and nuclear bulk-density readings.

Forward operators are toy depth-of-investigation kernels:
  - shallow:  Gaussian kernel of std 0.9 m at the bit
  - extra-deep:  Gaussian kernel of std 14.9 m, evaluated 10 m ahead
  - nuclear density:  Gaussian kernel of std 0.4 m

Per-layer resistivity comes from Archie:  Rt = a Rw / (phi^m Sw^n).
Per-layer bulk density from a clay-quartz-fluid mixture.

The implementation follows the paper's Appendix A1 update:

    m_{n+1} = m_n - C_x G_n^T (G_n C_x G_n^T + lambda C_d)^{-1} (d_pred - d_obs)

with G_n estimated empirically from the ensemble and an LM damping
schedule that increases lambda after a rejected step.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ------------------------------------------------ forward operators ---------

def archie_rt(phi, sw, Rw=0.05, m=2.0, n=2.0, a=1.0):
    """Archie's law for true resistivity."""
    # NOTE the historical argument order (phi, sw); canonical is (sw, rw, phi=).
    return petrolib.saturation_resistivity.archie_rt(sw, Rw, phi=phi, a=a, m=m, n=n)


def bulk_density(phi, sw, rho_q=2.65, rho_w=1.0, rho_h=0.8):
    """Two-mineral + two-fluid bulk density (g/cc) for a shaly-sand-style mix."""
    rho_f = sw * rho_w + (1.0 - sw) * rho_h
    return (1.0 - phi) * rho_q + phi * rho_f


def gaussian_kernel(depth_axis, centre, sigma):
    k = np.exp(-0.5 * ((depth_axis - centre) / sigma) ** 2)
    return k / k.sum()


def layered_profile(z, boundaries, layer_values):
    """Step-wise property profile from a list of (boundary, value) pairs."""
    out = np.empty_like(z, dtype=float)
    idx = np.searchsorted(boundaries, z, side="right")
    idx = np.clip(idx, 0, len(layer_values) - 1)
    return np.array(layer_values)[idx]


def forward(model, z, bit_depth, depth_ahead=10.0):
    """Return predicted (R_shallow, R_deep, rho) from the geosteering state.

    `model` is a dict with keys
        phi_layers (n+1 floats), sw_layers (n+1 floats), boundaries (n floats)
    """
    phi = layered_profile(z, model["boundaries"], model["phi_layers"])
    sw = layered_profile(z, model["boundaries"], model["sw_layers"])
    Rt = archie_rt(phi, sw)
    rho = bulk_density(phi, sw)

    k_sh = gaussian_kernel(z, bit_depth, 0.9)
    k_dp = gaussian_kernel(z, bit_depth + depth_ahead, 14.9)
    k_nu = gaussian_kernel(z, bit_depth, 0.4)
    R_sh = 1.0 / (k_sh * (1.0 / Rt)).sum()
    R_dp = 1.0 / (k_dp * (1.0 / Rt)).sum()
    rho_b = (k_nu * rho).sum()
    return np.array([R_sh, R_dp, rho_b])


# ------------------------------------------------ LM-EnRML inversion --------

def lm_enrml(prior_mean, prior_cov, obs, obs_cov, forward_fn,
             n_ens=80, n_iter=12, lam0=1.0, lam_up=4.0, lam_dn=0.4,
             seed=0):
    """Approximate LM-EnRML (Chen & Oliver 2013 / Appendix A1 of the paper).

    Returns the final ensemble (n_ens, n_dim).
    """
    rng = np.random.default_rng(seed)
    n_dim = len(prior_mean)
    n_obs = len(obs)
    L_x = np.linalg.cholesky(prior_cov + 1e-9 * np.eye(n_dim))
    L_d = np.linalg.cholesky(obs_cov + 1e-9 * np.eye(n_obs))

    # Initial ensemble drawn from the prior
    ens = prior_mean + rng.standard_normal((n_ens, n_dim)) @ L_x.T
    lam = lam0

    def avg_misfit(ens_now):
        m = 0.0
        for m_i in ens_now:
            d_i = forward_fn(m_i)
            r = np.linalg.solve(L_d, d_i - obs)
            m += float(r @ r)
        return m / len(ens_now)

    cur_mis = avg_misfit(ens)
    for it in range(n_iter):
        # Empirical sensitivity G = dD / dM (centred on ensemble)
        D = np.array([forward_fn(m_i) for m_i in ens])
        M_c = ens - ens.mean(0, keepdims=True)
        D_c = D - D.mean(0, keepdims=True)
        # Pseudo-inverse of M_c via SVD with regularisation
        U, s, Vt = np.linalg.svd(M_c, full_matrices=False)
        s_inv = s / (s ** 2 + 1e-8)
        G = (D_c.T @ U) * s_inv @ Vt          # (n_obs, n_dim)

        # Step
        Cd = obs_cov
        S = G @ prior_cov @ G.T + lam * Cd
        K = prior_cov @ G.T @ np.linalg.inv(S)
        d_perturbed = obs + rng.standard_normal((n_ens, n_obs)) @ L_d.T
        proposed = ens - (D - d_perturbed) @ K.T

        new_mis = avg_misfit(proposed)
        if new_mis < cur_mis:
            ens = proposed
            cur_mis = new_mis
            lam *= lam_dn
        else:
            lam *= lam_up
        if lam > 1e6:
            break
    return ens


# ----------------------------------------------------- tests ----------------

def test_all():
    print("=" * 60)
    print("Article 6: LM-EnRML Geosteering Inversion")
    print("=" * 60)

    # Synthetic three-layer earth around the bit at depth 50 m.
    z = np.linspace(20.0, 80.0, 121)
    bit = 50.0
    true_model = {
        "phi_layers": [0.05, 0.22, 0.08],
        "sw_layers":  [0.60, 0.30, 0.55],
        "boundaries": [44.0, 56.0],
    }
    obs = forward(true_model, z, bit)
    obs_noisy = obs * (1.0 + 0.02 * np.random.default_rng(1).standard_normal(3))
    print(f"  True forward (R_sh, R_dp, rho_b) = "
          f"{obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.3f}")

    # State vector: [phi_top, phi_mid, phi_bot, sw_top, sw_mid, sw_bot, b1, b2]
    prior_mean = np.array([0.08, 0.18, 0.10, 0.55, 0.40, 0.55, 42.0, 58.0])
    prior_var = np.array([0.04, 0.06, 0.04, 0.20, 0.20, 0.20, 6.0, 6.0]) ** 2
    prior_cov = np.diag(prior_var)
    obs_cov = np.diag((np.maximum(obs_noisy * 0.05, 0.02)) ** 2)

    def wrap(m_vec):
        return forward({"phi_layers": list(np.clip(m_vec[:3], 0.01, 0.40)),
                        "sw_layers":  list(np.clip(m_vec[3:6], 0.05, 1.00)),
                        "boundaries": sorted([m_vec[6], m_vec[7]])},
                       z, bit)

    ens0 = prior_mean + np.random.default_rng(2).standard_normal((40, 8)) \
                       * np.sqrt(prior_var)
    prior_misfit = float(np.mean([
        ((wrap(m) - obs_noisy) ** 2 / np.diag(obs_cov)).sum() for m in ens0
    ]))

    ens = lm_enrml(prior_mean, prior_cov, obs_noisy, obs_cov, wrap)
    post_misfit = float(np.mean([
        ((wrap(m) - obs_noisy) ** 2 / np.diag(obs_cov)).sum() for m in ens
    ]))

    m_mean = ens.mean(0)
    print(f"  Prior  reduced misfit chi^2/N = {prior_misfit:.2f}")
    print(f"  Posterior reduced misfit      = {post_misfit:.2f}")
    print(f"  Posterior mean phi_layers = "
          f"{m_mean[0]:.3f}, {m_mean[1]:.3f}, {m_mean[2]:.3f}   "
          f"(true {true_model['phi_layers']})")
    print(f"  Posterior mean boundaries = {sorted([m_mean[6], m_mean[7]])}   "
          f"(true {true_model['boundaries']})")

    assert post_misfit < 0.5 * prior_misfit, \
        "LM-EnRML should at least halve the misfit"
    print("  PASS")
    return {"prior_chi2": prior_misfit, "post_chi2": post_misfit}


if __name__ == "__main__":
    test_all()
