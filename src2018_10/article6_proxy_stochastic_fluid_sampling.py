"""
Article 6: Proxy-Enabled Stochastic Interpretation of Downhole Fluid Sampling
           Under Immiscible Flow Conditions
Kristensen, Chugunov, Cig, Jackson (2018)
DOI: 10.30632/PJV59N5-2018a5

During downhole fluid sampling, oil-based-mud filtrate contamination decays as
the tool pumps; a fast proxy model of the cleanup, combined with stochastic
(Monte Carlo / Bayesian) sampling, interprets the noisy live-contamination data
and quantifies the uncertainty on the cleanup parameters and the pumpout volume
needed to reach a target contamination.

Implements:

  - Power-law contamination cleanup  eta(V) = eta0*(1 + V/V*)^(-5/12)
  - Volume to reach a target contamination
  - Bayesian (Monte Carlo) posterior on the cleanup parameters from noisy data
  - Posterior predictive uncertainty on the pumpout volume

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the cleanup proxy + stochastic-interpretation method the paper applies.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

EXPONENT = 5.0 / 12.0


# ---------------------------------------------- cleanup proxy -----------

def cleanup(eta0, v_star, volume, exponent=EXPONENT):
    """Contamination vs pumped volume  eta(V) = eta0*(1 + V/V*)^(-5/12)."""
    return petrolib.geochem_fluids.contamination.cleanup_powerlaw(volume, eta0, v_star, exponent=exponent)


def volume_to_target(eta0, v_star, eta_target, exponent=EXPONENT):
    """Pumped volume required to reach a target contamination."""
    return petrolib.geochem_fluids.contamination.volume_to_target(eta0, v_star, eta_target, exponent=exponent)


# ---------------------------------------------- stochastic --------------

def mc_posterior(volumes, eta_obs, noise, n_samples=4000, seed=0):
    """Monte-Carlo posterior over (eta0, V*) from noisy contamination data.

    Importance/rejection-free Metropolis sampler; returns arrays of accepted
    (eta0, V*).
    """
    rng = np.random.default_rng(seed)
    volumes = np.asarray(volumes, float); eta_obs = np.asarray(eta_obs, float)

    def loglik(eta0, vstar):
        pred = cleanup(eta0, vstar, volumes)
        return -0.5 * np.sum(((pred - eta_obs) / noise) ** 2)

    eta0, vstar = 0.5, 10.0
    ll = loglik(eta0, vstar)
    samples = []
    for _ in range(n_samples):
        e2 = eta0 * np.exp(rng.normal(0, 0.05))
        v2 = vstar * np.exp(rng.normal(0, 0.08))
        ll2 = loglik(e2, v2)
        if np.log(rng.uniform()) < ll2 - ll:
            eta0, vstar, ll = e2, v2, ll2
        samples.append((eta0, vstar))
    return np.array(samples)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Proxy-Enabled Stochastic Fluid Sampling")
    print("=" * 60)

    # Cleanup decreases monotonically with pumped volume
    V = np.array([0.0, 10.0, 50.0, 200.0])
    eta = cleanup(0.4, 20.0, V)
    print(f"  contamination vs vol   = {np.array2string(eta, precision=3)}")
    assert np.all(np.diff(eta) < 0) and abs(eta[0] - 0.4) < 1e-9

    # Volume-to-target round-trips with the cleanup model
    v_need = volume_to_target(0.4, 20.0, 0.05)
    assert abs(cleanup(0.4, 20.0, v_need) - 0.05) < 1e-9

    # Stochastic interpretation: recover the cleanup parameters from noisy data
    eta0_true, vstar_true = 0.45, 25.0
    Vd = np.array([5, 15, 40, 90, 180.0])
    rng = np.random.default_rng(1)
    eta_obs = cleanup(eta0_true, vstar_true, Vd) * (1 + rng.normal(0, 0.05, len(Vd)))
    post = mc_posterior(Vd, eta_obs, noise=0.01, seed=2)
    burn = post[len(post) // 3:]
    eta0_hat, vstar_hat = burn[:, 0].mean(), burn[:, 1].mean()
    print(f"  posterior eta0 / V*    = {eta0_hat:.3f} / {vstar_hat:.1f} (true {eta0_true} / {vstar_true})")
    assert abs(eta0_hat - eta0_true) < 0.05 and abs(vstar_hat - vstar_true) < 8.0

    # Posterior predictive uncertainty on the volume to reach 5% contamination
    vols = volume_to_target(burn[:, 0], burn[:, 1], 0.05)
    print(f"  V to 5% : mean {vols.mean():.0f} +/- {vols.std():.0f}")
    assert vols.std() > 0
    print("  PASS")
    return {"eta0_hat": float(eta0_hat), "vstar_hat": float(vstar_hat),
            "V5_std": float(vols.std())}


if __name__ == "__main__":
    test_all()
