"""
Article 5: Data-Driven Interpretation of Ultradeep Azimuthal Propagation
           Resistivity Measurements: Transdimensional Stochastic Inversion and
           Uncertainty Quantification
Shen, Chen, Wang (2018)
DOI: 10.30632/PJV59N6Y2018a4

Ultradeep azimuthal propagation-resistivity data are inverted for a layered
earth whose number of layers is itself unknown.  A transdimensional (reversible-
jump) Markov-chain Monte Carlo sampler explores models of varying dimension,
producing a posterior ensemble that both recovers the layering and quantifies
its uncertainty.

Implements:

  - Layered-earth forward response (depth-sampled resistivity profile)
  - Transdimensional MCMC with birth/death (variable layer count) + perturb moves
  - Posterior mean and uncertainty (per-depth resistivity spread)
  - Recovery of the boundary count and resistivities

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so this is a faithful standard-form reconstruction of the
transdimensional Bayesian inversion the paper applies (a compact RJ-MCMC).
"""

import numpy as np


# ---------------------------------------------- forward -----------------

def layered_profile(boundaries, resistivities, depths):
    """Resistivity at each depth from layer boundaries and per-layer resistivity."""
    depths = np.asarray(depths, float)
    out = np.full(len(depths), resistivities[0], float)
    for b, r in zip(boundaries, resistivities[1:]):
        out[depths >= b] = r
    return out


# ---------------------------------------------- RJ-MCMC -----------------

def transdim_invert(depths, data, noise, n_iter=6000, seed=0):
    """Compact reversible-jump MCMC over a layered resistivity model.

    Moves: perturb a resistivity, move a boundary, birth (add boundary), death
    (remove boundary).  Returns the posterior ensemble of depth profiles.
    """
    rng = np.random.default_rng(seed)
    dmin, dmax = depths.min(), depths.max()

    def loglik(profile):
        return -0.5 * np.sum(((profile - data) / noise) ** 2)

    # start from a 1-layer model at the data mean
    bnds = []
    res = [float(np.mean(data))]
    cur = layered_profile(bnds, res, depths)
    cur_ll = loglik(cur)
    ensemble = []
    for it in range(n_iter):
        move = rng.integers(0, 4)
        nb, nr = list(bnds), list(res)
        if move == 0:                              # perturb a resistivity
            i = rng.integers(0, len(nr)); nr[i] *= np.exp(rng.normal(0, 0.1))
        elif move == 1 and nb:                     # move a boundary
            i = rng.integers(0, len(nb)); nb[i] = float(np.clip(nb[i] + rng.normal(0, 5.0), dmin, dmax)); nb.sort()
        elif move == 2 and len(nb) < 6:            # birth
            nb.append(float(rng.uniform(dmin, dmax))); nb.sort()
            nr.insert(rng.integers(1, len(nr) + 1), float(np.mean(data)) * np.exp(rng.normal(0, 0.3)))
        elif move == 3 and nb:                     # death
            i = rng.integers(0, len(nb)); nb.pop(i); nr.pop(i + 1)
        if len(nr) != len(nb) + 1:
            continue
        prof = layered_profile(nb, nr, depths)
        ll = loglik(prof)
        if np.log(rng.uniform()) < ll - cur_ll:    # Metropolis acceptance
            bnds, res, cur, cur_ll = nb, nr, prof, ll
        if it > n_iter // 3:
            ensemble.append(cur.copy())
    return np.array(ensemble)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Ultradeep Resistivity Transdimensional Inversion")
    print("=" * 60)

    # True 3-layer earth (2 boundaries)
    depths = np.linspace(0, 100, 60)
    true = layered_profile([35.0, 70.0], [10.0, 100.0, 20.0], depths)
    rng = np.random.default_rng(0)
    data = true * np.exp(rng.normal(0, 0.05, len(true)))    # 5% noise
    noise = 0.05 * true

    ensemble = transdim_invert(depths, data, noise, n_iter=6000, seed=1)
    post_mean = ensemble.mean(0)
    post_std = ensemble.std(0)
    rel_err = np.linalg.norm(post_mean - true) / np.linalg.norm(true)
    print(f"  posterior rel. error   = {rel_err:.3f}")
    assert rel_err < 0.15                          # recovers the layering

    # Uncertainty is elevated near the boundaries (where the model is least sure)
    near_b = (np.abs(depths - 35.0) < 5) | (np.abs(depths - 70.0) < 5)
    print(f"  post std near/away bnd = {post_std[near_b].mean():.2f} / {post_std[~near_b].mean():.2f}")
    assert post_std[near_b].mean() > post_std[~near_b].mean()
    # the high/low resistivity contrast is recovered
    assert post_mean[depths < 35].mean() < post_mean[(depths >= 35) & (depths < 70)].mean()
    print("  PASS")
    return {"rel_error": float(rel_err), "n_samples": len(ensemble)}


if __name__ == "__main__":
    test_all()
