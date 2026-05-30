"""
Article 3: Past, Present, and Future Applications of Ultradeep Directional
Resistivity Measurements: A Case History From the Norwegian Continental Shelf
Sinha, Walmsley, Clegg, Vicuna, Wu, McGill, Paiva dos Reis, Nygard, Ulfsnes,
Constable, Antonsen, Danielsen (2022)
DOI: 10.30632/PJV63N6-2022a3

The paper traces UDAR-LWD evolution from 1D Occam-style stochastic
inversion at the transmitter measure-point through 2.5D / 3D EM inversion.
This module implements a tractable analogue of that 1D-at-the-bit
workflow:

  - Forward operator for an azimuthal-deep resistivity tool over a layered
    earth: per spacing / frequency the tool sees a geometric-factor
    weighted apparent conductivity.
  - Synthetic "8-curtain-section" measurement vector built from 4 spacings
    (7, 15, 30, 60 m) at 2 frequencies (2, 8 kHz).
  - 1D Occam-style stochastic inversion: random parameter perturbation +
    Metropolis acceptance, returning posterior samples for the
    (resistivity, boundary) state.
  - "Geostop" decision rule: if posterior 5th-percentile distance-to-base
    falls below a TVD safety margin, recommend stop.
"""

import numpy as np


# --------------------------------------------- forward operator -----------

def geometric_factor(z_centre, depth_axis, spacing_m, omega_factor=1.0):
    """Approximate boundary-aware geometric factor of an azimuthal-deep
    resistivity tool.  Gaussian profile centred on the transmitter
    measure-point with std = 0.6 * spacing (boundary sensitivity grows
    with spacing) and amplitude modulated by frequency."""
    sigma = 0.6 * spacing_m
    g = np.exp(-0.5 * ((depth_axis - z_centre) / sigma) ** 2) * omega_factor
    return g / g.sum()


def layered_resistivity(depth_axis, boundaries, resistivities):
    """Step-wise R(z) for an n-layer model."""
    idx = np.searchsorted(boundaries, depth_axis, side="right")
    idx = np.clip(idx, 0, len(resistivities) - 1)
    return np.array(resistivities)[idx]


def udar_forward(model, bit_depth, spacings_m=(7, 15, 30, 60),
                 freqs_kHz=(2, 8)):
    """Predict the multi-spacing / multi-frequency apparent log-conductivity
    vector measured at the bit.  Returns a 1-D array of length
    len(spacings) * len(freqs).
    """
    z = np.arange(bit_depth - 120.0, bit_depth + 120.0, 0.25)
    R = layered_resistivity(z, model["boundaries"], model["resistivities"])
    sigma = 1.0 / np.maximum(R, 1e-6)
    out = []
    for sp in spacings_m:
        for f in freqs_kHz:
            # Higher frequency tightens the kernel (smaller effective DOI)
            omega = 1.0 / (1.0 + 0.05 * (f - 2))
            ker = geometric_factor(bit_depth, z, sp, omega)
            sigma_app = float((ker * sigma).sum())
            out.append(np.log(sigma_app))
    return np.asarray(out)


# --------------------------------------------- inversion ------------------

def occam_stochastic_inversion(prior_model, obs, obs_sigma=0.05,
                               n_samples=4000, step=None, seed=0):
    """Metropolis sampler over (resistivities, boundaries).

    `prior_model` is the initial layered model dict.
    Returns the chain of accepted models.
    """
    rng = np.random.default_rng(seed)
    if step is None:
        step = dict(res=0.10, bd=1.0)  # multiplicative log-R step, m boundary step

    def loglik(model):
        d_pred = udar_forward(model, bit_depth=prior_model["bit_depth"])
        return -0.5 * np.sum(((d_pred - obs) / obs_sigma) ** 2)

    state = {k: list(v) if isinstance(v, list) else v for k, v in prior_model.items()}
    ll = loglik(state)
    chain = []
    accept = 0
    for i in range(n_samples):
        prop = {k: list(v) if isinstance(v, list) else v for k, v in state.items()}
        # Perturb either a resistivity or a boundary
        if rng.random() < 0.5:
            j = rng.integers(0, len(prop["resistivities"]))
            prop["resistivities"][j] *= np.exp(rng.normal(0, step["res"]))
        else:
            j = rng.integers(0, len(prop["boundaries"]))
            prop["boundaries"][j] += rng.normal(0, step["bd"])
            prop["boundaries"] = sorted(prop["boundaries"])
        ll_prop = loglik(prop)
        if np.log(rng.random()) < ll_prop - ll:
            state = prop
            ll = ll_prop
            accept += 1
        chain.append(dict(boundaries=list(state["boundaries"]),
                          resistivities=list(state["resistivities"])))
    return chain, accept / n_samples


# --------------------------------------------- geostop decision ----------

def geostop_recommendation(chain, bit_depth, safety_margin_m=5.0,
                           target_layer_index=-1, p_low=5.0):
    """If the 5th-percentile distance from bit to top of the deepest
    boundary falls below the safety margin, recommend stop.
    """
    dists = []
    for s in chain:
        boundary = s["boundaries"][target_layer_index]
        dists.append(boundary - bit_depth)
    d_p = float(np.percentile(dists, p_low))
    return d_p < safety_margin_m, d_p


# --------------------------------------------- tests ---------------------

def test_all():
    print("=" * 60)
    print("Article 3: UDAR-LWD 1-D Stochastic Inversion + Geostop")
    print("=" * 60)

    # Snorre-style scenario: bit at 3000 mTVD, BCU at 3008 m, Mime marl 5 ohm.m
    # caprock and 80 ohm.m oil zone below
    true_model = dict(
        boundaries=[2992.0, 3008.0],
        resistivities=[5.0, 80.0, 200.0],
        bit_depth=3000.0,
    )
    obs = udar_forward(true_model, bit_depth=true_model["bit_depth"])
    obs += np.random.default_rng(1).normal(0, 0.05, len(obs))
    print(f"  Number of UDAR curves        = {len(obs)}")

    # Prior - biased upward by 4 m and assumes different resistivities
    prior = dict(
        boundaries=[2988.0, 3012.0],
        resistivities=[3.0, 30.0, 100.0],
        bit_depth=3000.0,
    )
    chain, acc = occam_stochastic_inversion(prior, obs, n_samples=4000)
    burn = chain[len(chain) // 2:]
    print(f"  Sampler acceptance rate      = {acc:.2f}")
    print(f"  Posterior boundary [0]  mean = "
          f"{np.mean([s['boundaries'][0] for s in burn]):.2f} m   "
          f"(true {true_model['boundaries'][0]})")
    print(f"  Posterior boundary [1]  mean = "
          f"{np.mean([s['boundaries'][1] for s in burn]):.2f} m   "
          f"(true {true_model['boundaries'][1]})")
    print(f"  Posterior R[1] mean          = "
          f"{np.mean([s['resistivities'][1] for s in burn]):.1f} ohm.m   "
          f"(true {true_model['resistivities'][1]})")

    geostop, d_p = geostop_recommendation(burn,
                                          bit_depth=true_model["bit_depth"],
                                          safety_margin_m=5.0)
    print(f"  Geostop recommendation       = {geostop}  "
          f"(5th-pctile distance = {d_p:.2f} m)")

    # Sanity: posterior should reduce data misfit relative to the prior.
    # (Individual model parameters can drift to non-unique aliases that
    # explain the same data - which is exactly why the paper recommends
    # 2.5D / 3D inversion as a follow-on; the single-station 1D problem
    # is under-determined.)
    def misfit(m):
        return float(np.sum((udar_forward(m, m["bit_depth"]) - obs) ** 2))
    mis_prior = misfit(prior)
    mis_post = float(np.mean([misfit(dict(boundaries=list(s["boundaries"]),
                                          resistivities=list(s["resistivities"]),
                                          bit_depth=true_model["bit_depth"]))
                              for s in burn[::20]]))
    print(f"  Misfit  prior = {mis_prior:.4f}   posterior = {mis_post:.4f}")
    assert mis_post < 0.5 * mis_prior, "Posterior must halve the prior misfit"
    print("  PASS")
    return {"acceptance": float(acc),
            "misfit_prior": mis_prior, "misfit_post": mis_post,
            "geostop": geostop}


if __name__ == "__main__":
    test_all()
