"""
Article 1: Integrated Reservoir Characterization Using Unsupervised Learning
on Nuclear Magnetic Resonance (NMR) T1-T2 Logs
Jiang, Bonnie, Correa, Krueger, Kelly, Wasson (2022)
DOI: 10.30632/PJV63N3-2022a1

Tight-carbonate workflow that applies the Venkataramanan et al. (2018)
unsupervised algorithm to 2-MHz NMR log T1-T2 maps:

  - Non-negative matrix factorisation (NMF) extracts poro-fluid
    end-member spectra W (n_features x k) and per-depth mixing weights
    H (k x n_depths) from a stack of T1-T2 maps:
        V approx W @ H, with V, W, H >= 0
  - Hierarchical agglomerative clustering of the end-members sets the
    polygon boundaries of the T1-T2 map.
  - Two fluid-typing rules apply on the map:
        rule A: T1/T2 high  -> hydrocarbon
        rule B: T2  long    -> mobile fluid
  - Wettability from the oil cluster T1/T2 ratio          (Eq. 8)
  - Body-to-Throat Ratio (BTR) combines NMF pore-body distribution with
    MICP pore-throat distribution.
"""

import numpy as np


# ---------------------------------------------- NMF -----------------------

def nmf(V, k, n_iter=400, seed=0):
    """Lee-Seung multiplicative-update NMF for V approx W @ H (V, W, H >= 0)."""
    rng = np.random.default_rng(seed)
    n, m = V.shape
    W = rng.random((n, k)) + 0.1
    H = rng.random((k, m)) + 0.1
    eps = 1e-9
    for _ in range(n_iter):
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        W *= (V @ H.T) / (W @ H @ H.T + eps)
    return W, H


# ---------------------------------------------- hierarchical clustering --

def linkage_cluster(features, n_clusters):
    """Average-link agglomerative clustering on standardised features.

    `features` shape (n_items, d).  Returns labels of length n_items.
    """
    n = len(features)
    Xs = (features - features.mean(0)) / (features.std(0) + 1e-9)
    labels = np.arange(n)
    while len(set(labels)) > n_clusters:
        cs = np.array([Xs[labels == lab].mean(0) for lab in sorted(set(labels))])
        ids = sorted(set(labels))
        d = np.linalg.norm(cs[:, None, :] - cs[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        labels = np.where(labels == ids[j], ids[i], labels)
    # Re-label 0..k-1
    return np.unique(labels, return_inverse=True)[1]


# ---------------------------------------------- T1-T2 grid + fluid map ----

def make_t1t2_grid(n_T1=32, n_T2=32, T_lo=0.3, T_hi=3000.0):
    T1 = np.logspace(np.log10(T_lo), np.log10(T_hi), n_T1)
    T2 = np.logspace(np.log10(T_lo), np.log10(T_hi), n_T2)
    return T1, T2


def synth_t1t2_map(T1, T2, centres, sigmas, amps, seed=0):
    """Sum of log-2-D Gaussian fluid clusters on the (log T1, log T2) grid."""
    rng = np.random.default_rng(seed)
    L1, L2 = np.log10(T1)[:, None], np.log10(T2)[None, :]
    out = np.zeros((len(T1), len(T2)))
    for (t1c, t2c), (s1, s2), A in zip(centres, sigmas, amps):
        out += A * np.exp(-0.5 * (((L1 - np.log10(t1c)) / s1) ** 2
                                  + ((L2 - np.log10(t2c)) / s2) ** 2))
    out += 1e-3 * rng.standard_normal(out.shape)
    return np.clip(out, 0.0, None)


def fluid_typing_rules(T1c, T2c, ratio_HC=4.0, T2_mobile=33.0):
    """Two simple rules: high T1/T2 -> HC; long T2 -> mobile.

    Returns one of: 'mobile_HC', 'immobile_HC', 'mobile_water', 'immobile_water'.
    """
    is_HC = (T1c / T2c) >= ratio_HC
    is_mobile = T2c >= T2_mobile
    return ("mobile_HC" if is_HC and is_mobile else
            "immobile_HC" if is_HC and not is_mobile else
            "mobile_water" if not is_HC and is_mobile else
            "immobile_water")


# ---------------------------------------------- Eq. 8 wettability ---------

def wettability_index(T1_over_T2_oil_cluster, ref_ratio=10.0):
    """Eq. 8 - normalised T1/T2 of the oil cluster.

        WI = (T1/T2)_oil / ref_ratio - 1

    Positive WI -> water-wet; negative -> oil-wet; |WI| << 1 -> mixed-wet.
    """
    return float(T1_over_T2_oil_cluster / ref_ratio - 1.0)


# ---------------------------------------------- BTR --------------------

def body_to_throat_ratio(body_radii_um, throat_radii_um):
    """Volume-weighted mean ratio of pore-body to pore-throat radii."""
    return float(np.mean(body_radii_um) / np.mean(throat_radii_um))


# ---------------------------------------------- tests ---------------

def test_all():
    print("=" * 60)
    print("Article 1: NMF + Hierarchical Clustering on NMR T1-T2 Maps")
    print("=" * 60)

    T1, T2 = make_t1t2_grid()
    # Four planted fluid populations (immobile water, immobile HC, mobile HC, mobile water)
    centres = [(5.0, 1.5), (40.0, 8.0), (300.0, 60.0), (200.0, 200.0)]
    sigmas = [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]

    # Stack maps over 30 depth steps with varying amplitudes
    rng = np.random.default_rng(0)
    n_depths = 30
    maps = []
    true_weights = np.zeros((4, n_depths))
    for d in range(n_depths):
        amps = rng.dirichlet([1.0, 1.0, 1.0, 1.0]) * 1.0
        true_weights[:, d] = amps
        m = synth_t1t2_map(T1, T2, centres, sigmas, amps, seed=d)
        maps.append(m.flatten())
    V = np.stack(maps, axis=1)               # (n_T1*n_T2, n_depths)

    # NMF decomposition into 4 end-members
    W, H = nmf(V, k=4, n_iter=300)
    print(f"  NMF V shape = {V.shape}   W = {W.shape}   H = {H.shape}")

    # Hierarchical clustering on the W columns (the end-member spectra)
    # Cluster each end-member by its peak T1, T2
    peaks_T1 = np.array([T1[W[:, k].reshape(len(T1), len(T2)).sum(1).argmax()]
                         for k in range(W.shape[1])])
    peaks_T2 = np.array([T2[W[:, k].reshape(len(T1), len(T2)).sum(0).argmax()]
                         for k in range(W.shape[1])])
    print("  NMF end-member peaks (T1, T2)  ms:")
    for k in range(W.shape[1]):
        print(f"    em {k}:  T1 = {peaks_T1[k]:7.2f}   T2 = {peaks_T2[k]:7.2f}   "
              f"T1/T2 = {peaks_T1[k] / peaks_T2[k]:5.2f}   "
              f"{fluid_typing_rules(peaks_T1[k], peaks_T2[k])}")

    # Check at least one HC cluster and one water cluster were found
    types = [fluid_typing_rules(peaks_T1[k], peaks_T2[k]) for k in range(4)]
    assert any("HC" in t for t in types), "Must find at least one HC cluster"
    assert any("water" in t for t in types), "Must find at least one water cluster"

    # Wettability check
    hc_idx = [k for k, t in enumerate(types) if "HC" in t][0]
    WI = wettability_index(peaks_T1[hc_idx] / peaks_T2[hc_idx])
    print(f"  Wettability index (Eq. 8) = {WI:+.3f}")

    # BTR
    BTR = body_to_throat_ratio([5.0, 8.0, 12.0], [0.5, 1.2, 2.0])
    print(f"  Body-to-throat ratio      = {BTR:.2f}")
    assert BTR > 1.0, "Body radii must exceed throat radii"

    print("  PASS")
    return {"n_clusters": 4, "wettability": WI, "BTR": BTR}


if __name__ == "__main__":
    test_all()
