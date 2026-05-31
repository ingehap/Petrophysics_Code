"""
Article 2: Predicting Carbonate Rock Properties Using NMR Data and Generalized
           Interpolation-Based Techniques
Kwak, Hursan, Shao, Chen, Balliet, Eid, Guergueb (2016)
Reference: Petrophysics Vol. 57, No. 4 (August 2016), pp. 351-368
DOI: none assigned (this issue predates SPWLA DOI assignment)

For heterogeneous carbonates the relationship between NMR relaxation and
petrophysical properties (permeability, pore-throat size, Thomeer parameters) is
strongly nonlinear, so a closed-form transform underperforms.  This paper uses a
radial basis function (RBF) generalized-interpolation mapping from NMR features
to the target property, with the high-dimensional T2 distribution first reduced
by principal component analysis (PCA).  The classical Coates and SDR (Kenyon)
permeability transforms are the closed-form baselines.

Implements:

  - NMR pore-size relation  1/T = rho*(S/V)  (Eq. 1)
  - Coates et al. (1991) permeability  k = (phi/C)^4 * (FFI/BVI)^2  (Eq. 2)
  - SDR / Kenyon permeability  k = a * phi^4 * T2gm^2  (Eq. 3)
  - T2 logarithmic-mean (geometric mean) relaxation time
  - PCA dimensionality reduction of T2 distributions
  - RBF interpolation fit and evaluation (Gaussian / multiquadric)

Note: this issue's PDF has a text layer; the pore-size (Eq. 1), Coates (Eq. 2)
and SDR (Eq. 3) relations are transcribed from the body, while the typeset
glyphs were dropped and reconstructed in standard form.  The pore-throat models
of Clerke (Eq. 4) and Thomeer (Eq. 5) are referenced but not reproduced here;
the RBF/PCA machinery is the paper's actual contribution.  Permeability in mD,
times in ms, porosity as a fraction.
"""

import numpy as np


# ---------------------------------------------- NMR pore size --------------

def surface_to_volume(relaxation_time, rho):
    """Pore surface-to-volume ratio from the relaxation time (Eq. 1)

        1/T = rho*(S/V)  ->  S/V = 1/(rho*T),

    with rho the (T1 or T2) surface relaxivity.
    """
    return 1.0 / (rho * np.asarray(relaxation_time, float))


# ---------------------------------------------- closed-form permeability --------------

def coates_permeability(phi, ffi, bvi, c=10.0):
    """Coates et al. (1991) permeability (Eq. 2)

        k = (phi/C)^4 * (FFI/BVI)^2,

    with FFI/BVI the free-fluid / bound-fluid ratio split at a T2 cutoff.
    """
    return (phi / c) ** 4 * (ffi / bvi) ** 2


def sdr_permeability(phi, t2_gm, a=4.0):
    """SDR / Kenyon (1986) permeability (Eq. 3)

        k = a * phi^4 * T2gm^2,

    with T2gm the geometric-mean (log-mean) T2.
    """
    return a * phi ** 4 * t2_gm ** 2


def t2_logmean(t2_bins, amplitudes):
    """Logarithmic-mean (geometric-mean) T2 of a distribution

        T2gm = exp( sum(A_i * ln T2_i) / sum(A_i) ).
    """
    t2 = np.asarray(t2_bins, float)
    a = np.asarray(amplitudes, float)
    return float(np.exp(np.sum(a * np.log(t2)) / np.sum(a)))


# ---------------------------------------------- PCA --------------

def pca_reduce(distributions, n_components):
    """Reduce T2 distributions to their leading principal components.

    distributions: (n_samples, n_bins) array.  Returns (scores, components,
    mean) where scores is (n_samples, n_components); only the first few PCs of
    the T2 distribution are used as RBF inputs.
    """
    x = np.asarray(distributions, float)
    mean = x.mean(axis=0)
    xc = x - mean
    # SVD of the centered data; right-singular vectors are the principal axes
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:n_components]
    scores = xc @ components.T
    return scores, components, mean


# ---------------------------------------------- RBF interpolation --------------

def _rbf_kernel(r, kind, epsilon):
    if kind == "gaussian":
        return np.exp(-(epsilon * r) ** 2)
    if kind == "multiquadric":
        return np.sqrt(1.0 + (epsilon * r) ** 2)
    raise ValueError(f"unknown RBF kind: {kind}")


def rbf_fit(centers, values, kind="gaussian", epsilon=1.0):
    """Fit RBF interpolation weights so that f(center_i) = value_i

        f(x) = sum_j w_j * phi(||x - center_j||),

    solving the (n x n) linear system Phi w = values.  Returns a dict model.
    """
    centers = np.atleast_2d(np.asarray(centers, float))
    values = np.asarray(values, float)
    dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    phi = _rbf_kernel(dist, kind, epsilon)
    weights = np.linalg.solve(phi, values)
    return {"centers": centers, "weights": weights, "kind": kind, "epsilon": epsilon}


def rbf_eval(model, x):
    """Evaluate a fitted RBF model at points x."""
    x = np.atleast_2d(np.asarray(x, float))
    dist = np.linalg.norm(x[:, None, :] - model["centers"][None, :, :], axis=-1)
    phi = _rbf_kernel(dist, model["kind"], model["epsilon"])
    return phi @ model["weights"]


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Carbonate NMR + RBF Interpolation")
    print("=" * 60)

    # Pore S/V from relaxation time (larger T -> larger pore -> smaller S/V)
    assert surface_to_volume(100.0, 20e-6) < surface_to_volume(10.0, 20e-6)

    # Coates and SDR permeability increase strongly with porosity
    k_coates = coates_permeability(0.20, ffi=0.6, bvi=0.4)
    k_sdr = sdr_permeability(0.20, t2_gm=80.0)
    print(f"  Coates / SDR k         = {k_coates:.3f} / {k_sdr:.1f} mD")
    assert coates_permeability(0.30, 0.6, 0.4) > k_coates > 0
    assert sdr_permeability(0.30, 80.0) > k_sdr > 0

    # T2 log-mean lies within the distribution's T2 range
    bins = np.array([1.0, 10.0, 100.0, 1000.0])
    amps = np.array([0.1, 0.4, 0.4, 0.1])
    gm = t2_logmean(bins, amps)
    print(f"  T2 log-mean            = {gm:.2f} ms")
    assert bins[0] < gm < bins[-1]

    # PCA reduces dimensionality and keeps the leading variance
    rng = np.random.default_rng(3)
    base = np.linspace(0, 1, 20)
    dists = np.array([np.exp(-((base - c) ** 2) / 0.05) for c in rng.uniform(0.2, 0.8, 30)])
    scores, comps, mean = pca_reduce(dists, n_components=3)
    assert scores.shape == (30, 3) and comps.shape == (3, 20)

    # RBF interpolation exactly reproduces the training targets (nonlinear map)
    centers = rng.uniform(0, 1, (8, 2))
    targets = np.sin(centers[:, 0] * 3) + centers[:, 1] ** 2
    model = rbf_fit(centers, targets, kind="multiquadric", epsilon=2.0)
    pred = rbf_eval(model, centers)
    print(f"  RBF train RMSE         = {np.sqrt(np.mean((pred - targets) ** 2)):.2e}")
    assert np.allclose(pred, targets, atol=1e-8)
    print("  PASS")
    return {"k_coates": float(k_coates), "k_sdr": float(k_sdr), "T2gm": gm}


if __name__ == "__main__":
    test_all()
