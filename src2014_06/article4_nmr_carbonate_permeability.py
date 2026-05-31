"""
Article 4: Method for Predicting Permeability of Complex Carbonate Reservoirs
           Using NMR Logging Measurements
Willian Trevizan, Paulo Neto, Bernardo Coutinho, Vinicius F. Machado,
Edmilson H. Rios, Songhua Chen, Wei Shao, Pedro Romero (2014)
Reference: Petrophysics Vol. 55, No. 3 (June 2014), pp. 240-252
DOI: none assigned (this issue predates SPWLA DOI assignment)

The classic NMR permeability transforms (Timur-Coates and SDR) perform poorly in
complex carbonates, so the paper trains a regularized radial-basis-function (RBF)
model on the full normalized T2 distribution plus porosity, after reducing the
inputs with principal-component analysis.

Implements:

  - T2 logarithmic (geometric) mean of a distribution
  - FFI/BVI partition of the T2 distribution at a cutoff
  - Timur-Coates permeability  k = (phi/C)^4*(FFI/BVI)^2  (Eq. 1)
  - SDR permeability  k = a*phi^4*T2gm^2  (Eq. 2)
  - Gaussian radial basis function and an RBF predictor  k = R(phi, N(T2))
  - PCA variance fraction for input reduction (Eq. 6)

Note: this issue's PDF dropped most display equations in extraction; the Coates
(Eq. 1) and SDR (Eq. 2) transforms are reconstructed in their standard forms, and
the RBF / PCA relations follow the described regularized-RBF-with-PCA method.
A default T2 cutoff of 100 ms is reported.  Permeability in mD, T2 in ms.
"""

import numpy as np


# ---------------------------------------------- T2 statistics --------------

def t2_logmean(t2, amplitudes):
    """Logarithmic (geometric) mean of a T2 distribution

        T2gm = exp(sum_i A_i*ln(T2_i)/sum_i A_i).
    """
    t2 = np.asarray(t2, float)
    a = np.asarray(amplitudes, float)
    return float(np.exp(np.sum(a * np.log(t2)) / np.sum(a)))


def ffi_bvi(t2, amplitudes, t2_cutoff=100.0):
    """Free-fluid and bound-volume indices from the T2 distribution at a cutoff

        BVI = sum(A_i : T2_i <= cutoff),  FFI = sum(A_i : T2_i > cutoff).

    Returns (FFI, BVI).
    """
    t2 = np.asarray(t2, float)
    a = np.asarray(amplitudes, float)
    bvi = float(a[t2 <= t2_cutoff].sum())
    ffi = float(a[t2 > t2_cutoff].sum())
    return ffi, bvi


# ---------------------------------------------- classic transforms --------------

def coates_permeability(phi, ffi, bvi, c=10.0):
    """Timur-Coates permeability (Eq. 1)

        k = (phi/C)^4*(FFI/BVI)^2,

    with the calibration coefficient C (~10) and the free/bound fluid ratio.
    """
    return (phi / c) ** 4 * (ffi / bvi) ** 2


def sdr_permeability(phi, t2gm, a=4.0):
    """Schlumberger-Doll-Research permeability (Eq. 2)

        k = a*phi^4*T2gm^2,

    with the calibration constant a and the logarithmic-mean T2.
    """
    return a * phi ** 4 * t2gm ** 2


# ---------------------------------------------- RBF model --------------

def gaussian_rbf(r, sigma):
    """Gaussian radial basis function  g(r) = exp(-r^2/(2*sigma^2))."""
    return np.exp(-np.asarray(r, float) ** 2 / (2.0 * sigma ** 2))


def rbf_predict(x, centers, weights, sigma):
    """Radial-basis-function predictor

        R(x) = sum_j w_j*g(||x - c_j||),

    with Gaussian kernels at the centers c_j.  ``x`` is one input vector,
    ``centers`` has shape (M, n_features).
    """
    x = np.asarray(x, float)
    c = np.asarray(centers, float)
    r = np.linalg.norm(c - x[None, :], axis=1)
    return float(np.asarray(weights, float) @ gaussian_rbf(r, sigma))


def rbf_fit(inputs, targets, sigma, ridge=1e-6):
    """Fit RBF weights by regularized least squares using the training inputs as
    centers

        (G^T G + lambda I) w = G^T y,   G_ij = g(||x_i - c_j||).

    Returns the weights w.
    """
    x = np.asarray(inputs, float)
    y = np.asarray(targets, float)
    n = x.shape[0]
    g = np.empty((n, n))
    for i in range(n):
        g[i] = gaussian_rbf(np.linalg.norm(x - x[i][None, :], axis=1), sigma)
    w = np.linalg.solve(g.T @ g + ridge * np.eye(n), g.T @ y)
    return w


# ---------------------------------------------- PCA --------------

def pca_variance_fraction(data):
    """Fraction of variance explained by each principal component (Eq. 6)

        fraction_k = lambda_k/sum_i lambda_i,

    with lambda_i the eigenvalues of the (centered) data covariance matrix.
    Returns the fractions sorted in descending order.
    """
    x = np.asarray(data, float)
    x = x - x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    eig = np.linalg.eigvalsh(cov)[::-1]
    eig = np.clip(eig, 0, None)
    return eig / eig.sum()


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: NMR Permeability of Complex Carbonates")
    print("=" * 60)

    # Synthetic bimodal T2 distribution
    t2 = np.logspace(0, 3.5, 60)        # 1 ms to ~3 s
    amps = (np.exp(-((np.log10(t2) - 1.0) / 0.3) ** 2)
            + 0.8 * np.exp(-((np.log10(t2) - 2.7) / 0.3) ** 2))

    t2gm = t2_logmean(t2, amps)
    ffi, bvi = ffi_bvi(t2, amps, t2_cutoff=100.0)
    print(f"  T2gm={t2gm:.1f} ms  FFI={ffi:.2f}  BVI={bvi:.2f}")
    assert t2gm > 0 and ffi > 0 and bvi > 0

    # Classic transforms give positive permeabilities; k rises with porosity
    k_co = coates_permeability(0.20, ffi, bvi)
    k_sdr = sdr_permeability(0.20, t2gm)
    print(f"  k_Coates={k_co:.3f} mD  k_SDR={k_sdr:.3f} mD")
    assert k_co > 0 and k_sdr > 0
    assert coates_permeability(0.30, ffi, bvi) > k_co
    assert sdr_permeability(0.30, t2gm) > k_sdr

    # Gaussian RBF is 1 at the center and decays
    assert np.isclose(gaussian_rbf(0.0, 1.0), 1.0)
    assert gaussian_rbf(2.0, 1.0) < gaussian_rbf(1.0, 1.0)

    # RBF regression reproduces a smooth synthetic permeability mapping
    rng = np.random.default_rng(0)
    X = rng.uniform(0.05, 0.35, (40, 2))
    y = 1000 * X[:, 0] ** 4 * (X[:, 1] * 100) ** 2   # SDR-like truth
    w = rbf_fit(X, y, sigma=0.1)
    pred = rbf_predict(X[5], X, w, sigma=0.1)
    print(f"  RBF prediction error = {abs(pred - y[5]):.3e}")
    assert np.isclose(pred, y[5], rtol=1e-2)

    # PCA: first few components explain most of the variance
    frac = pca_variance_fraction(X)
    print(f"  PCA variance fractions = {np.round(frac, 3)}")
    assert np.isclose(frac.sum(), 1.0) and frac[0] >= frac[-1]
    print("  PASS")
    return {"T2gm": t2gm, "k_Coates": float(k_co), "k_SDR": float(k_sdr)}


if __name__ == "__main__":
    test_all()
