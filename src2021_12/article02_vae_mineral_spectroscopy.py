"""
Article 2: Enhanced Mineral Quantification and Uncertainty Analysis From
           Downhole Spectroscopy Logs Using Variational Autoencoders
Craddock, Srivastava, Datir, Rose, Zhou, Mosse, Venkataramanan (2021)
DOI: 10.30632/PJV62N6-2021a2

A variational-autoencoder framework maps dry-weight elemental mass fractions
(from neutron-induced geochemical spectroscopy) to dry-weight mineral mass
fractions and their uncertainties.  An encoder ANN maps elements -> minerals
(the labelled latent space); a decoder reconstructs elements from minerals as
a QC.  Training samples each element from a Gaussian at its measurement
uncertainty (aleatoric propagation) and minimises a heteroscedastic
(variance-weighted) Gaussian negative-log-likelihood cost.

Implements:

  - Forward element model  e = A @ m  (stoichiometric sensitivity matrix)
  - Heteroscedastic per-term loss  (y-yhat)^2/(2 sigma^2) + ln(sigma)  (Eq. 1)
  - Total cost summed over outputs and samples                         (Eq. 2)
  - Non-negative, closure-constrained mineral inversion (encoder proxy)
  - Matrix grain density from mineral fractions

Note: the journal's Eqs. 1-2 are image-rendered and were not in the text; the
forms here are faithful reconstructions of the heteroscedastic-Gaussian NLL
the paper describes.  The element->mineral matrix uses standard mineral-
formula mass fractions (the paper publishes no numeric matrix).  A full VAE
is replaced by a closure-constrained least-squares inversion so the module
runs with numpy alone while demonstrating the same forward physics and loss.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# Elements (rows) and minerals (columns) of the sensitivity matrix
ELEMENTS = ["Si", "Al", "Ca", "Fe", "Mg", "K", "Na", "Ti", "S"]
MINERALS = ["Quartz", "K-Feldspar", "Plagioclase", "Illite", "Kaolinite",
            "Calcite", "Dolomite", "Siderite", "Pyrite", "Anhydrite"]

# Idealised dry-weight element mass fractions per mineral formula (dimensionless)
#   rows = ELEMENTS, cols = MINERALS
_A = np.array([
    # Qz    Kfs    Plag   Ill    Kao    Cal    Dol    Sid    Pyr    Anh
    [0.467, 0.302, 0.310, 0.250, 0.218, 0.000, 0.000, 0.000, 0.000, 0.000],  # Si
    [0.000, 0.097, 0.130, 0.180, 0.209, 0.000, 0.000, 0.000, 0.000, 0.000],  # Al
    [0.000, 0.000, 0.030, 0.000, 0.000, 0.400, 0.217, 0.000, 0.000, 0.294],  # Ca
    [0.000, 0.000, 0.000, 0.030, 0.000, 0.000, 0.000, 0.482, 0.466, 0.000],  # Fe
    [0.000, 0.000, 0.000, 0.010, 0.000, 0.000, 0.132, 0.000, 0.000, 0.000],  # Mg
    [0.000, 0.140, 0.000, 0.060, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # K
    [0.000, 0.000, 0.070, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # Na
    [0.000, 0.000, 0.000, 0.005, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # Ti
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.534, 0.235],  # S
])

# Mineral grain densities (g/cc) for the matrix-density product
_RHO_GRAIN = np.array([2.65, 2.55, 2.62, 2.75, 2.60, 2.71, 2.85, 3.94, 5.01, 2.96])


def element_mineral_matrix():
    """Return (A, ELEMENTS, MINERALS): the stoichiometric sensitivity matrix."""
    return _A.copy(), list(ELEMENTS), list(MINERALS)


# ---------------------------------------------- forward model -----------

def forward_elements(m, A=_A):
    """Predict element mass fractions from minerals:  e = A @ m."""
    return A @ np.asarray(m, float)


# ---------------------------------------------- Eqs. 1-2: loss ----------

def heteroscedastic_term(y, yhat, sigma):
    """Per-term Gaussian NLL  (y-yhat)^2/(2 sigma^2) + ln(sigma)  (Eq. 1)."""
    sigma = np.maximum(np.asarray(sigma, float), 1e-9)
    return (np.asarray(y, float) - np.asarray(yhat, float)) ** 2 / (2 * sigma ** 2) \
        + np.log(sigma)


def total_cost(Y, Yhat, Sigma):
    """Total cost summed over all outputs and samples (Eq. 2)."""
    return float(np.sum(heteroscedastic_term(Y, Yhat, Sigma)))


# ---------------------------------------------- encoder (inversion) -----

def _project_simplex(v):
    """Euclidean projection onto the probability simplex {m >= 0, sum(m) = 1}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(len(u)) + 1) > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def invert_minerals(e, A=_A, iters=6000):
    """Recover non-negative, sum-to-one mineral fractions from elements.

    Projected-gradient least squares minimising ||A m - e||^2 subject to
    m >= 0 and sum(m) = 1, using the exact Euclidean simplex projection
    (a numpy stand-in for the trained encoder).  Returns (m_hat, e_recon).
    """
    e = np.asarray(e, float)
    n = A.shape[1]
    m = np.full(n, 1.0 / n)
    AtA = A.T @ A
    Ate = A.T @ e
    lr = 1.0 / np.linalg.norm(AtA, 2)      # 1/Lipschitz step size
    for _ in range(iters):
        m = _project_simplex(m - lr * (AtA @ m - Ate))
    return m, A @ m


# ---------------------------------------------- matrix density ----------

def matrix_density(m, rho_grain=_RHO_GRAIN):
    """Mass-weighted matrix grain density  rho = sum(m_k * rho_k)."""
    return float(petrolib.porosity_lithology.matrix_density_from_volumes(m, rho_grain))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: VAE Mineral Quantification From Spectroscopy")
    print("=" * 60)

    A, elems, mins = element_mineral_matrix()
    print(f"  matrix shape           = {A.shape} ({len(elems)} elems x {len(mins)} mins)")

    # Plant a realistic shaly-sand mineralogy
    rng = np.random.default_rng(3)
    m_true = np.array([0.45, 0.08, 0.07, 0.12, 0.05, 0.10, 0.04, 0.02, 0.04, 0.03])
    m_true = m_true / m_true.sum()
    e = forward_elements(m_true)
    print(f"  forward elements e     = {np.array2string(e, precision=3)}")

    # Encoder inversion reconstructs the element log (the paper's QC criterion)
    m_hat, e_rec = invert_minerals(e)
    rec_err = float(np.max(np.abs(e_rec - e)))
    print(f"  max element reconstruction error = {rec_err:.4f}")
    assert abs(m_hat.sum() - 1.0) < 1e-6
    assert np.all(m_hat >= -1e-9)
    assert rec_err < 5e-3, "decoder QC: reconstructed elements must match input"

    # Matrix density lands in a sensible sedimentary range
    rho_ma = matrix_density(m_true)
    print(f"  matrix grain density   = {rho_ma:.3f} g/cc")
    assert 2.55 < rho_ma < 3.1

    # Heteroscedastic loss: smaller when sigma matches the true error scale
    y = m_true
    yhat = m_true + rng.normal(0, 0.02, m_true.size)
    cost_matched = total_cost(y, yhat, np.full_like(y, 0.02))
    cost_overconf = total_cost(y, yhat, np.full_like(y, 0.002))  # too-tight sigma
    print(f"  cost (sigma matched)   = {cost_matched:.2f}")
    print(f"  cost (sigma too tight) = {cost_overconf:.2f}")
    assert cost_matched < cost_overconf, "overconfident sigma must be penalised"
    print("  PASS")
    return {"rec_err": rec_err, "rho_ma": rho_ma, "m_hat": m_hat}


if __name__ == "__main__":
    test_all()
