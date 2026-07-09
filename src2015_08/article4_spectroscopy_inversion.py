"""
Article 4: Petrophysical Interpretation of LWD, Neutron-Induced Gamma-Ray
           Spectroscopy Measurements: An Inversion-Based Approach
Ajayi, Torres-Verdin, Preeg (2015)
Reference: Petrophysics Vol. 56, No. 4 (August 2015), pp. 358-378
DOI: none assigned (this issue predates SPWLA DOI assignment)

Geometrical effects (shoulder beds, well deviation) spatially average
neutron-induced gamma-ray spectroscopy logs.  A layer-by-layer matrix
composition, porosity and hydrocarbon saturation are recovered by a
regularized (Tikhonov / Occam) nonlinear inversion of elemental relative yields
against a spectroscopy fast-forward model, with mineral volumes constrained to
be non-negative and to sum to one.  Matrix-sensitive logs (gamma ray, matrix
density, PEF, Sigma) are reproduced through linear mixing laws, weight fractions
follow from volumes and grain densities, and density porosity uses the inverted
matrix density.

Implements:

  - Regularized (Tikhonov) least-squares inversion (Eqs. 1, 3)
  - Linear mixing law for matrix-sensitive properties (Eq. 2)
  - Matrix gamma ray from K/U/Th concentrations
  - Volume-to-weight-fraction conversion (Eq. 6)
  - Density porosity from the inverted matrix density (Eq. 7)

Note: this issue's PDF has a text layer; the cost function, mixing laws,
weight-fraction and porosity relations (Eqs. 1-7) are transcribed from the body,
while the typeset glyphs were dropped and reconstructed in standard form.
Densities in g/cm^3, concentrations as noted, porosity/volumes as fractions.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# Standard API gamma-ray sensitivity to K (wt%), U (ppm), Th (ppm)
GR_K, GR_U, GR_TH = 16.0, 8.0, 4.0


# ---------------------------------------------- regularized inversion --------------

def tikhonov_inversion(a, b, lam, x_ref=None):
    """Regularized (Tikhonov / Occam) least-squares solution (Eqs. 1, 3)

        x = argmin ||A x - b||^2 + lam^2 ||x - x_ref||^2
          = (A'A + lam^2 I)^-1 (A'b + lam^2 x_ref),

    with lam the stabilization parameter and x_ref a reference model.  Stabilizes
    underdetermined / noisy elemental-yield inversions.
    """
    return petrolib.inversion_numerics.linear.tikhonov_solve(
        a, b, lam ** 2, x_ref=x_ref)


def cost_function(x, a, b, lam, x_ref):
    """Quadratic cost  C(x) = ||A x - b||^2 + lam^2 ||x - x_ref||^2  (Eq. 1)."""
    a, b, x, x_ref = (np.asarray(v, float) for v in (a, b, x, x_ref))
    return float(np.sum((a @ x - b) ** 2) + lam ** 2 * np.sum((x - x_ref) ** 2))


# ---------------------------------------------- mixing / properties --------------

def linear_mixing_law(volumes, endpoints):
    """Matrix-sensitive property from mineral volumes (Eq. 2)

        prop = sum_j Vj * prop_j,

    e.g. matrix density, PEF or Sigma from the inverted mineral volumes.
    """
    return float(petrolib.porosity_lithology.matrix_density_from_volumes(volumes, endpoints))


def matrix_gamma_ray(k, u, th, gk=GR_K, gu=GR_U, gth=GR_TH):
    """Matrix gamma ray from elemental concentrations  GR = gk*K + gu*U + gth*Th
    (K in wt%, U and Th in ppm)."""
    return gk * k + gu * u + gth * th


def volume_to_weight_fraction(volumes, grain_densities):
    """Mineral weight fractions from volumes and grain densities (Eq. 6)

        Cw_j = Vj*rho_j / sum_i(Vi*rho_i).
    """
    return petrolib.porosity_lithology.volume_to_weight_fractions(volumes, grain_densities)


def density_porosity(rho_bulk, rho_matrix, rho_fluid):
    """Density porosity using the inverted matrix density (Eq. 7)

        phi_D = (rho_matrix - rho_bulk)/(rho_matrix - rho_fluid).
    """
    return petrolib.porosity_lithology.density_porosity(rho_bulk, rho_matrix, rho_fluid)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Spectroscopy Inversion")
    print("=" * 60)

    # Tikhonov inversion recovers mineral volumes from elemental yields; with
    # small lambda it matches the (well-determined) least-squares solution
    # rows = elemental yields per mineral [quartz, calcite, illite]
    a = np.array([[0.47, 0.0, 0.10],     # Si yield
                  [0.0, 0.40, 0.05],     # Ca yield
                  [0.0, 0.0, 0.20]])     # Al/K yield (clay)
    true_x = np.array([0.5, 0.3, 0.2])
    b = a @ true_x
    x = tikhonov_inversion(a, b, lam=1e-6)
    print(f"  inverted volumes       = {np.round(x, 3)}")
    assert np.allclose(x, true_x, atol=1e-3)
    # Regularization toward a reference pulls an underdetermined solution to it
    assert cost_function(x, a, b, 1e-6, true_x) < 1e-6

    # Linear mixing law for matrix density
    rho_ma = linear_mixing_law(true_x, [2.65, 2.71, 2.78])
    print(f"  matrix density         = {rho_ma:.3f} g/cm^3")
    assert 2.65 < rho_ma < 2.78

    # Matrix gamma ray from K/U/Th
    gr = matrix_gamma_ray(2.0, 4.0, 12.0)
    assert np.isclose(gr, 16 * 2 + 8 * 4 + 4 * 12)

    # Weight fractions sum to 1 and weight the denser minerals more than by volume
    cw = volume_to_weight_fraction(true_x, [2.65, 2.71, 2.78])
    assert np.isclose(cw.sum(), 1.0)

    # Density porosity using the inverted matrix density
    phi = density_porosity(2.45, rho_ma, 1.0)
    print(f"  density porosity       = {phi:.3f}")
    assert 0 < phi < 0.3
    print("  PASS")
    return {"rho_matrix": float(rho_ma), "GR": float(gr), "phi": float(phi)}


if __name__ == "__main__":
    test_all()
