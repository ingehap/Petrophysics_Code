"""
Article 3: Response Characteristics of an Array Induction Tool (HDIL) in
           Heterogeneous Anisotropic Formations
Liu, Zhang, Zhang, Xu, Kang, Xiao (2020)
DOI: 10.30632/PJV61N1-2020a2

An array-induction tool synthesizes focused responses by combining subarray
measurements with focusing weights.  In anisotropic formations the relative dip
angle between the borehole and the formation distorts the apparent resistivity.
A stable software-focusing scheme solves the focusing weights as a constrained
least-squares problem (real parts summing to one, imaginary parts to zero), and
the apparent resistivity follows from the focused apparent conductivity.

Implements:

  - Anisotropy coefficient  lambda = sqrt(Rh/Rv)
  - Constrained least-squares focusing weights (sum(w) = 1)        (Eqs. 2-3, 8-9)
  - Focused apparent resistivity  rho_a = 1/Re(sum(w*sigma_a))     (Eq. 4)
  - Anisotropic apparent resistivity vs relative dip angle

Note: this issue's source-PDF text extract ended partway through this article
(page 78 of 72-85), so these are the standard induction-focusing and
anisotropic-response forms anchored to the preserved equation definitions.
Paper anchors: lambda = sqrt(Rh/Rv); anisotropy negligible at dip = 0 deg,
small for dip <= 40 deg, significant for dip >= 60 deg.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- anisotropy --------------

def anisotropy_coefficient(rh, rv):
    """Anisotropy coefficient  lambda = sqrt(Rh/Rv)  (paper's convention)."""
    return petrolib.em_dielectric.anisotropy_coefficient(rv, rh)


# ---------------------------------------------- focusing ----------------

def focusing_weights(G, target, sum_constraint=1.0):
    """Constrained least-squares focusing weights.

    Minimise ||G w - target||^2 subject to sum(w) = sum_constraint, solved via
    the KKT system (Eqs. 2-3, 8-9 of the paper).  Returns the weight vector w.
    """
    G = np.asarray(G, float)
    t = np.asarray(target, float)
    n = G.shape[1]
    GtG = G.T @ G
    Gt = G.T @ t
    ones = np.ones(n)
    # KKT: [2 GtG, 1; 1^T, 0] [w; mu] = [2 Gt; c]
    KKT = np.zeros((n + 1, n + 1))
    KKT[:n, :n] = 2.0 * GtG
    KKT[:n, n] = ones
    KKT[n, :n] = ones
    rhs = np.zeros(n + 1)
    rhs[:n] = 2.0 * Gt
    rhs[n] = sum_constraint
    sol = np.linalg.solve(KKT, rhs)
    return sol[:n]


def focused_apparent_resistivity(weights, sigma_a):
    """Focused apparent resistivity  rho_a = 1/Re(sum(w*sigma_a))  (Eq. 4)."""
    s = np.sum(np.asarray(weights, float) * np.asarray(sigma_a, complex))
    return 1.0 / s.real


# ---------------------------------------------- dip response ------------

def anisotropic_apparent_resistivity(rh, rv, dip_deg):
    """Apparent resistivity of a homogeneous anisotropic bed vs relative dip.

        rho_a = rh*sqrt(cos^2(dip) + (rv/rh)*sin^2(dip))
    Reads Rh at dip = 0 and rises toward sqrt(Rh*Rv) at 90 deg.
    """
    return petrolib.em_dielectric.apparent_resistivity_dip(rh, rv, dip_deg)


def anisotropy_effect_significant(rh, rv, dip_deg, tol=0.05):
    """True if the dip-induced apparent-resistivity change exceeds `tol` of Rh."""
    ra = anisotropic_apparent_resistivity(rh, rv, dip_deg)
    return abs(ra - rh) / rh > tol


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: HDIL Array Induction in Anisotropic Formations")
    print("=" * 60)

    # Anisotropy coefficient (paper convention sqrt(Rh/Rv))
    assert abs(anisotropy_coefficient(100.0, 25.0) - 2.0) < 1e-12

    # Focusing weights: recover a known weight set and satisfy sum(w) = 1
    rng = np.random.default_rng(1)
    G = rng.normal(size=(12, 4))
    w_true = np.array([0.4, 0.3, 0.2, 0.1])    # sums to 1
    target = G @ w_true
    w = focusing_weights(G, target, sum_constraint=1.0)
    print(f"  recovered weights      = {np.array2string(w, precision=3)}")
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.max(np.abs(w - w_true)) < 1e-6   # exact (consistent system)

    # Focused apparent resistivity from a focused apparent conductivity
    ra = focused_apparent_resistivity([0.5, 0.5], [0.02 + 0.001j, 0.02 - 0.001j])
    print(f"  focused apparent res   = {ra:.1f} ohm-m")
    assert abs(ra - 50.0) < 1e-6               # 1/Re(0.02) = 50

    # Dip response: reads Rh at 0 deg; effect negligible <=40, significant >=60
    rh, rv = 10.0, 40.0
    assert abs(anisotropic_apparent_resistivity(rh, rv, 0.0) - rh) < 1e-9
    print(f"  rho_a @0/40/60/90 deg  = "
          f"{anisotropic_apparent_resistivity(rh, rv, 0):.1f} / "
          f"{anisotropic_apparent_resistivity(rh, rv, 40):.1f} / "
          f"{anisotropic_apparent_resistivity(rh, rv, 60):.1f} / "
          f"{anisotropic_apparent_resistivity(rh, rv, 90):.1f}")
    # apparent resistivity grows monotonically with dip; the deviation from Rh
    # is far larger at high dip than at low dip
    dips = np.array([0.0, 10.0, 40.0, 60.0, 90.0])
    ra_curve = anisotropic_apparent_resistivity(rh, rv, dips)
    assert np.all(np.diff(ra_curve) > 0)
    assert not anisotropy_effect_significant(rh, rv, 10.0)   # negligible at low dip
    assert anisotropy_effect_significant(rh, rv, 70.0)       # significant at high dip
    print("  PASS")
    return {"lambda": anisotropy_coefficient(100.0, 25.0),
            "rho_a_60": float(anisotropic_apparent_resistivity(rh, rv, 60))}


if __name__ == "__main__":
    test_all()
