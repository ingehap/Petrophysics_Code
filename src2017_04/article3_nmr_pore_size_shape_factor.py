"""
Article 3: A Laboratory Study of the Link Between NMR Relaxation Data and Pore
           Size In Carbonate Skeletal Grains and Micrite
El-Husseiny, Knight (2017)
Reference: Petrophysics Vol. 58, No. 2 (April 2017), pp. 116-125
DOI: none assigned (this issue predates SPWLA DOI assignment)

NMR T2 relates to pore size through the surface relaxation rate and a geometric
shape factor.  In the fast-diffusion regime 1/T2 = rho*(S/V); writing S/V = alpha/r
gives pore radius r = alpha*rho*T2.  The shape factor alpha is not universal in
carbonates (it differs for micrite, skeletal-grain intergranular, and
intragranular pores), so it is calibrated from the mean-log T2, the surface
relaxivity, and a microscopy pore radius.

Implements:

  - Surface relaxation rate  1/T2 = rho*(S/V)
  - Shape-factor surface-to-volume  S/V = alpha/r
  - Pore radius from T2  r = alpha*rho*T2
  - Shape factor calibrated from the mean-log T2  alpha = r_ML/(rho*T2_ML)

Note: this issue's PDF has a text layer and this article's equations survived as
inline ASCII; the relations below are transcribed.  Relaxivity rho in m/s, T2 in
s, radius in m (alpha = 3 for a sphere).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- relaxation / pore size --------------

def surface_relaxation_rate(rho, s_over_v):
    """Surface relaxation rate  1/T2 = rho*(S/V)  (Eq. 2)."""
    return petrolib.nmr.relaxation_rate(rho=rho, s_over_v=s_over_v)


def shape_factor_sv(alpha, r):
    """Surface-to-volume from the shape factor  S/V = alpha/r  (Eq. 5)."""
    return alpha / np.asarray(r, float)


def pore_radius(alpha, rho, t2):
    """Pore radius from T2  r = alpha*rho*T2  (Eq. 6)."""
    # Kept as the direct alpha*rho*T2 product: the library's
    # shape_factor/surface_to_volume form differs by 1 ULP, which flips this
    # article's reported radius across a rounding boundary (36.75 um).
    return alpha * rho * np.asarray(t2, float)


def shape_factor(r_mean_log, rho, t2_mean_log):
    """Calibrate the shape factor  alpha = r_ML/(rho*T2_ML)  (Eq. 7)."""
    return r_mean_log / (rho * t2_mean_log)


def multiexponential_decay(t, amplitudes, t2s):
    """Multiexponential magnetization decay  M(t) = sum_i A_i*exp(-t/T2_i)  (Eq. 3)."""
    return petrolib.nmr.multiexp_decay(t, amplitudes, t2s)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: NMR Relaxation & Pore Size (shape factor)")
    print("=" * 60)

    # Sphere: S/V = 3/r; surface relaxation rate is rho*S/V
    assert np.isclose(shape_factor_sv(3.0, 1e-5), 3.0 / 1e-5)
    assert surface_relaxation_rate(2.62e-6, 3.0 / 1e-5) > 0

    # Pore radius from T2 and the round-trip shape-factor calibration
    rho = 1.47e-6           # m/s (skeletal grains)
    t2 = 0.5                # s
    alpha = 50.0            # skeletal-grain intergranular
    r = pore_radius(alpha, rho, t2)
    print(f"  pore radius            = {r * 1e6:.1f} um")
    assert np.isclose(shape_factor(r, rho, t2), alpha)

    # Larger shape factor (skeletal grains) gives larger inferred radius than micrite
    assert pore_radius(50.0, rho, t2) > pore_radius(4.0, 2.62e-6, t2)

    # Multiexponential decay starts at the total amplitude and decays
    m = multiexponential_decay(np.linspace(0, 2, 30), [0.6, 0.4], [0.5, 0.05])
    assert np.isclose(m[0], 1.0) and np.all(np.diff(m) < 0)
    print("  PASS")
    return {"radius_um": float(r * 1e6), "alpha": alpha}


if __name__ == "__main__":
    test_all()
