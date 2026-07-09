"""
Article 2: Empirical Relation for Capillary Pressure in Shale
Alipour K., Kasha, Sakhaee-Pour, Sadooni, Al-Kuwari (2022)
DOI: 10.30632/PJV63N5-2022a2

Proposes a three-parameter Pc(Sw) model for shale that admits a non-zero
entry pressure (unlike van Genuchten) and a non-plateau trend (unlike
Brooks-Corey).  Implements all three models, fits each to a synthetic
MICP dataset, and reports comparative R^2 and MSE statistics.

  Young-Laplace                Pc = 4 gamma cos(theta) / d        (Eq. 1)
  Brooks-Corey                 Pc = pe * (Sw*) ^ (-1/lambda)      (Eq. 3)
  van Genuchten                Pc = (1/alpha) * (Sw*^(-1/m) - 1)^(1/n)  (Eq. 5)
  Proposed (Alipour et al.)    Pc = pe + alpha1 * ((1 - Sw*) / Sw*)^alpha2  (Eq. 6)
  MSE                          MSE = sum (Y_pred - Y_obs)^2 / N    (Eq. 7)

The proposed form recovers a non-zero entry pressure pe while keeping
the slope (alpha1) and curvature (alpha2) separately tunable.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib
from scipy.optimize import curve_fit


# ------------------------------------------------ Young-Laplace ---------

def young_laplace(d_m, gamma_N_m=0.072, theta_deg=180.0):
    """Pc = 4 gamma cos(theta) / d   (Eq. 1)."""
    # Diameter form: 4*gamma*cos/d == 2*gamma*cos/r with r = d/2 (signed cos).
    return petrolib.capillary_pressure.young_laplace_pc(
        d_m / 2.0, sigma=gamma_N_m, theta_deg=theta_deg, absolute=False)


# ------------------------------------------------ normalised saturation -

def sw_star(sw, swirr):
    """Sw* = (Sw - Swirr) / (1 - Swirr) (Eq. 4)."""
    return np.clip((sw - swirr) / (1.0 - swirr), 1e-4, 1.0)


# ------------------------------------------------ Brooks-Corey ---------

def brooks_corey(sw, pe, lam, swirr):
    # sw_star is this article's normalized+clipped saturation; delegate the
    # Pe*Se^(-1/lam) kernel (Se pre-normalized -> swirr=0).
    return petrolib.capillary_pressure.brooks_corey_pc(
        sw_star(sw, swirr), pc_entry=pe, lam=lam, swirr=0.0)


# ------------------------------------------------ van Genuchten --------

def van_genuchten(sw, alpha, m, n, swirr):
    return (1.0 / alpha) * (sw_star(sw, swirr) ** (-1.0 / m) - 1.0) ** (1.0 / n)


# ------------------------------------------------ proposed Eq. 6 -------

def proposed_alipour(sw, pe, alpha1, alpha2, swirr):
    s = sw_star(sw, swirr)
    return pe + alpha1 * ((1.0 - s) / s) ** alpha2


# ------------------------------------------------ MSE / R^2 ------------

def mse(y_pred, y_obs):
    return float(np.mean((y_pred - y_obs) ** 2))


def r2(y_pred, y_obs):
    # HAZARD (LIBRARY_MERGE_PLAN.md section 9): this article's historical
    # argument order takes the PREDICTIONS first.  The canonical r2_score
    # takes (y_true, y_pred), so the arguments are mapped explicitly here —
    # a positional migration would compute ss_tot from the predictions.
    return petrolib.ml_stats.r2_score(y_obs, y_pred)


# ------------------------------------------------ fit helpers ----------

def fit_brooks_corey(sw, pc_obs):
    swirr = 0.02
    def f(s, pe, lam):
        return brooks_corey(s, pe, lam, swirr)
    p, _ = curve_fit(f, sw, pc_obs,
                     p0=[1.0, 0.5], bounds=([0.0, 0.05], [1e4, 5.0]),
                     maxfev=5000)
    return p, swirr


def fit_van_genuchten(sw, pc_obs):
    swirr = 0.02
    def f(s, alpha, m, n):
        return van_genuchten(s, alpha, m, n, swirr)
    p, _ = curve_fit(f, sw, pc_obs,
                     p0=[0.1, 0.5, 2.0],
                     bounds=([1e-4, 0.05, 1.05], [10.0, 0.99, 10.0]),
                     maxfev=10000)
    return p, swirr


def fit_proposed(sw, pc_obs):
    swirr = 0.02
    def f(s, pe, a1, a2):
        return proposed_alipour(s, pe, a1, a2, swirr)
    p, _ = curve_fit(f, sw, pc_obs,
                     p0=[pc_obs.min(), 1.0, 1.0],
                     bounds=([0.0, 1e-3, 0.05], [1e4, 1e4, 10.0]),
                     maxfev=10000)
    return p, swirr


# ------------------------------------------------ synthetic data ------

def make_micp_dataset(pe_true=2.0, alpha1_true=8.0, alpha2_true=1.5,
                      swirr=0.02, n=25, noise=0.05, seed=0):
    """Generate synthetic Pc-Sw points from the proposed model + noise."""
    rng = np.random.default_rng(seed)
    sw = np.linspace(swirr + 0.02, 0.99, n)
    pc = proposed_alipour(sw, pe_true, alpha1_true, alpha2_true, swirr)
    pc *= (1.0 + noise * rng.standard_normal(n))
    return sw, pc


# ------------------------------------------------ tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Shale Capillary-Pressure Model Comparison")
    print("=" * 60)

    sw, pc = make_micp_dataset()
    print(f"  Synthetic dataset: {len(sw)} points, "
          f"Pc range {pc.min():.2f} - {pc.max():.2f} MPa")

    p_bc, swirr_bc = fit_brooks_corey(sw, pc)
    p_vg, swirr_vg = fit_van_genuchten(sw, pc)
    p_pa, swirr_pa = fit_proposed(sw, pc)

    pc_bc = brooks_corey(sw, *p_bc, swirr_bc)
    pc_vg = van_genuchten(sw, *p_vg, swirr_vg)
    pc_pa = proposed_alipour(sw, *p_pa, swirr_pa)

    print(f"  Brooks-Corey    pe={p_bc[0]:6.3f}  lam={p_bc[1]:.3f}  "
          f"R2={r2(pc_bc, pc):6.3f}  MSE={mse(pc_bc, pc):7.3e}")
    print(f"  van Genuchten   alpha={p_vg[0]:.3f}  m={p_vg[1]:.3f}  n={p_vg[2]:.3f}  "
          f"R2={r2(pc_vg, pc):6.3f}  MSE={mse(pc_vg, pc):7.3e}")
    print(f"  Proposed        pe={p_pa[0]:6.3f}  a1={p_pa[1]:.3f}  a2={p_pa[2]:.3f}  "
          f"R2={r2(pc_pa, pc):6.3f}  MSE={mse(pc_pa, pc):7.3e}")

    # The proposed model is the true generator, so it must dominate the
    # two-parameter Brooks-Corey on this dataset.
    assert mse(pc_pa, pc) < mse(pc_bc, pc), \
        "Proposed model must beat Brooks-Corey on its own dataset"
    print("  PASS")
    return {"r2_brooks": r2(pc_bc, pc),
            "r2_vg": r2(pc_vg, pc),
            "r2_proposed": r2(pc_pa, pc)}


if __name__ == "__main__":
    test_all()
