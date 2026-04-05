#!/usr/bin/env python3
"""
Cross-Calibrated Permeabilities: A Reliable and Physically Coherent Methodology
================================================================================
Implements the methodology from:
  Sifontes, V., Leal, L.A., Farfan, M., and Mussio Arias, A., 2025,
  "Cross-Calibrated Permeabilities: A Reliable and Physically Coherent
  Methodology,"
  Petrophysics, Vol. 66, No. 3, pp. 449–466.

Key ideas implemented:
  - Timur permeability equation (Timur, 1968) – Eq. 1.
  - Coates permeability equation (Coates, 1973) – Eqs. 2-3.
  - Four-step cross-calibration methodology.
  - SwXCal correlation for assisted calibration.
  - Pore-throat classification (nano to mega).

References:
  Tixier, M.P., 1949. Wyllie, M.R.J. and Rose, W., 1950.
  Timur, A., 1968. Coates, G.R., 1973.
  Coates, G.R. and Denoo, S., 1981.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# ============================================================
# Permeability equations
# ============================================================

def timur_permeability(phi: np.ndarray, sw: np.ndarray,
                       a: float = 4800.0,
                       b: float = 4.4,
                       c: float = 2.0) -> np.ndarray:
    """
    Timur permeability equation (Eq. 1 of paper).

    K = a · φ^b / Sw^c

    Parameters
    ----------
    phi : porosity (fraction)
    sw  : water saturation (fraction)
    a, b, c : calibration parameters

    Returns
    -------
    Permeability in mD
    """
    phi = np.clip(np.asarray(phi, dtype=float), 1e-6, 1.0)
    sw = np.clip(np.asarray(sw, dtype=float), 1e-6, 1.0)
    return a * phi ** b / sw ** c


def coates_permeability(phi: np.ndarray, sw: np.ndarray,
                         C: float = 70.0,
                         phi_exp: float = 2.0,
                         k_exp: float = 2.0) -> np.ndarray:
    """
    Coates permeability equation (Eq. 2 of paper).

    K = C · φ^phi_exp · ((1-Sw)/Sw)^k_exp

    Parameters
    ----------
    phi : porosity (fraction)
    sw  : water saturation (fraction)
    C   : constant related to pore-throat size
    phi_exp : porosity exponent
    k_exp   : saturation exponent

    Returns
    -------
    Permeability in mD
    """
    phi = np.clip(np.asarray(phi, dtype=float), 1e-6, 1.0)
    sw = np.clip(np.asarray(sw, dtype=float), 1e-6, 0.9999)
    bvi_ratio = (1.0 - sw) / sw
    return C * phi ** phi_exp * bvi_ratio ** k_exp


# ============================================================
# Pore-throat classification
# ============================================================

@dataclass
class PoreThroatClass:
    """Pore-throat classification with Coates parameters (Table 1 of paper)."""
    name: str
    C: float
    phi_exp: float
    k_exp: float
    rt_range: Tuple[float, float]  # pore-throat radius range (microns)


PORE_THROAT_CLASSES = [
    PoreThroatClass("Nano",  0.5,   0.5, 3.0,  (0.001, 0.1)),
    PoreThroatClass("Micro", 5.0,   1.0, 2.5,  (0.1, 1.0)),
    PoreThroatClass("Meso",  50.0,  1.5, 2.0,  (1.0, 10.0)),
    PoreThroatClass("Macro", 400.0, 1.0, 1.42, (10.0, 100.0)),
    PoreThroatClass("Mega",  5000., 2.0, 1.0,  (100.0, 1000.0)),
]


def classify_pore_throat(k_phi: float) -> str:
    """
    Classify pore-throat size from K/φ ratio (proxy for pore-throat radius).

    K/φ ranges: nano < 0.01, micro 0.01-1, meso 1-100, macro 100-10000, mega >10000
    """
    if k_phi < 0.01:
        return "Nano"
    elif k_phi < 1.0:
        return "Micro"
    elif k_phi < 100:
        return "Meso"
    elif k_phi < 10000:
        return "Macro"
    else:
        return "Mega"


# ============================================================
# Cross-calibration methodology
# ============================================================

def step1_manual_calibration(phi_core: np.ndarray,
                              sw_core: np.ndarray,
                              k_core: np.ndarray) -> dict:
    """
    Step 1: Manual calibration at core level.
    Fits the Coates equation to core data using least-squares
    in log-space.

    Returns dict with fitted C, phi_exp, k_exp and R².
    """
    phi = np.clip(phi_core, 1e-6, 1.0)
    sw = np.clip(sw_core, 1e-6, 0.9999)
    k = np.clip(k_core, 1e-6, None)

    # log(K) = log(C) + phi_exp·log(φ) + k_exp·log((1-Sw)/Sw)
    log_k = np.log10(k)
    log_phi = np.log10(phi)
    log_bvi = np.log10((1.0 - sw) / sw)

    # Design matrix  [1, log_phi, log_bvi]
    A = np.column_stack([np.ones_like(log_k), log_phi, log_bvi])
    params, residuals, _, _ = np.linalg.lstsq(A, log_k, rcond=None)

    C = 10.0 ** params[0]
    phi_exp = params[1]
    k_exp = params[2]

    k_pred = coates_permeability(phi, sw, C, phi_exp, k_exp)
    ss_res = np.sum((np.log10(k) - np.log10(k_pred)) ** 2)
    ss_tot = np.sum((np.log10(k) - np.mean(np.log10(k))) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    return {"C": C, "phi_exp": phi_exp, "k_exp": k_exp, "R2": r2}


def step2_welllog_calibration(phi_log: np.ndarray,
                               sw_log: np.ndarray,
                               k_core: np.ndarray,
                               phi_core: np.ndarray,
                               depth_log: np.ndarray,
                               depth_core: np.ndarray) -> dict:
    """
    Step 2: Cross-calibration at well-log level.
    Interpolates core data to log depths and re-calibrates.
    """
    # Interpolate core K at log depths
    k_at_log = np.interp(depth_log, depth_core, k_core)
    phi_at_log = np.interp(depth_log, depth_core, phi_core)

    # Use valid points only
    valid = (k_at_log > 0) & (phi_log > 0.01) & (sw_log > 0.01) & (sw_log < 0.99)
    if np.sum(valid) < 5:
        return {"C": 70, "phi_exp": 2.0, "k_exp": 2.0, "R2": 0.0}

    return step1_manual_calibration(phi_log[valid], sw_log[valid], k_at_log[valid])


# ============================================================
# SwXCal assisted calibration
# ============================================================

def sw_xcal(phi: np.ndarray, k: np.ndarray,
            C_sw: float = 1.0, K_sw: float = 0.5) -> np.ndarray:
    """
    SwXCal correlation for water saturation estimation
    from permeability and porosity.

    Sw_XCal = 1 / (1 + C_sw · (K/φ)^K_sw)

    This is the unique correlation developed in the paper for
    calibration in absence of core Sw data.
    """
    phi = np.clip(phi, 1e-6, 1.0)
    k = np.clip(k, 1e-6, None)
    k_phi = k / phi
    return 1.0 / (1.0 + C_sw * k_phi ** K_sw)


def step3_swxcal_calibration(phi: np.ndarray,
                              sw_log: np.ndarray,
                              k_init: np.ndarray) -> Tuple[float, float]:
    """
    Step 3: Adjust SwXCal parameters (C_sw, K_sw) until
    SwXCal matches the log-derived Sw.

    Returns optimised (C_sw, K_sw).
    """
    from scipy.optimize import minimize

    def objective(params):
        C_sw, K_sw = params
        sw_pred = sw_xcal(phi, k_init, C_sw, K_sw)
        return np.sum((sw_pred - sw_log) ** 2)

    try:
        result = minimize(objective, x0=[1.0, 0.5],
                          bounds=[(0.01, 100), (0.1, 3.0)],
                          method="L-BFGS-B")
        return result.x[0], result.x[1]
    except Exception:
        # Fallback: grid search
        best_err = np.inf
        best_c, best_k = 1.0, 0.5
        for c in np.logspace(-1, 2, 20):
            for k in np.linspace(0.1, 3, 15):
                sw_pred = sw_xcal(phi, k_init, c, k)
                err = np.sum((sw_pred - sw_log) ** 2)
                if err < best_err:
                    best_err = err
                    best_c, best_k = c, k
        return best_c, best_k


# ============================================================
# Porosity partitioning for carbonates
# ============================================================

def porosity_partitioning(phi_total: np.ndarray,
                           phi_matrix_max: float = 0.08) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate matrix porosity from moldic/vuggy porosity
    for multimodal carbonate reservoirs.

    Returns (phi_matrix, phi_secondary).
    """
    phi_matrix = np.minimum(phi_total, phi_matrix_max)
    phi_secondary = np.clip(phi_total - phi_matrix_max, 0, None)
    return phi_matrix, phi_secondary


# ============================================================
# Test
# ============================================================

def test_all():
    """Test all functions with synthetic data."""
    print("=" * 60)
    print("Testing cross_calibrated_permeability module (Sifontes et al., 2025)")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n = 80

    # Synthetic core data
    phi = 0.05 + 0.25 * rng.rand(n)
    sw = 0.2 + 0.6 * rng.rand(n)
    # True Coates with C=400, phi_exp=1.0, k_exp=1.42
    k_true = 400 * phi ** 1.0 * ((1 - sw) / sw) ** 1.42
    k_noisy = k_true * np.exp(0.3 * rng.randn(n))

    # 1. Timur
    k_timur = timur_permeability(phi, sw)
    print(f"\n1) Timur: K range = {k_timur.min():.2f} – {k_timur.max():.1f} mD")

    # 2. Coates
    k_coates = coates_permeability(phi, sw, C=400, phi_exp=1.0, k_exp=1.42)
    print(f"   Coates: K range = {k_coates.min():.2f} – {k_coates.max():.1f} mD")

    # 3. Step 1: Manual calibration
    result = step1_manual_calibration(phi, sw, k_noisy)
    print(f"\n3) Step 1 calibration: C={result['C']:.1f}, "
          f"φ_exp={result['phi_exp']:.2f}, K_exp={result['k_exp']:.2f}, "
          f"R²={result['R2']:.3f}")
    assert result['R2'] > 0.5, "Calibration R² too low"

    # 4. Pore-throat classification
    k_phi = k_noisy / phi
    for i in range(0, n, 20):
        cls = classify_pore_throat(k_phi[i])
        print(f"\n4) K/φ={k_phi[i]:.1f} → {cls}")

    # 5. SwXCal
    sw_pred = sw_xcal(phi, k_noisy, C_sw=1.0, K_sw=0.5)
    print(f"\n5) SwXCal: Sw range = {sw_pred.min():.3f} – {sw_pred.max():.3f}")

    # 6. SwXCal calibration
    c_opt, k_opt = step3_swxcal_calibration(phi, sw, k_noisy)
    print(f"\n6) Optimised SwXCal: C_sw={c_opt:.3f}, K_sw={k_opt:.3f}")

    # 7. Step 2: Well-log calibration
    depth_log = np.linspace(1000, 1050, n)
    depth_core = depth_log.copy()
    result2 = step2_welllog_calibration(phi, sw, k_noisy, phi,
                                         depth_log, depth_core)
    print(f"\n7) Step 2 well-log calibration: R²={result2['R2']:.3f}")

    # 8. Carbonate porosity partitioning
    phi_m, phi_s = porosity_partitioning(np.array([0.05, 0.12, 0.25]))
    print(f"\n8) Porosity partitioning: matrix={phi_m}, secondary={phi_s}")

    print("\n✓ All cross_calibrated_permeability tests passed.\n")


if __name__ == "__main__":
    test_all()
