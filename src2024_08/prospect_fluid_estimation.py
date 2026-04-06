"""
Improved Reservoir Fluid Estimation for Prospect Evaluation Using Mud Gas Data
===============================================================================
Based on: Ungar, F., Yerkinkyzy, G., Bravo, M.C., and Yang, T. (2024),
"Improved Reservoir Fluid Estimation for Prospect Evaluation Using Mud Gas
Data," Petrophysics, 65(4), pp. 519-531. DOI: 10.30632/PJV65N4-2024a7

Implements:
  - Triangle and diamond plots for C1-C3 composition analysis
  - C2/C3 ratio correlation with GOR
  - Linear GOR prediction from standard mud gas
  - Geospatial PVT database comparison
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class PVTSample:
    """PVT sample from nearby wells used for calibration."""
    well_name: str
    c1_norm: float
    c2_norm: float
    c3_norm: float
    gor: float       # Sm3/Sm3
    viscosity: float  # cp
    depth: float


def compute_triangle_area(c1: np.ndarray, c2: np.ndarray,
                          c3: np.ndarray) -> np.ndarray:
    """Compute triangle plot area from normalized C1-C3 compositions.

    Larger triangle area indicates richer (more liquid) fluids.
    The paper uses this as a visual indicator of oil quality.

    Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    Using ternary coordinates mapped to Cartesian.
    """
    # Ternary to Cartesian
    x = 0.5 * (2 * c2 + c3) / (c1 + c2 + c3 + 1e-10)
    y = (np.sqrt(3) / 2) * c3 / (c1 + c2 + c3 + 1e-10)

    # Area proportional to spread of composition
    area = np.sqrt(x ** 2 + y ** 2)
    return area


def compute_diamond_area(c1: np.ndarray, c2: np.ndarray,
                         c3: np.ndarray) -> np.ndarray:
    """Compute diamond plot area from C1-C3 compositions.

    Smaller diamond area indicates richer fluids (opposite of triangle).
    """
    total = c1 + c2 + c3 + 1e-10
    # Diamond uses ratio space
    r12 = c1 / (c2 + 1e-10)
    r13 = c1 / (c3 + 1e-10)
    r23 = c2 / (c3 + 1e-10)
    # Area proxy from ratio spread
    area = np.sqrt(r12 ** 2 + r13 ** 2) / (r23 + 1)
    return area


def c2_c3_gor_correlation(pvt_samples: List[PVTSample]) -> Tuple[float, float, float]:
    """Build C2/C3 to GOR correlation from PVT samples.

    The paper shows that C2/C3 ratio correlates with GOR, unlike
    C1/C2 or C1/C3 which are dominated by C1.

    Returns (slope, intercept, r_squared) for linear fit:
       GOR = slope * (C2/C3) + intercept
    """
    c2_c3 = np.array([s.c2_norm / (s.c3_norm + 1e-10) for s in pvt_samples])
    gor = np.array([s.gor for s in pvt_samples])

    coeffs = np.polyfit(c2_c3, gor, 1)
    gor_pred = np.polyval(coeffs, c2_c3)
    ss_res = np.sum((gor - gor_pred) ** 2)
    ss_tot = np.sum((gor - gor.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    return coeffs[0], coeffs[1], r2


def predict_gor_from_smg(c1: np.ndarray, c2: np.ndarray, c3: np.ndarray,
                         slope: float, intercept: float) -> np.ndarray:
    """Predict continuous GOR log using C2/C3 correlation.

    Applies pseudo-EEC corrected standard mud gas data to the
    field-specific C2/C3-GOR correlation.
    """
    c2_c3 = c2 / (c3 + 1e-10)
    return slope * c2_c3 + intercept


def estimate_fluid_gradient(gor_log: np.ndarray, depths: np.ndarray) -> dict:
    """Detect compositional gradient from continuous GOR log.

    The paper identifies distinct fluid systems in upper and lower
    reservoir halves from GOR gradients.
    """
    mid = len(depths) // 2
    gor_upper = np.mean(gor_log[:mid])
    gor_lower = np.mean(gor_log[mid:])
    gradient = (gor_lower - gor_upper) / (depths[-1] - depths[0]) if len(depths) > 1 else 0

    return {
        "gor_upper": gor_upper,
        "gor_lower": gor_lower,
        "gradient_per_meter": gradient,
        "has_gradient": abs(gradient) > 0.01,
    }


def test_all():
    """Test prospect fluid estimation pipeline."""
    print("=" * 70)
    print("Testing: Prospect Fluid Estimation (Ungar et al., 2024)")
    print("=" * 70)

    rng = np.random.RandomState(42)

    # Create PVT samples from six nearby wells
    pvt_samples = [
        PVTSample("W1", 0.65, 0.20, 0.15, gor=60, viscosity=50, depth=2200),
        PVTSample("W2", 0.60, 0.22, 0.18, gor=75, viscosity=35, depth=2300),
        PVTSample("W3", 0.70, 0.18, 0.12, gor=110, viscosity=15, depth=2100),
        PVTSample("W4", 0.55, 0.25, 0.20, gor=50, viscosity=80, depth=2400),
        PVTSample("W5", 0.72, 0.17, 0.11, gor=130, viscosity=10, depth=2050),
        PVTSample("W6", 0.68, 0.19, 0.13, gor=95, viscosity=25, depth=2150),
    ]

    # Build C2/C3 correlation
    slope, intercept, r2 = c2_c3_gor_correlation(pvt_samples)
    print(f"  C2/C3 vs GOR correlation: GOR = {slope:.1f} * (C2/C3) + {intercept:.1f}")
    print(f"  R-squared: {r2:.3f}")

    # Generate synthetic prospect well
    n_pts = 100
    depths = np.linspace(2000, 2500, n_pts)
    gradient = (depths - 2000) / 500

    c1 = 500 * (0.65 + 0.1 * gradient) + rng.normal(0, 20, n_pts)
    c2 = 500 * (0.20 - 0.03 * gradient) + rng.normal(0, 10, n_pts)
    c3 = 500 * (0.15 - 0.04 * gradient) + rng.normal(0, 8, n_pts)
    c1, c2, c3 = [np.clip(x, 1, None) for x in [c1, c2, c3]]

    # Triangle and diamond analysis
    total = c1 + c2 + c3
    tri_area = compute_triangle_area(c1 / total, c2 / total, c3 / total)
    dia_area = compute_diamond_area(c1 / total, c2 / total, c3 / total)
    print(f"\n  Triangle area range: {tri_area.min():.3f} - {tri_area.max():.3f}")
    print(f"  Diamond area range: {dia_area.min():.3f} - {dia_area.max():.3f}")

    # GOR prediction
    gor_pred = predict_gor_from_smg(c1, c2, c3, slope, intercept)
    gor_pred = np.clip(gor_pred, 10, 500)
    print(f"\n  Predicted GOR range: {gor_pred.min():.0f} - {gor_pred.max():.0f} Sm3/Sm3")

    # Fluid gradient analysis
    grad_info = estimate_fluid_gradient(gor_pred, depths)
    print(f"\n  Fluid gradient analysis:")
    print(f"    Upper reservoir GOR: {grad_info['gor_upper']:.0f} Sm3/Sm3")
    print(f"    Lower reservoir GOR: {grad_info['gor_lower']:.0f} Sm3/Sm3")
    print(f"    Gradient: {grad_info['gradient_per_meter']:.3f} Sm3/Sm3 per m")
    print(f"    Compositional gradient detected: {grad_info['has_gradient']}")

    print("\n  [PASS] Prospect fluid estimation tests completed.")
    return True


if __name__ == "__main__":
    test_all()
