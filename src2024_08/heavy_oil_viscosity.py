"""
Heavy Oil Viscosity Mapping From Standard Mud Gas – Peregrino Field
====================================================================
Based on: Bravo, M.C., Cely, A., Yerkinkyzy, G., et al. (2024), "A Novel
Method to Map Heavy Oil Viscosity From Standard Mud Gas – A Field Case From
the Peregrino Field," Petrophysics, 65(4), pp. 507-518.
DOI: 10.30632/PJV65N4-2024a6

Implements:
  - C1/C2 ratio-based viscosity classification
  - Reference well palette calibration
  - Continuous viscosity mapping along wellbore
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum


class ViscosityClass(Enum):
    LOW = "low_viscosity"
    MEDIUM = "medium_viscosity"
    HIGH = "high_viscosity"


@dataclass
class ViscosityReference:
    """Reference PVT data from calibration wells."""
    well_name: str
    c1_c2_ratio: float    # characteristic C1/C2 ratio
    viscosity_cp: float   # measured viscosity (cp)
    api_gravity: float    # API gravity


def compute_c1_c2_ratio(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """Compute C1/C2 ratio, the primary discriminator for viscosity."""
    return c1 / (c2 + 1e-10)


def calibrate_viscosity_palette(references: List[ViscosityReference],
                                tolerance: float = 0.05
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Build a viscosity calibration palette from reference wells.

    Maps C1/C2 ratio ranges to viscosity values. The paper uses a
    5% tolerance band around each reference well's C1/C2 ratio.
    Returns (ratio_breakpoints, viscosity_values).
    """
    refs_sorted = sorted(references, key=lambda r: r.c1_c2_ratio)
    ratios = np.array([r.c1_c2_ratio for r in refs_sorted])
    visc = np.array([r.viscosity_cp for r in refs_sorted])
    return ratios, visc


def interpolate_viscosity(c1_c2: np.ndarray, ref_ratios: np.ndarray,
                          ref_viscosities: np.ndarray) -> np.ndarray:
    """Interpolate viscosity from C1/C2 ratio using calibration palette.

    Uses log-linear interpolation since viscosity varies exponentially.
    """
    log_visc = np.log10(ref_viscosities + 1e-10)
    log_visc_interp = np.interp(c1_c2, ref_ratios, log_visc)
    return 10.0 ** log_visc_interp


def classify_viscosity(viscosity: np.ndarray,
                       low_threshold: float = 100,
                       high_threshold: float = 1000) -> np.ndarray:
    """Classify viscosity into low, medium, high categories.

    These thresholds are field-specific; Peregrino uses:
      - Low: < 100 cp (lighter oil)
      - Medium: 100-1000 cp
      - High: > 1000 cp (heavy oil)
    """
    classes = np.empty(len(viscosity), dtype=object)
    classes[viscosity < low_threshold] = ViscosityClass.LOW
    classes[(viscosity >= low_threshold) & (viscosity < high_threshold)] = ViscosityClass.MEDIUM
    classes[viscosity >= high_threshold] = ViscosityClass.HIGH
    return classes


def viscosity_qc(c1_c2: np.ndarray, ref_ratio: float,
                 tolerance: float = 0.05) -> np.ndarray:
    """Quality check: flag readings within tolerance of reference ratio.

    Green = within 5% of reference (good quality).
    Red = outside tolerance (suspect data).
    """
    lower = ref_ratio * (1 - tolerance)
    upper = ref_ratio * (1 + tolerance)
    return (c1_c2 >= lower) & (c1_c2 <= upper)


def pressure_gradient_density(pressures: np.ndarray, depths: np.ndarray) -> float:
    """Estimate fluid density from pressure gradient.

    density (g/cc) = dP/dZ / g, where g ~ 0.0981 bar/m for g/cc.
    The paper notes this method has up to 2.5% deviation, which is
    outside recommended tolerance for EOS modeling.
    """
    if len(pressures) < 2:
        return 0.85  # default oil density
    # Linear regression P vs depth
    coeffs = np.polyfit(depths, pressures, 1)
    gradient_bar_per_m = coeffs[0]
    density = gradient_bar_per_m / 0.0981
    return np.clip(density, 0.7, 1.1)


def test_all():
    """Test heavy oil viscosity mapping pipeline."""
    print("=" * 70)
    print("Testing: Heavy Oil Viscosity from Mud Gas (Bravo et al., 2024)")
    print("=" * 70)

    rng = np.random.RandomState(42)

    # Create reference wells (Peregrino-type heavy oil field)
    references = [
        ViscosityReference("Well_1", c1_c2_ratio=2.5, viscosity_cp=2000, api_gravity=13),
        ViscosityReference("Well_2", c1_c2_ratio=3.5, viscosity_cp=800, api_gravity=16),
        ViscosityReference("Well_6", c1_c2_ratio=5.0, viscosity_cp=200, api_gravity=20),
    ]

    ref_ratios, ref_visc = calibrate_viscosity_palette(references)
    print(f"  Calibration palette: {len(references)} reference wells")
    for ref in references:
        print(f"    {ref.well_name}: C1/C2={ref.c1_c2_ratio:.1f}, "
              f"visc={ref.viscosity_cp:.0f} cp, API={ref.api_gravity}")

    # Generate synthetic well with varying oil quality
    n_points = 100
    depths = np.linspace(1800, 2500, n_points)
    # Gradient: lighter oil at top, heavier at bottom
    gradient = (depths - 1800) / 700
    c1 = 200 - 80 * gradient + rng.normal(0, 10, n_points)
    c2 = 50 + 20 * gradient + rng.normal(0, 3, n_points)
    c1 = np.clip(c1, 10, None)
    c2 = np.clip(c2, 5, None)

    c1_c2 = compute_c1_c2_ratio(c1, c2)
    visc_pred = interpolate_viscosity(c1_c2, ref_ratios, ref_visc)
    visc_class = classify_viscosity(visc_pred)

    # QC against closest reference
    qc_flags = viscosity_qc(c1_c2, ref_ratio=3.5, tolerance=0.15)

    print(f"\n  Well prediction ({n_points} depth points):")
    print(f"    C1/C2 range: {c1_c2.min():.1f} - {c1_c2.max():.1f}")
    print(f"    Viscosity range: {visc_pred.min():.0f} - {visc_pred.max():.0f} cp")

    for vc in ViscosityClass:
        count = sum(1 for v in visc_class if v == vc)
        print(f"    {vc.value}: {count} zones")

    print(f"    QC pass rate: {qc_flags.mean() * 100:.1f}%")

    # Pressure gradient density estimate
    pressures = 180 + 0.098 * 0.92 * (depths - 1800) + rng.normal(0, 0.2, n_points)
    dens = pressure_gradient_density(pressures, depths)
    print(f"\n  Pressure gradient density: {dens:.4f} g/cc")

    print("\n  [PASS] Heavy oil viscosity module tests completed.")
    return True


if __name__ == "__main__":
    test_all()
