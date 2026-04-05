"""
Effect of Drilling Fluid Filter Cake on Cement Zonal Isolation
================================================================
Based on: Yang et al., "Effect of Water-Based-Mud Filter Cake on Zonal
Isolation at the Interface Between Cement Sheath and Formation",
Petrophysics, Vol. 66, No. 2, April 2025, pp. 318–330.

Implements:
  - DFFC structural layer classification (virtual / underpressure / dense)
  - Second interface (SI) shear strength model
  - SI channeling pressure model
  - Curing time effects on sealing capacity
  - DFFC residue fraction vs. sealing capacity relationship

Reference: https://doi.org/10.30632/PJV66N2-2025a9 (SPWLA)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class FilterCakeLayer(Enum):
    """DFFC structural layer classification (Fengshan et al., 1999)."""
    VIRTUAL = "virtual"           # 3.0–4.0 mm, low structural force
    UNDERPRESSURE = "underpressure"  # 1.0–3.0 mm, medium structural force
    DENSE = "dense"               # 0–1.0 mm, high structural force


@dataclass
class DFFCProperties:
    """Properties of the drilling fluid filter cake."""
    total_thickness_mm: float = 4.0
    residue_fraction: float = 1.0       # Fraction of original DFFC remaining (0–1)
    drilling_fluid_density_g_cm3: float = 1.3

    @property
    def residue_thickness_mm(self) -> float:
        return self.total_thickness_mm * self.residue_fraction

    def classify_layer(self) -> FilterCakeLayer:
        """Classify DFFC layer based on residue thickness."""
        t = self.residue_thickness_mm
        if t > 3.0:
            return FilterCakeLayer.VIRTUAL
        elif t > 1.0:
            return FilterCakeLayer.UNDERPRESSURE
        else:
            return FilterCakeLayer.DENSE


@dataclass
class CementProperties:
    """Properties of the cement slurry/stone."""
    density_g_cm3: float = 1.90
    compressive_strength_24h_MPa: float = 14.0
    water_cement_ratio: float = 0.44


def si_shear_strength(residue_fraction: float,
                      curing_days: float,
                      base_strength_MPa: float = 3.0) -> float:
    """
    Model for second interface (SI) shear strength.

    Based on experimental data from Yang et al. (2025):
    - Shear strength decreases as DFFC residue increases
    - At 15% residue (0.6 mm), strength is 733% higher than at 100% (4.0 mm)
    - Strength increases with curing time, stabilizing at ~30 days

    Parameters
    ----------
    residue_fraction : float
        Fraction of DFFC remaining (0.0 = fully removed, 1.0 = full cake).
    curing_days : float
        Curing time in days.
    base_strength_MPa : float
        Base shear strength with no DFFC residue at 1 day.

    Returns
    -------
    float : SI shear strength in MPa.
    """
    # Residue effect: exponential decay of strength with residue amount
    # At 15% (0.15): strength ≈ 2.5 MPa, at 100% (1.0): strength ≈ 0.3 MPa
    residue_factor = np.exp(-3.0 * residue_fraction)

    # Curing time effect: logarithmic growth, stabilizing at 30 days
    curing_factor = 1.0 + 0.6 * np.log(1.0 + curing_days) / np.log(31.0)
    # Plateaus around 30 days
    if curing_days > 30:
        curing_factor = 1.0 + 0.6

    strength = base_strength_MPa * residue_factor * curing_factor
    return max(strength, 0.01)


def si_channeling_pressure(residue_fraction: float,
                           curing_days: float,
                           base_pressure_MPa: float = 10.0) -> float:
    """
    Model for SI channeling (hydraulic) pressure.

    From Yang et al. (2025):
    - At 15% residue: channeling pressure is 8.53 MPa (1118.6% higher
      than 100% residue)
    - Channeling pressure peaks at ~7 days curing, then decreases due
      to cement stone volume shrinkage creating fluid channels

    Parameters
    ----------
    residue_fraction : float
    curing_days : float
    base_pressure_MPa : float

    Returns
    -------
    float : SI channeling pressure in MPa.
    """
    # Residue effect: similar exponential decay
    residue_factor = np.exp(-3.5 * residue_fraction)

    # Curing time: peaks at ~7 days then decreases
    if curing_days <= 7:
        curing_factor = 1.0 + 0.5 * (curing_days / 7.0)
    else:
        # Decrease after peak due to cement shrinkage
        curing_factor = 1.5 - 0.3 * np.log(curing_days / 7.0)

    curing_factor = max(curing_factor, 0.5)

    pressure = base_pressure_MPa * residue_factor * curing_factor
    return max(pressure, 0.01)


def dffc_structural_force(thickness_mm: float,
                          drilling_fluid_density: float = 1.3) -> float:
    """
    Estimate DFFC structural force from thickness.

    The structural force is measured using the layered scraping method
    described by Sheng et al. (2016). This simplified model gives an
    approximate structural force based on layer thickness.

    Parameters
    ----------
    thickness_mm : float  Total DFFC thickness
    drilling_fluid_density : float  (g/cm³)

    Returns
    -------
    float : Structural force in N (approximate).
    """
    # Virtual layer: very low structural force
    # Dense layer: high structural force
    # Approximation based on exponential relationship
    force = 2.0 * drilling_fluid_density * np.exp(-0.5 * thickness_mm)
    return force


def sealing_capacity_vs_residue(residue_fractions: np.ndarray,
                                curing_days: float = 7.0) -> dict:
    """
    Compute sealing capacity metrics across DFFC residue levels.

    Reproduces the experimental framework of Yang et al. (2025) with
    residue levels of 0%, 15%, 30%, 50%, and 100%.

    Parameters
    ----------
    residue_fractions : np.ndarray
        Array of residue fractions to evaluate.
    curing_days : float

    Returns
    -------
    dict with "shear_strength_MPa" and "channeling_pressure_MPa" arrays.
    """
    shear = np.array([si_shear_strength(rf, curing_days)
                      for rf in residue_fractions])
    channel = np.array([si_channeling_pressure(rf, curing_days)
                        for rf in residue_fractions])
    return {
        "residue_fractions": residue_fractions,
        "shear_strength_MPa": shear,
        "channeling_pressure_MPa": channel,
    }


def curing_time_evolution(residue_fraction: float = 0.15,
                          max_days: int = 60) -> dict:
    """
    Compute sealing capacity evolution with curing time.

    Parameters
    ----------
    residue_fraction : float
    max_days : int

    Returns
    -------
    dict with "days", "shear_strength_MPa", "channeling_pressure_MPa".
    """
    days = np.array([1, 3, 7, 15, 30, 60])
    days = days[days <= max_days]
    shear = np.array([si_shear_strength(residue_fraction, d) for d in days])
    channel = np.array([si_channeling_pressure(residue_fraction, d) for d in days])

    return {
        "days": days,
        "shear_strength_MPa": shear,
        "channeling_pressure_MPa": channel,
    }


def removal_efficiency(original_thickness_mm: float,
                       remaining_thickness_mm: float) -> float:
    """
    Calculate DFFC removal efficiency.

    Parameters
    ----------
    original_thickness_mm : float
    remaining_thickness_mm : float

    Returns
    -------
    float : Removal efficiency (0–1).
    """
    if original_thickness_mm <= 0:
        return 1.0
    return 1.0 - remaining_thickness_mm / original_thickness_mm


def test_all():
    """Test all functions with synthetic data."""
    print("=" * 70)
    print("Testing: filter_cake_isolation (Yang et al., 2025)")
    print("=" * 70)

    # Test DFFC classification
    for frac, expected in [(1.0, FilterCakeLayer.VIRTUAL),
                           (0.5, FilterCakeLayer.UNDERPRESSURE),
                           (0.15, FilterCakeLayer.DENSE)]:
        dffc = DFFCProperties(total_thickness_mm=4.0, residue_fraction=frac)
        layer = dffc.classify_layer()
        print(f"  Residue {frac:.0%} ({dffc.residue_thickness_mm:.1f} mm) → {layer.value}")
        assert layer == expected, f"Expected {expected}, got {layer}"

    # Test shear strength model
    # Paper: at 15% residue, strength ≈ 2.5 MPa; at 100%, ≈ 0.3 MPa
    s_15 = si_shear_strength(0.15, curing_days=7)
    s_100 = si_shear_strength(1.0, curing_days=7)
    ratio = s_15 / s_100
    print(f"  Shear strength: 15% residue = {s_15:.2f} MPa, "
          f"100% residue = {s_100:.2f} MPa, ratio = {ratio:.0f}x")
    assert ratio > 3, "15% residue should have much higher strength"

    # Test channeling pressure
    cp_15 = si_channeling_pressure(0.15, curing_days=7)
    cp_100 = si_channeling_pressure(1.0, curing_days=7)
    print(f"  Channeling pressure: 15% = {cp_15:.2f} MPa, 100% = {cp_100:.2f} MPa")
    assert cp_15 > cp_100

    # Test sealing capacity vs residue
    fractions = np.array([0.0, 0.15, 0.30, 0.50, 1.0])
    result = sealing_capacity_vs_residue(fractions, curing_days=7)
    print(f"  Shear strengths: {result['shear_strength_MPa']}")
    print(f"  Channeling pressures: {result['channeling_pressure_MPa']}")
    # Should be monotonically decreasing
    assert np.all(np.diff(result['shear_strength_MPa']) < 0)

    # Test curing time evolution
    evo = curing_time_evolution(residue_fraction=0.15, max_days=60)
    print(f"  Curing days: {evo['days']}")
    print(f"  Shear strength evolution: {evo['shear_strength_MPa']}")
    # Channeling pressure should peak around 7 days
    peak_idx = np.argmax(evo['channeling_pressure_MPa'])
    print(f"  Channeling pressure peaks at day {evo['days'][peak_idx]}")

    # Test structural force
    for t in [0.5, 2.0, 4.0]:
        f = dffc_structural_force(t)
        print(f"  Structural force at {t:.1f} mm: {f:.2f} N")

    # Test removal efficiency
    eff = removal_efficiency(4.0, 0.6)
    print(f"  Removal efficiency (4.0→0.6 mm): {eff:.1%}")
    assert abs(eff - 0.85) < 0.01

    print("  All tests PASSED.\n")


if __name__ == "__main__":
    test_all()
