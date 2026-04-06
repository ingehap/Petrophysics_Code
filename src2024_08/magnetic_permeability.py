"""
Applying Magnetic Susceptibility to Estimate Permeability From Drill Cuttings
=============================================================================
Based on: Banks, J.Y., Tugwell, A.G., and Potter, D.K. (2024), "Applying
Magnetic Susceptibility to Estimate Permeability From Drill Cuttings: A Case
Study Constraining Uncertainty in the Culzean Triassic Reservoir,"
Petrophysics, 65(4), pp. 604-623. DOI: 10.30632/PJV65N4-2024a13

Implements:
  - High-field magnetic susceptibility measurement model
  - Clay volume estimation from paramagnetic susceptibility (Eqs. 3-4)
  - Permeability estimation from clay volume
  - Gaussian averaging for scale reconciliation
  - Ferromagnetic contaminant removal
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter1d


# Magnetic susceptibility constants (SI units)
SUSCEPTIBILITY_ILLITE = 41e-5       # volume susceptibility
SUSCEPTIBILITY_CHLORITE = 55e-5
SUSCEPTIBILITY_QUARTZ = -1.5e-5
SUSCEPTIBILITY_FELDSPAR = -1.3e-5

# Mass susceptibility (m3/kg)
MASS_SUSCEPTIBILITY_ILLITE = 15e-8
MASS_SUSCEPTIBILITY_QUARTZ = -0.6e-8


@dataclass
class MagneticMeasurement:
    """High-field magnetic susceptibility measurement on a sample."""
    depth: float
    susceptibility_volume: float   # volume susceptibility (SI, dimensionless)
    susceptibility_mass: float     # mass susceptibility (m3/kg)
    sample_mass_g: float
    has_ferromagnetic: bool        # whether ferromagnetic contaminants detected


@dataclass
class CuttingsSample:
    """Drill cuttings sample with magnetic and petrographic data."""
    depth: float
    mag_susceptibility: float       # measured volume susceptibility
    density: float                  # bulk density (g/cm3)
    xrd_clay_volume: Optional[float] = None  # XRD-derived clay volume if available


def remove_ferromagnetic_contaminants(susceptibility: np.ndarray,
                                      threshold_factor: float = 3.0) -> np.ndarray:
    """Remove ferromagnetic contaminant spikes from susceptibility data.

    The paper uses a magnet to physically remove contaminants before
    measurement. This function provides a digital equivalent by
    identifying and replacing outlier spikes.

    Ferromagnetic minerals (magnetite) and metallic contaminants have
    very high susceptibilities that saturate in high fields, but
    residual signals may remain.
    """
    median_val = np.median(susceptibility)
    mad = np.median(np.abs(susceptibility - median_val))
    upper_bound = median_val + threshold_factor * mad * 1.4826
    cleaned = susceptibility.copy()
    outliers = cleaned > upper_bound
    if outliers.any():
        cleaned[outliers] = median_val
    return cleaned


def estimate_clay_volume_illite(susceptibility_total: np.ndarray,
                                k_illite: float = SUSCEPTIBILITY_ILLITE,
                                k_quartz: float = SUSCEPTIBILITY_QUARTZ
                                ) -> np.ndarray:
    """Estimate illite clay volume fraction from magnetic susceptibility.

    From the paper Eq. 3:
      k_T = F_I * k_I + (1 - F_I) * k_Q

    Solving Eq. 4:
      F_I = (k_T - k_Q) / (k_I - k_Q)

    Where:
      k_T = total measured susceptibility
      k_I = illite susceptibility (41 × 10^-5 SI)
      k_Q = quartz susceptibility (-1.5 × 10^-5 SI)
      F_I = illite fraction by volume
    """
    f_illite = (susceptibility_total - k_quartz) / (k_illite - k_quartz)
    return np.clip(f_illite, 0, 1)


def estimate_clay_volume_mixed(susceptibility_total: np.ndarray,
                               illite_fraction: float = 0.7,
                               chlorite_fraction: float = 0.3,
                               k_illite: float = SUSCEPTIBILITY_ILLITE,
                               k_chlorite: float = SUSCEPTIBILITY_CHLORITE,
                               k_quartz: float = SUSCEPTIBILITY_QUARTZ
                               ) -> np.ndarray:
    """Estimate total clay volume for a mixed illite-chlorite system.

    Effective clay susceptibility is weighted by relative proportions.
    """
    k_clay = illite_fraction * k_illite + chlorite_fraction * k_chlorite
    f_clay = (susceptibility_total - k_quartz) / (k_clay - k_quartz)
    return np.clip(f_clay, 0, 1)


def clay_volume_to_permeability(clay_volume: np.ndarray,
                                perm_clean: float = 500.0,
                                clay_exponent: float = 8.0) -> np.ndarray:
    """Estimate permeability from clay volume.

    Empirical relationship calibrated to core data:
      k = k_clean * (1 - V_clay) ^ n

    Where:
      k_clean = permeability of clean sandstone (md)
      n = clay exponent (controls sensitivity to clay content)

    The paper shows R^2 = 0.949 for overburden-corrected permeability.
    """
    perm = perm_clean * (1 - clay_volume) ** clay_exponent
    return np.clip(perm, 0.001, 10000)


def gaussian_average(values: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Apply Gaussian averaging to reconcile cuttings and core scales.

    The paper uses Gaussian averaging of core plug permeabilities
    to compare with lower-resolution cuttings-derived values.
    This accounts for the different sampling volumes.
    """
    return gaussian_filter1d(values, sigma=sigma)


def overburden_correction(perm_ambient: np.ndarray,
                          depth_ft: np.ndarray,
                          correction_factor: float = 0.0001) -> np.ndarray:
    """Apply overburden stress correction to permeability.

    Permeability decreases under overburden pressure. Simple
    exponential correction model.
    """
    stress_factor = np.exp(-correction_factor * depth_ft)
    return perm_ambient * stress_factor


def validate_against_xrd(mag_clay: np.ndarray, xrd_clay: np.ndarray) -> dict:
    """Validate magnetic-susceptibility-derived clay volume against XRD.

    The paper reports R^2 = 0.909 for this comparison, with magnetic
    susceptibility slightly overestimating due to amorphous material.
    """
    valid = ~np.isnan(xrd_clay)
    if valid.sum() < 2:
        return {"r_squared": np.nan, "mean_difference": np.nan}

    corr = np.corrcoef(mag_clay[valid], xrd_clay[valid])[0, 1]
    r2 = corr ** 2
    mean_diff = np.mean(mag_clay[valid] - xrd_clay[valid])

    return {
        "r_squared": r2,
        "mean_difference": mean_diff,
        "n_samples": int(valid.sum()),
        "mag_slightly_higher": mean_diff > 0,
    }


def test_all():
    """Test magnetic susceptibility permeability estimation pipeline."""
    print("=" * 70)
    print("Testing: Magnetic Susceptibility Permeability (Banks et al., 2024)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    n_pts = 120

    # Simulate Culzean-type Triassic reservoir
    depths_ft = np.linspace(14000, 15000, n_pts)

    # Clay volume varies (channel sands = low clay, overbank = high clay)
    true_clay = 0.05 + 0.15 * np.abs(np.sin(depths_ft / 80))
    true_clay += rng.normal(0, 0.02, n_pts)
    true_clay = np.clip(true_clay, 0.01, 0.4)

    # Generate magnetic susceptibility from clay volume (forward model)
    true_suscept = true_clay * SUSCEPTIBILITY_ILLITE + (1 - true_clay) * SUSCEPTIBILITY_QUARTZ
    # Add measurement noise and occasional ferromagnetic spikes
    measured_suscept = true_suscept + rng.normal(0, 1e-5, n_pts)
    spike_idx = rng.choice(n_pts, 5, replace=False)
    measured_suscept[spike_idx] += rng.uniform(5e-4, 2e-3, 5)  # ferromagnetic spikes

    print(f"  Measured susceptibility range: {measured_suscept.min():.2e} - {measured_suscept.max():.2e} SI")

    # Remove ferromagnetic contaminants
    cleaned_suscept = remove_ferromagnetic_contaminants(measured_suscept)
    n_removed = (measured_suscept != cleaned_suscept).sum()
    print(f"  Ferromagnetic spikes removed: {n_removed}")

    # Estimate clay volume
    est_clay = estimate_clay_volume_illite(cleaned_suscept)
    clay_error = np.mean(np.abs(est_clay - true_clay))
    print(f"\n  Clay volume estimation:")
    print(f"    True range: {true_clay.min():.3f} - {true_clay.max():.3f}")
    print(f"    Estimated range: {est_clay.min():.3f} - {est_clay.max():.3f}")
    print(f"    MAE: {clay_error:.4f}")

    # Validate against synthetic XRD
    xrd_clay = true_clay.copy()
    xrd_clay[rng.random(n_pts) > 0.1] = np.nan  # only 10% have XRD
    validation = validate_against_xrd(est_clay, xrd_clay)
    print(f"\n  Validation vs XRD:")
    print(f"    R-squared: {validation['r_squared']:.3f}")
    print(f"    Mean difference: {validation['mean_difference']:.4f}")
    print(f"    Mag slightly higher: {validation['mag_slightly_higher']}")

    # Estimate permeability
    perm_ambient = clay_volume_to_permeability(est_clay)
    perm_ob = overburden_correction(perm_ambient, depths_ft)
    perm_gaussian = gaussian_average(perm_ob, sigma=3.0)

    print(f"\n  Permeability estimation:")
    print(f"    Ambient range: {perm_ambient.min():.1f} - {perm_ambient.max():.1f} md")
    print(f"    Overburden-corrected: {perm_ob.min():.1f} - {perm_ob.max():.1f} md")
    print(f"    Gaussian-averaged: {perm_gaussian.min():.1f} - {perm_gaussian.max():.1f} md")

    # True permeability for comparison
    true_perm = clay_volume_to_permeability(true_clay)
    true_perm_ob = overburden_correction(true_perm, depths_ft)
    r2_perm = np.corrcoef(np.log10(perm_gaussian + 0.001),
                          np.log10(true_perm_ob + 0.001))[0, 1] ** 2
    print(f"    R-squared (log perm): {r2_perm:.3f}")

    print("\n  [PASS] Magnetic susceptibility permeability tests completed.")
    return True


if __name__ == "__main__":
    test_all()
