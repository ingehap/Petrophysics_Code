#!/usr/bin/env python3
"""
Permeability Anisotropy in Carbonates via Digital Rock Petrophysics
====================================================================
Based on: Silva Junior et al. (2024), "Permeability Anisotropy in
Brazilian Presalt Carbonates at Core Scale Using Digital Rock
Petrophysics", Petrophysics, Vol. 65, No. 5, pp. 711-738.

Implements:
- Hydraulic flow unit (HFU) classification
- Flow zone indicator (FZI) and reservoir quality index (RQI)
- Permeability upscaling (arithmetic, harmonic, geometric means)
- Kv/Kh anisotropy ratio computation at various vertical windows
- Facies-based permeability statistics

Reference: DOI:10.30632/PJV65N5-2024a4
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


def reservoir_quality_index(k_md: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute the Reservoir Quality Index (RQI).

    RQI = 0.0314 * sqrt(k / phi)   (Amaefule et al., 1993)

    Parameters
    ----------
    k_md : array-like
        Permeability (md).
    phi : array-like
        Porosity (fraction).

    Returns
    -------
    np.ndarray
        RQI (µm).
    """
    k = np.asarray(k_md, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return 0.0314 * np.sqrt(np.maximum(k, 1e-6) / np.maximum(phi, 1e-6))


def flow_zone_indicator(k_md: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute the Flow Zone Indicator (FZI).

    FZI = RQI / (phi / (1 - phi))

    Parameters
    ----------
    k_md : array-like
        Permeability (md).
    phi : array-like
        Porosity (fraction).

    Returns
    -------
    np.ndarray
        FZI (µm).
    """
    rqi = reservoir_quality_index(k_md, phi)
    phi_arr = np.asarray(phi, dtype=float)
    phi_z = phi_arr / (1.0 - phi_arr)
    return np.divide(rqi, phi_z, out=np.zeros_like(rqi),
                     where=phi_z > 1e-8)


def classify_hfu(fzi: np.ndarray, n_units: int = 4
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify samples into Hydraulic Flow Units based on FZI
    using log-spaced thresholds.

    Parameters
    ----------
    fzi : array-like
        Flow Zone Indicator values.
    n_units : int
        Number of HFU classes.

    Returns
    -------
    labels : np.ndarray
        HFU class labels (0 to n_units-1).
    thresholds : np.ndarray
        FZI threshold values used for classification.
    """
    fzi = np.asarray(fzi, dtype=float)
    log_fzi = np.log10(np.maximum(fzi, 1e-6))
    thresholds = np.linspace(log_fzi.min(), log_fzi.max(), n_units + 1)
    labels = np.digitize(log_fzi, thresholds[1:-1])
    return labels, 10.0 ** thresholds


def upscale_permeability_arithmetic(k: np.ndarray, weights: Optional[np.ndarray] = None
                                    ) -> float:
    """Arithmetic mean permeability (horizontal flow parallel to layers)."""
    k = np.asarray(k, dtype=float)
    if weights is None:
        return float(np.mean(k))
    return float(np.average(k, weights=weights))


def upscale_permeability_harmonic(k: np.ndarray, weights: Optional[np.ndarray] = None
                                  ) -> float:
    """Harmonic mean permeability (vertical flow perpendicular to layers)."""
    k = np.asarray(k, dtype=float)
    k_safe = np.maximum(k, 1e-6)
    if weights is None:
        return float(len(k_safe) / np.sum(1.0 / k_safe))
    return float(np.sum(weights) / np.sum(weights / k_safe))


def upscale_permeability_geometric(k: np.ndarray, weights: Optional[np.ndarray] = None
                                   ) -> float:
    """Geometric mean permeability."""
    k = np.asarray(k, dtype=float)
    k_safe = np.maximum(k, 1e-6)
    if weights is None:
        return float(np.exp(np.mean(np.log(k_safe))))
    return float(np.exp(np.average(np.log(k_safe), weights=weights)))


def kv_kh_ratio(k_vertical: np.ndarray, k_horizontal: np.ndarray,
                window_sizes: Optional[List[int]] = None
                ) -> Dict[int, float]:
    """
    Compute Kv/Kh anisotropy ratios at various vertical investigation
    windows by upscaling local permeability values.

    Kh = arithmetic mean over window (parallel to bedding)
    Kv = harmonic mean over window (perpendicular to bedding)

    Parameters
    ----------
    k_vertical : np.ndarray
        Vertical permeability values along the core (md).
    k_horizontal : np.ndarray
        Horizontal permeability values along the core (md).
    window_sizes : list of int, optional
        Window sizes (number of samples). Default: [1, 5, 10, 20].

    Returns
    -------
    dict
        {window_size: median Kv/Kh ratio}
    """
    if window_sizes is None:
        window_sizes = [1, 5, 10, 20]

    results = {}
    n = len(k_vertical)

    for ws in window_sizes:
        ratios = []
        for start in range(0, n - ws + 1, max(1, ws // 2)):
            end = start + ws
            kv_up = upscale_permeability_harmonic(k_vertical[start:end])
            kh_up = upscale_permeability_arithmetic(k_horizontal[start:end])
            if kh_up > 0:
                ratios.append(kv_up / kh_up)
        results[ws] = float(np.median(ratios)) if ratios else np.nan

    return results


def facies_permeability_stats(k: np.ndarray, facies: np.ndarray
                              ) -> Dict[str, Dict[str, float]]:
    """
    Compute permeability statistics by facies.

    Parameters
    ----------
    k : np.ndarray
        Permeability array (md).
    facies : np.ndarray
        Facies label array (same length as k).

    Returns
    -------
    dict
        {facies_label: {'mean': ..., 'median': ..., 'std': ..., 'p10': ..., 'p90': ...}}
    """
    stats = {}
    for f in np.unique(facies):
        mask = facies == f
        kf = k[mask]
        stats[str(f)] = {
            'mean': float(np.mean(kf)),
            'median': float(np.median(kf)),
            'std': float(np.std(kf)),
            'p10': float(np.percentile(kf, 10)),
            'p90': float(np.percentile(kf, 90)),
            'count': int(mask.sum())
        }
    return stats


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Permeability Anisotropy Module Demo ===\n")
    np.random.seed(42)

    n = 200
    phi = np.random.uniform(0.05, 0.30, n)
    k_h = 10 ** (np.random.normal(1.5, 0.8, n))  # log-normal, md
    k_v = k_h * np.random.uniform(0.1, 1.0, n)    # Kv < Kh typically

    rqi = reservoir_quality_index(k_h, phi)
    fzi = flow_zone_indicator(k_h, phi)
    print(f"RQI range: [{rqi.min():.3f}, {rqi.max():.3f}] µm")
    print(f"FZI range: [{fzi.min():.3f}, {fzi.max():.3f}] µm")

    labels, thresholds = classify_hfu(fzi, n_units=4)
    print(f"HFU classes: {np.unique(labels)}")

    k_arith = upscale_permeability_arithmetic(k_h)
    k_harm = upscale_permeability_harmonic(k_v)
    k_geom = upscale_permeability_geometric(k_h)
    print(f"\nUpscaled Kh (arithmetic): {k_arith:.2f} md")
    print(f"Upscaled Kv (harmonic):   {k_harm:.2f} md")
    print(f"Upscaled K  (geometric):  {k_geom:.2f} md")

    ratios = kv_kh_ratio(k_v, k_h)
    print("\nKv/Kh ratios by window size:")
    for ws, r in ratios.items():
        print(f"  Window {ws:3d}: Kv/Kh = {r:.4f}")

    facies = np.random.choice(['spherulite', 'stromatolite'], n)
    stats = facies_permeability_stats(k_h, facies)
    for f, s in stats.items():
        print(f"\nFacies '{f}': n={s['count']}, "
              f"median={s['median']:.1f} md, P10-P90=[{s['p10']:.1f}, {s['p90']:.1f}]")
