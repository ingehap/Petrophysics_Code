#!/usr/bin/env python3
"""
Probe Permeameter Testing and Application
==========================================
Based on: Jensen & Uroza (2024), "Testing and Application of a Probe
Permeameter in the Lower Wilcox Formation, Onshore Texas, USA",
Petrophysics, Vol. 65, No. 5, pp. 665-681.

Implements:
- Geometric factor (GF) calculation for probe permeameter tips
- Depth of investigation (DOI) estimation at various response levels
- Surface impairment correction for long-stored cores
- Tip calibration (o-ring vs. silicone rubber)
- Permeability–grain-size relationship
- CO2 injectivity and trapping potential assessment

Reference: DOI:10.30632/PJV65N5-2024a1
"""

import numpy as np
from typing import Tuple, Optional


def geometric_factor(ri: float, ro: float) -> float:
    """
    Compute the geometric factor for a probe permeameter tip.

    The geometric factor relates the measured pressure response to
    permeability via the probe seal geometry (Goggin et al., 1988).

    Parameters
    ----------
    ri : float
        Inner seal radius (m).
    ro : float
        Outer seal radius (m).

    Returns
    -------
    float
        Geometric factor (m).
    """
    # GF ~ ri * f(ro/ri), approximation from the hemispherical flow model
    # For a flat-faced probe: GF ≈ (4 * ri) / (1 + ro/ri)
    ratio = ro / ri
    gf = (4.0 * ri) / (1.0 + ratio)
    return gf


def depth_of_investigation(ri: float, response_level: float = 0.90) -> float:
    """
    Estimate the depth of investigation (DOI) of a probe permeameter.

    Based on isopotential line analysis (Goggin et al., 1988; Manrique
    et al., 1997; Corbett and Jensen, 1992).

    Parameters
    ----------
    ri : float
        Inner seal radius (m).
    response_level : float
        Fraction of total permeability response (0 to 1).
        Common values: 0.50 -> DOI ≈ 0.7*ri
                       0.90 -> DOI ≈ 2.5*ri (Corbett & Jensen, 1992)
                       0.95 -> DOI ≈ 4*ri   (Goggin et al., 1988)

    Returns
    -------
    float
        Estimated depth of investigation (m).
    """
    if response_level <= 0.50:
        multiplier = 0.7
    elif response_level <= 0.90:
        # Linear interpolation between 50% (0.7ri) and 90% (2.5ri)
        multiplier = 0.7 + (response_level - 0.50) / (0.90 - 0.50) * (2.5 - 0.7)
    elif response_level <= 0.95:
        multiplier = 2.5 + (response_level - 0.90) / (0.95 - 0.90) * (4.0 - 2.5)
    else:
        multiplier = 4.0
    return multiplier * ri


def tip_calibration(kp_oring: np.ndarray, calibration_factor: float = 0.63
                    ) -> np.ndarray:
    """
    Convert o-ring tip permeabilities to equivalent silicone-rubber-tip
    values using an empirical calibration factor.

    From Fig. 8 of the paper: k_silicone = calibration_factor * k_oring.

    Parameters
    ----------
    kp_oring : array-like
        Probe permeabilities measured with the o-ring tip (md).
    calibration_factor : float
        Empirical ratio (default 0.63 from Jensen & Uroza).

    Returns
    -------
    np.ndarray
        Equivalent silicone-tip permeabilities (md).
    """
    return np.asarray(kp_oring) * calibration_factor


def surface_impairment_correction(kp: np.ndarray,
                                  kc: np.ndarray,
                                  method: str = "regression"
                                  ) -> Tuple[np.ndarray, float]:
    """
    Correct probe permeabilities for surface impairment on long-stored cores.

    Following Corbett and Jensen (1992) and Filomena et al. (2013), a
    multiplicative adjustment factor is derived from paired probe–plug
    data and applied to all probe readings.

    Parameters
    ----------
    kp : array-like
        Probe permeabilities at calibration points (md).
    kc : array-like
        Core-plug permeabilities at the same locations (md).
    method : str
        'regression' – least-squares fit in log-space (log kc = a + b*log kp).
        'ratio' – simple geometric-mean ratio kc/kp.

    Returns
    -------
    kp_corrected : np.ndarray
        Adjusted probe permeabilities (md).
    correction_factor : float
        Applied multiplicative factor (or slope in log-space).
    """
    kp = np.asarray(kp, dtype=float)
    kc = np.asarray(kc, dtype=float)
    mask = (kp > 0) & (kc > 0)
    kp_valid, kc_valid = kp[mask], kc[mask]

    if method == "ratio":
        factor = np.exp(np.mean(np.log(kc_valid / kp_valid)))
        return kp * factor, factor

    # Log-space regression: log10(kc) = a + b * log10(kp)
    log_kp = np.log10(kp_valid)
    log_kc = np.log10(kc_valid)
    b, a = np.polyfit(log_kp, log_kc, 1)
    kp_corrected = 10.0 ** (a + b * np.log10(np.maximum(kp, 1e-6)))
    return kp_corrected, b


def measurement_time(permeability_md: float) -> float:
    """
    Approximate measurement time required for a probe permeameter reading.

    Empirical fit based on Fig. 5a of the paper.

    Parameters
    ----------
    permeability_md : float
        Estimated sample permeability (md).

    Returns
    -------
    float
        Approximate measurement time (seconds).
    """
    if permeability_md <= 0:
        return 120.0
    # Empirical decay: t ≈ 100 for k ≤ 100 md, dropping to ~25 s at 500+ md
    t = 100.0 * np.exp(-np.log(4) * (permeability_md - 100.0) / 400.0)
    return float(np.clip(t, 25.0, 120.0))


def perm_grain_size(grain_diameter_mm: np.ndarray,
                    a: float = 760.0, b: float = 2.0
                    ) -> np.ndarray:
    """
    Estimate permeability from grain size using an empirical power law.

    k = a * d^b   (simplified Kozeny–Carmen-style relationship)

    Parameters
    ----------
    grain_diameter_mm : array-like
        Median grain diameter (mm).
    a : float
        Coefficient (md / mm^b). Default tuned to Lower Wilcox channels.
    b : float
        Exponent.

    Returns
    -------
    np.ndarray
        Estimated permeability (md).
    """
    d = np.asarray(grain_diameter_mm, dtype=float)
    return a * d ** b


def co2_injectivity_index(k_md: float, h_ft: float,
                          mu_cp: float = 0.05,
                          Bo: float = 1.0) -> float:
    """
    Simple injectivity index estimate for CO2 injection.

    II = k * h / (141.2 * mu * Bo)   (field units, bbl/d/psi)

    Parameters
    ----------
    k_md : float
        Permeability (md).
    h_ft : float
        Net pay thickness (ft).
    mu_cp : float
        CO2 viscosity at reservoir conditions (cp).
    Bo : float
        Formation volume factor.

    Returns
    -------
    float
        Injectivity index (bbl/d/psi).
    """
    return (k_md * h_ft) / (141.2 * mu_cp * Bo)


def trapping_capacity(porosity: float, k_md: float,
                      Sgr: float = 0.25) -> float:
    """
    Estimate residual CO2 trapping capacity.

    Parameters
    ----------
    porosity : float
        Fractional porosity.
    k_md : float
        Permeability (md). Used qualitatively; higher k → lower Sgr.
    Sgr : float
        Residual gas saturation (fraction). Default 0.25 for moderate perm.

    Returns
    -------
    float
        Trapping capacity per unit bulk volume (fraction).
    """
    # Empirical: Sgr decreases with higher permeability
    Sgr_adj = Sgr * np.exp(-k_md / 500.0) + 0.05
    return porosity * Sgr_adj


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Probe Permeameter Module Demo ===\n")

    ri_mm = 3.9  # inner seal radius in mm
    ro_mm = 6.4  # outer seal radius in mm
    ri_m = ri_mm / 1000.0
    ro_m = ro_mm / 1000.0

    gf = geometric_factor(ri_m, ro_m)
    print(f"Geometric factor (ri={ri_mm} mm, ro={ro_mm} mm): {gf*1000:.3f} mm")

    for level in [0.50, 0.90, 0.95]:
        doi = depth_of_investigation(ri_m, level)
        print(f"DOI at {level*100:.0f}% response: {doi*1000:.2f} mm")

    kp_oring_vals = np.array([10, 50, 100, 200, 500])
    kp_sil = tip_calibration(kp_oring_vals)
    print(f"\nO-ring perms (md): {kp_oring_vals}")
    print(f"Silicone-tip perms (md): {kp_sil}")

    print(f"\nMeasurement time at 50 md: {measurement_time(50):.1f} s")
    print(f"Measurement time at 300 md: {measurement_time(300):.1f} s")

    grain_d = np.array([0.1, 0.2, 0.3, 0.5])
    print(f"\nGrain diameters (mm): {grain_d}")
    print(f"Estimated permeabilities (md): {perm_grain_size(grain_d)}")

    ii = co2_injectivity_index(k_md=200, h_ft=30)
    print(f"\nCO2 injectivity index (k=200 md, h=30 ft): {ii:.2f} bbl/d/psi")

    tc = trapping_capacity(porosity=0.25, k_md=200)
    print(f"Trapping capacity (phi=0.25, k=200 md): {tc:.4f}")
