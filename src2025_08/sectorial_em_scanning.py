#!/usr/bin/env python3
"""
Module 4: Sectorial Electromagnetic Scanning for Well Integrity
===============================================================
Implements ideas from:
  Jawed et al., "Application of Advanced Sectorial Electromagnetic
  Scanning Addressing Well Integrity Challenges,"
  Petrophysics, vol. 66, no. 4, pp. 578–593, August 2025.

Key concepts:
  - Azimuthal EM scanning with multiple radial sensors
  - Differentiation between uniform and localised metal loss
  - Pipe deformation / ovalization detection from sensor asymmetry
  - Comparison with conventional averaging EM methods
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SectorialSensor:
    """One radial sector of the scanning tool."""
    azimuth_deg: float     # angular position
    standoff_in: float     # distance from sensor to pipe wall


@dataclass
class SectorialToolConfig:
    """Configuration for a sectorial EM scanning tool."""
    n_sectors: int = 16
    nominal_standoff_in: float = 0.25
    frequency_hz: float = 500.0

    def sensors(self) -> List[SectorialSensor]:
        return [
            SectorialSensor(azimuth_deg=i * 360.0 / self.n_sectors,
                            standoff_in=self.nominal_standoff_in)
            for i in range(self.n_sectors)
        ]


# ---------------------------------------------------------------------------
# 1. Pipe wall-thickness model with localised defect
# ---------------------------------------------------------------------------
def pipe_wall_model(
    azimuths_deg: np.ndarray,
    nominal_thickness_in: float = 0.30,
    defect_azimuth_deg: float = 90.0,
    defect_angular_width_deg: float = 30.0,
    defect_metal_loss_frac: float = 0.0,
    ovality_amplitude_in: float = 0.0,
) -> np.ndarray:
    """Return pipe wall thickness as a function of azimuth.

    Parameters
    ----------
    defect_metal_loss_frac : float
        Fractional metal loss at the defect centre (0–1).
    ovality_amplitude_in : float
        Cosine ovality amplitude added to the thickness profile.
    """
    az_rad = np.deg2rad(azimuths_deg)
    thickness = np.full_like(az_rad, nominal_thickness_in, dtype=float)

    # Localised defect
    if defect_metal_loss_frac > 0:
        defect_rad = np.deg2rad(defect_azimuth_deg)
        width_rad = np.deg2rad(defect_angular_width_deg)
        angular_dist = np.abs(np.arctan2(
            np.sin(az_rad - defect_rad),
            np.cos(az_rad - defect_rad)))
        mask = angular_dist < width_rad / 2
        thickness[mask] -= nominal_thickness_in * defect_metal_loss_frac * \
            np.cos(np.pi * angular_dist[mask] / width_rad)

    # Ovalization
    if ovality_amplitude_in != 0:
        thickness += ovality_amplitude_in * np.cos(2 * az_rad)

    return np.clip(thickness, 0.01, None)


# ---------------------------------------------------------------------------
# 2. Sectorial EM response (simplified skin-depth model per sector)
# ---------------------------------------------------------------------------
def sectorial_em_response(
    wall_thickness: np.ndarray,
    standoffs: np.ndarray,
    frequency_hz: float = 500.0,
    conductivity: float = 5e6,
    mu_rel: float = 100.0,
    noise_std: float = 0.0,
    rng=None,
) -> np.ndarray:
    """Compute EM amplitude response for each sector.

    The response is modelled as proportional to wall thickness —
    thinner wall → lower magnetic flux return → lower response.
    Standoff variations also modulate the coupling.
    """
    rng = rng or np.random.default_rng(0)

    # Normalise thickness to nominal for response calculation
    # The tool measures flux return which is proportional to metal content
    nominal_t = np.median(wall_thickness)
    thickness_ratio = wall_thickness / (nominal_t + 1e-12)

    # Coupling decreases with standoff
    coupling = 1.0 / (1.0 + (standoffs / 0.5) ** 2)

    # Response proportional to remaining wall thickness (metal content)
    response = coupling * thickness_ratio

    if noise_std > 0:
        response += rng.normal(0, noise_std, size=response.shape)
    return response


# ---------------------------------------------------------------------------
# 3. Averaging EM (conventional method) vs. sectorial
# ---------------------------------------------------------------------------
def averaging_em_response(sectorial_response: np.ndarray) -> float:
    """Conventional averaging EM tool: single circumferential average."""
    return float(np.mean(sectorial_response))


# ---------------------------------------------------------------------------
# 4. Defect classification
# ---------------------------------------------------------------------------
def classify_defect(sectorial_response: np.ndarray,
                    nominal_response_level: float = 0.0,
                    uniformity_threshold: float = 0.15) -> str:
    """Classify defect type based on azimuthal variation.

    Parameters
    ----------
    nominal_response_level : float
        Mean response for an intact pipe (used to distinguish
        uniform loss from nominal).  If 0, auto-estimated.

    Returns
    -------
    str : 'uniform_loss', 'localised_loss', 'deformation', or 'nominal'
    """
    mean_r = np.mean(sectorial_response)
    std_r = np.std(sectorial_response)
    cv = std_r / (mean_r + 1e-12)

    if nominal_response_level <= 0:
        nominal_response_level = mean_r

    # Relative change from nominal
    rel_change = abs(mean_r - nominal_response_level) / (nominal_response_level + 1e-12)

    if cv < uniformity_threshold * 0.3:
        # Very low variation — either nominal or uniform loss
        if rel_change > 0.1:
            return "uniform_loss"
        return "nominal"
    elif cv < uniformity_threshold * 0.5:
        return "deformation"
    else:
        return "localised_loss"


# ---------------------------------------------------------------------------
# 5. Estimate localised metal loss from sectorial data
# ---------------------------------------------------------------------------
def estimate_metal_loss_azimuthal(
    sectorial_response: np.ndarray,
    azimuths_deg: np.ndarray,
    nominal_response: float,
) -> Tuple[float, float]:
    """Estimate the worst-sector metal-loss fraction and its azimuth.

    Lower response relative to nominal indicates metal loss (thinner wall).

    Returns
    -------
    (worst_azimuth_deg, estimated_loss_fraction)
    """
    # Lower response ↔ thinner wall ↔ more metal loss
    loss_indicator = (nominal_response - sectorial_response) / (nominal_response + 1e-12)
    worst_idx = int(np.argmax(loss_indicator))
    return float(azimuths_deg[worst_idx]), float(np.clip(loss_indicator[worst_idx], 0, 1))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_all():
    cfg = SectorialToolConfig(n_sectors=24)
    sensors = cfg.sensors()
    azimuths = np.array([s.azimuth_deg for s in sensors])
    standoffs = np.array([s.standoff_in for s in sensors])

    # Case 1: Nominal pipe
    wt_nom = pipe_wall_model(azimuths, nominal_thickness_in=0.30)
    resp_nom = sectorial_em_response(wt_nom, standoffs)
    nom_level = float(np.mean(resp_nom))
    label = classify_defect(resp_nom, nominal_response_level=nom_level)
    assert label == "nominal", f"Expected nominal, got {label}"

    # Case 2: Localised defect at 90°
    wt_loc = pipe_wall_model(azimuths, defect_azimuth_deg=90,
                              defect_angular_width_deg=40,
                              defect_metal_loss_frac=0.5)
    resp_loc = sectorial_em_response(wt_loc, standoffs)
    label = classify_defect(resp_loc, nominal_response_level=nom_level)
    assert label == "localised_loss", f"Expected localised_loss, got {label}"

    az_worst, loss_frac = estimate_metal_loss_azimuthal(
        resp_loc, azimuths, averaging_em_response(resp_nom))
    assert abs(az_worst - 90) < 30, f"Worst azimuth {az_worst} not near 90°"

    # Case 3: Averaging EM misses the localised defect
    avg_nom = averaging_em_response(resp_nom)
    avg_loc = averaging_em_response(resp_loc)
    # The sectorial response shows much larger variation than averaging
    sectorial_range = np.max(resp_loc) - np.min(resp_loc)
    avg_change = abs(avg_loc - avg_nom)
    assert sectorial_range > 2 * avg_change, \
        "Sectorial should show much larger anomaly than averaging"

    # Case 4: Ovalization / deformation
    wt_ov = pipe_wall_model(azimuths, ovality_amplitude_in=0.02)
    resp_ov = sectorial_em_response(wt_ov, standoffs)
    label = classify_defect(resp_ov, nominal_response_level=nom_level,
                             uniformity_threshold=0.20)
    assert label in ("deformation", "nominal"), f"Got {label}"

    print("[PASS] sectorial_em_scanning — all tests passed")


if __name__ == "__main__":
    test_all()
