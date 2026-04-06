#!/usr/bin/env python3
"""
Thin-Bed NMR Response Characterisation in Horizontal Wells
============================================================
Based on: Ramadan et al. (2024), "Characterizing Thin-Bed Responses in
Horizontal Wells Using LWD NMR Tools: Insights From a Water Tank
Experiment", Petrophysics, Vol. 65, No. 5, pp. 765-771.

Implements:
- LWD NMR tool response simulation for thin beds
- Sensitivity kernel / vertical resolution modelling
- Apparent porosity computation with shoulder-bed effects
- Bed-boundary detection from NMR porosity profiles
- Correction for tool stand-off in horizontal wells

Reference: DOI:10.30632/PJV65N5-2024a6
"""

import numpy as np
from typing import Tuple, Optional


def nmr_sensitivity_kernel(z: np.ndarray, aperture: float = 0.3,
                           sigma: float = 0.08) -> np.ndarray:
    """
    1-D NMR tool sensitivity (vertical response) kernel.

    Modelled as a Gaussian centred on the measurement point.

    Parameters
    ----------
    z : np.ndarray
        Vertical offset from the tool measurement point (m).
    aperture : float
        Effective antenna aperture (m).
    sigma : float
        Gaussian width parameter (m). Controls vertical resolution.

    Returns
    -------
    np.ndarray
        Normalised sensitivity weights.
    """
    kernel = np.exp(-0.5 * (z / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def apparent_porosity(true_porosity: np.ndarray,
                      bed_boundaries: np.ndarray,
                      dz: float = 0.01,
                      sigma: float = 0.08) -> np.ndarray:
    """
    Simulate apparent NMR porosity in a layered formation with thin beds.

    Convolves the true porosity profile with the NMR sensitivity kernel,
    demonstrating shoulder-bed averaging effects.

    Parameters
    ----------
    true_porosity : np.ndarray
        True porosity profile along the wellbore (fraction).
    bed_boundaries : np.ndarray
        Ignored here; profile is convolved directly.
    dz : float
        Depth sampling interval (m).
    sigma : float
        Kernel width (m). Typical LWD NMR: 0.05-0.15 m.

    Returns
    -------
    np.ndarray
        Apparent (tool-measured) porosity profile.
    """
    n = len(true_porosity)
    half_win = int(3 * sigma / dz)
    z_kernel = np.arange(-half_win, half_win + 1) * dz
    kernel = nmr_sensitivity_kernel(z_kernel, sigma=sigma)

    apparent = np.convolve(true_porosity, kernel, mode='same')
    return np.clip(apparent, 0, None)


def thin_bed_correction(apparent_phi: np.ndarray,
                        bed_thickness: float,
                        tool_resolution: float = 0.15
                        ) -> np.ndarray:
    """
    First-order thin-bed correction factor.

    If bed thickness < tool vertical resolution, the measured porosity
    is diluted. Approximate correction:

        phi_corrected = phi_apparent * (tool_resolution / bed_thickness)

    clamped to physically reasonable bounds.

    Parameters
    ----------
    apparent_phi : np.ndarray
        Measured apparent porosity.
    bed_thickness : float
        True bed thickness (m).
    tool_resolution : float
        Tool vertical resolution (m).

    Returns
    -------
    np.ndarray
        Corrected porosity.
    """
    if bed_thickness >= tool_resolution:
        return apparent_phi.copy()
    factor = tool_resolution / bed_thickness
    return np.clip(apparent_phi * factor, 0, 0.50)


def standoff_correction(phi_measured: float,
                        standoff: float,
                        doi: float = 0.05) -> float:
    """
    Correct NMR porosity for tool stand-off in horizontal wells.

    The sensitive volume partially probes the borehole fluid (water)
    instead of the formation if the tool is eccentralized.

    phi_corrected = phi_measured - (standoff / doi) * phi_fluid

    Parameters
    ----------
    phi_measured : float
        Raw NMR porosity (fraction).
    standoff : float
        Gap between tool and borehole wall (m).
    doi : float
        NMR depth of investigation (m).

    Returns
    -------
    float
        Stand-off-corrected porosity.
    """
    phi_fluid = 1.0  # borehole water
    correction = (standoff / doi) * phi_fluid
    return max(phi_measured - correction, 0.0)


def detect_bed_boundaries(porosity_profile: np.ndarray,
                          dz: float = 0.01,
                          threshold: float = 0.03
                          ) -> np.ndarray:
    """
    Detect bed boundaries from abrupt changes in the porosity profile.

    Parameters
    ----------
    porosity_profile : np.ndarray
        Porosity profile along wellbore.
    dz : float
        Depth sampling interval (m).
    threshold : float
        Minimum gradient magnitude to flag a boundary.

    Returns
    -------
    np.ndarray
        Indices of detected bed boundaries.
    """
    gradient = np.abs(np.gradient(porosity_profile, dz))
    boundaries = np.where(gradient > threshold)[0]

    # Merge nearby detections
    if len(boundaries) == 0:
        return boundaries
    merged = [boundaries[0]]
    for b in boundaries[1:]:
        if b - merged[-1] > 3:
            merged.append(b)
    return np.array(merged)


def generate_layered_porosity(n_points: int = 500,
                              dz: float = 0.01,
                              bed_props: Optional[list] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic layered porosity profile with thin beds.

    Returns
    -------
    depths : np.ndarray
    porosity : np.ndarray
    """
    depths = np.arange(n_points) * dz
    porosity = np.full(n_points, 0.05)  # background shale

    if bed_props is None:
        bed_props = [
            (0.5, 1.5, 0.25),   # thick sand
            (2.0, 2.1, 0.30),   # thin bed (10 cm)
            (2.5, 2.55, 0.28),  # very thin bed (5 cm)
            (3.0, 4.0, 0.22),   # thick sand
        ]

    for z_top, z_bot, phi_bed in bed_props:
        mask = (depths >= z_top) & (depths < z_bot)
        porosity[mask] = phi_bed

    return depths, porosity


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Thin-Bed NMR Module Demo ===\n")

    depths, true_phi = generate_layered_porosity()
    print(f"Profile: {len(depths)} points, dz = {depths[1]-depths[0]:.3f} m")
    print(f"True porosity range: [{true_phi.min():.3f}, {true_phi.max():.3f}]")

    app_phi = apparent_porosity(true_phi, bed_boundaries=None, sigma=0.08)
    print(f"Apparent porosity range: [{app_phi.min():.3f}, {app_phi.max():.3f}]")

    # Thin-bed correction for the 5-cm bed
    idx_thin = np.argmin(np.abs(depths - 2.52))
    corrected = thin_bed_correction(
        np.array([app_phi[idx_thin]]), bed_thickness=0.05
    )
    print(f"\nAt thin bed (5 cm): apparent={app_phi[idx_thin]:.4f}, "
          f"corrected={corrected[0]:.4f}, true={true_phi[idx_thin]:.4f}")

    # Stand-off correction
    phi_raw = 0.22
    phi_corr = standoff_correction(phi_raw, standoff=0.01)
    print(f"\nStand-off correction: raw={phi_raw:.3f}, corrected={phi_corr:.3f}")

    # Bed boundary detection
    boundaries = detect_bed_boundaries(true_phi)
    print(f"\nDetected {len(boundaries)} bed boundaries in true profile")
    boundaries_app = detect_bed_boundaries(app_phi, threshold=0.01)
    print(f"Detected {len(boundaries_app)} bed boundaries in apparent profile")
