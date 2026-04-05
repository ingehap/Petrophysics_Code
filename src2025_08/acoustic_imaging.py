#!/usr/bin/env python3
"""
Module 6: High-Resolution Acoustic Imaging for Casing Integrity
===============================================================
Implements ideas from:
  Alatigue et al., "Revolutionizing Complex Casing Integrity Analysis
  in the Middle East Using High-Resolution Acoustic Imaging,"
  Petrophysics, vol. 66, no. 4, pp. 616–630, August 2025.

Key concepts:
  - 512-sensor phased-array acoustic imaging
  - Amplitude + time-of-flight measurement per sensor
  - Sub-millimetric (0.25 mm) imaging resolution
  - 3-D point-cloud generation for ID / OD wall loss mapping
  - Defect detection and classification (pitting, corrosion, deformation)
  - Electronic focal-distance adjustment for multi-diameter inspection
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class AcousticImagingConfig:
    """Configuration for the 512-sensor acoustic imaging tool."""
    n_sensors: int = 512
    axial_resolution_mm: float = 0.25
    circumferential_resolution_deg: float = 360.0 / 512
    sound_velocity_m_s: float = 1480.0   # borehole fluid
    focal_distance_mm: float = 50.0


@dataclass
class Defect:
    """A detected defect on a casing surface."""
    azimuth_deg: float
    depth_ft: float
    radial_loss_mm: float
    angular_extent_deg: float
    axial_extent_mm: float
    defect_type: str   # 'pit', 'general_corrosion', 'deformation', 'perforation'


# ---------------------------------------------------------------------------
# 1. Generate synthetic casing surface (ID profile)
# ---------------------------------------------------------------------------
def generate_casing_surface(
    n_axial: int = 1000,
    n_azimuthal: int = 512,
    nominal_id_mm: float = 121.4,     # ~4.78 in for 5.5-in casing
    defects: Optional[List[dict]] = None,
    noise_mm: float = 0.02,
    rng=None,
) -> np.ndarray:
    """Create a 2-D casing ID map (axial × azimuthal).

    Parameters
    ----------
    defects : list of dict
        Each dict has keys: 'az' (deg), 'ax' (index), 'radius_mm',
        'az_width_deg', 'ax_width' (samples), 'type'.

    Returns
    -------
    ndarray, shape (n_axial, n_azimuthal), values in mm.
    """
    rng = rng or np.random.default_rng(0)
    surface = np.full((n_axial, n_azimuthal), nominal_id_mm / 2.0)

    if defects:
        az_axis = np.linspace(0, 360, n_azimuthal, endpoint=False)
        for d in defects:
            az0 = d['az']
            ax0 = d['ax']
            r_loss = d['radius_mm']
            az_w = d.get('az_width_deg', 10)
            ax_w = d.get('ax_width', 40)
            for i in range(n_axial):
                for j in range(n_azimuthal):
                    daz = min(abs(az_axis[j] - az0),
                              360 - abs(az_axis[j] - az0))
                    dax = abs(i - ax0)
                    g = np.exp(-0.5 * ((daz / (az_w / 2)) ** 2 +
                                        (dax / (ax_w / 2)) ** 2))
                    surface[i, j] += r_loss * g

    surface += rng.normal(0, noise_mm, surface.shape)
    return surface


# ---------------------------------------------------------------------------
# 2. Simulate time-of-flight measurement
# ---------------------------------------------------------------------------
def time_of_flight(
    surface_radius_mm: np.ndarray,
    tool_radius_mm: float = 30.0,
    sound_velocity_m_s: float = 1480.0,
) -> np.ndarray:
    """Compute two-way travel time from tool to casing surface.

    Returns time in microseconds.
    """
    standoff_mm = surface_radius_mm - tool_radius_mm
    twt_us = 2.0 * standoff_mm / (sound_velocity_m_s * 1e-3)
    return twt_us


# ---------------------------------------------------------------------------
# 3. Amplitude image (reflection strength)
# ---------------------------------------------------------------------------
def amplitude_image(
    surface_radius_mm: np.ndarray,
    nominal_radius_mm: float,
) -> np.ndarray:
    """Compute normalised amplitude image.

    Defects (larger radius → material loss) produce lower amplitude
    due to scattering at rough or corroded surfaces.
    """
    deviation = np.abs(surface_radius_mm - nominal_radius_mm)
    amp = np.exp(-deviation / 0.5)  # sharper defects → lower amplitude
    return amp


# ---------------------------------------------------------------------------
# 4. 3-D point cloud from ToF
# ---------------------------------------------------------------------------
def generate_point_cloud(
    surface_radius_mm: np.ndarray,
    axial_pitch_mm: float = 0.25,
) -> np.ndarray:
    """Convert the 2-D surface map to a 3-D point cloud.

    Returns
    -------
    ndarray, shape (N, 3) — (x, y, z) in mm
    """
    n_ax, n_az = surface_radius_mm.shape
    az_deg = np.linspace(0, 360, n_az, endpoint=False)
    points = []
    for i in range(n_ax):
        z = i * axial_pitch_mm
        for j in range(n_az):
            r = surface_radius_mm[i, j]
            theta = np.deg2rad(az_deg[j])
            points.append([r * np.cos(theta), r * np.sin(theta), z])
    return np.array(points)


# ---------------------------------------------------------------------------
# 5. Defect detection algorithm
# ---------------------------------------------------------------------------
def detect_defects(
    surface_radius_mm: np.ndarray,
    nominal_radius_mm: float,
    threshold_mm: float = 0.3,
    min_area_pixels: int = 20,
) -> List[Defect]:
    """Detect defects via thresholding the deviation from nominal.

    Groups contiguous pixels into defect regions and classifies them.
    """
    deviation = surface_radius_mm - nominal_radius_mm
    mask = deviation > threshold_mm   # material loss → increased radius

    # Simple connected-component labelling (row-scan)
    n_ax, n_az = mask.shape
    labels = np.zeros_like(mask, dtype=int)
    current_label = 0
    for i in range(n_ax):
        for j in range(n_az):
            if mask[i, j] and labels[i, j] == 0:
                current_label += 1
                _flood_fill(mask, labels, i, j, current_label)

    defects: List[Defect] = []
    az_axis = np.linspace(0, 360, n_az, endpoint=False)
    for lbl in range(1, current_label + 1):
        region = np.argwhere(labels == lbl)
        if len(region) < min_area_pixels:
            continue
        ax_indices = region[:, 0]
        az_indices = region[:, 1]
        max_dev_idx = np.unravel_index(
            np.argmax(deviation * (labels == lbl)), deviation.shape)
        max_loss = deviation[max_dev_idx]

        ax_extent = (ax_indices.max() - ax_indices.min()) * 0.25  # mm
        az_extent = (az_axis[az_indices.max()] - az_axis[az_indices.min()])

        # Classify
        if ax_extent < 5 and az_extent < 15:
            dtype = 'pit'
        elif ax_extent > 50 and az_extent > 60:
            dtype = 'general_corrosion'
        elif max_loss > 3:
            dtype = 'perforation'
        else:
            dtype = 'general_corrosion'

        defects.append(Defect(
            azimuth_deg=float(az_axis[int(np.median(az_indices))]),
            depth_ft=float(np.median(ax_indices)) * 0.25 / 304.8,
            radial_loss_mm=float(max_loss),
            angular_extent_deg=float(az_extent),
            axial_extent_mm=float(ax_extent),
            defect_type=dtype,
        ))

    return defects


def _flood_fill(mask, labels, i, j, label):
    """Simple iterative flood fill."""
    stack = [(i, j)]
    n_ax, n_az = mask.shape
    while stack:
        ci, cj = stack.pop()
        if ci < 0 or ci >= n_ax or cj < 0 or cj >= n_az:
            continue
        if not mask[ci, cj] or labels[ci, cj] != 0:
            continue
        labels[ci, cj] = label
        stack.extend([(ci + 1, cj), (ci - 1, cj),
                      (ci, cj + 1), (ci, cj - 1)])


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_all():
    cfg = AcousticImagingConfig()
    nom_id_mm = 121.4
    defects_spec = [
        {'az': 90, 'ax': 300, 'radius_mm': 1.5, 'az_width_deg': 20, 'ax_width': 60},
        {'az': 250, 'ax': 700, 'radius_mm': 0.8, 'az_width_deg': 8, 'ax_width': 15},
    ]

    surface = generate_casing_surface(n_axial=1000, n_azimuthal=512,
                                       nominal_id_mm=nom_id_mm,
                                       defects=defects_spec)
    assert surface.shape == (1000, 512)

    tof = time_of_flight(surface)
    assert tof.shape == surface.shape
    assert np.all(tof > 0)

    amp = amplitude_image(surface, nom_id_mm / 2)
    assert amp.min() >= 0

    # 3D point cloud (subsample for speed)
    pc = generate_point_cloud(surface[:10, :64])
    assert pc.shape[1] == 3

    # Defect detection
    detected = detect_defects(surface, nom_id_mm / 2, threshold_mm=0.3)
    assert len(detected) >= 1, "Should detect at least one defect"
    assert any(abs(d.azimuth_deg - 90) < 30 for d in detected), \
        "Should detect defect near 90°"

    print("[PASS] acoustic_imaging — all tests passed")


if __name__ == "__main__":
    test_all()
