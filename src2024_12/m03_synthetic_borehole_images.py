#!/usr/bin/env python3
"""
Synthetic Borehole Images From Outcrop Photographs
====================================================
Based on: Fornero, Menezes de Jesus, Fernandes, and Trevizan (2024),
Petrophysics 65(6), pp. 887-894. DOI: 10.30632/PJV65N6-2024a3

Implements the methodology to create pseudo-borehole image logs from
outcrop photographs by:
  1. Cutting a strip from the photograph matching well diameter.
  2. Extruding the 2D strip into a 3D volume (mirror symmetry).
  3. Intersecting the volume with a virtual cylinder (the borehole).
  4. Unwrapping the cylindrical surface into a 2D borehole image.
  5. Applying a standard color palette for borehole images.
"""

import numpy as np
from typing import Tuple, Optional


def cut_photo_strip(
    photo: np.ndarray,
    strip_width_cm: float = 30.0,
    photo_width_cm: float = 100.0,
    center_offset_fraction: float = 0.5,
) -> np.ndarray:
    """
    Cut a vertical strip from the outcrop photograph matching the
    well diameter coverage.

    Parameters
    ----------
    photo : np.ndarray
        Input outcrop photograph (H, W) or (H, W, C).
    strip_width_cm : float
        Width of strip in cm (30 cm ~ 12.25 in. well).
    photo_width_cm : float
        Total width of the photo in cm.
    center_offset_fraction : float
        Horizontal center position (0-1).

    Returns
    -------
    np.ndarray
        Extracted strip.
    """
    if photo.ndim == 3:
        h, w, c = photo.shape
    else:
        h, w = photo.shape
        c = 1

    pixels_per_cm = w / photo_width_cm
    strip_pixels = int(strip_width_cm * pixels_per_cm)
    center_col = int(w * center_offset_fraction)
    left = max(0, center_col - strip_pixels // 2)
    right = min(w, left + strip_pixels)

    if photo.ndim == 3:
        return photo[:, left:right, :]
    return photo[:, left:right]


def extrude_to_volume(strip: np.ndarray) -> np.ndarray:
    """
    Extrude a 2D strip into a 3D volume using mirror symmetry.

    The strip is treated as a face of a rectangular prism, and the
    perpendicular dimension is created by mirroring the strip.

    Parameters
    ----------
    strip : np.ndarray of shape (H, W) or (H, W, C)

    Returns
    -------
    np.ndarray of shape (H, W, W) or (H, W, W, C)
        3D volume where the third axis is the extruded (depth-into-surface)
        dimension created via mirror symmetry.
    """
    if strip.ndim == 3:
        h, w, c = strip.shape
        volume = np.zeros((h, w, w, c), dtype=strip.dtype)
        for k in range(w):
            # Mirror symmetry: each layer is a blend of the strip
            # with decreasing weight toward the mirror plane
            weight = 1.0 - 0.3 * abs(k - w // 2) / max(w // 2, 1)
            volume[:, :, k, :] = strip * weight
    else:
        h, w = strip.shape
        volume = np.zeros((h, w, w), dtype=strip.dtype)
        for k in range(w):
            weight = 1.0 - 0.3 * abs(k - w // 2) / max(w // 2, 1)
            volume[:, :, k] = strip * weight
    return volume


def intersect_cylinder(
    volume: np.ndarray,
    well_diameter_pixels: int = None,
    n_azimuth: int = 360,
) -> np.ndarray:
    """
    Intersect the 3D volume with a virtual cylinder (borehole) and
    unwrap the surface into a 2D borehole image.

    Parameters
    ----------
    volume : np.ndarray
        3D volume of shape (H, Wx, Wy, ...).
    well_diameter_pixels : int
        Diameter of the virtual borehole in pixel units.
        If None, uses min(volume.shape[1], volume.shape[2]) - 2.
    n_azimuth : int
        Number of azimuthal columns in the output image.

    Returns
    -------
    np.ndarray
        Unwrapped borehole image of shape (H, n_azimuth) or (H, n_azimuth, C).
    """
    if volume.ndim == 4:
        h, wx, wy, c = volume.shape
        has_color = True
    else:
        h, wx, wy = volume.shape
        c = 1
        has_color = False

    if well_diameter_pixels is None:
        well_diameter_pixels = min(wx, wy) - 2
    radius = well_diameter_pixels / 2.0
    center_x = wx / 2.0
    center_y = wy / 2.0

    angles = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)

    if has_color:
        bhi = np.zeros((h, n_azimuth, c), dtype=np.float64)
    else:
        bhi = np.zeros((h, n_azimuth), dtype=np.float64)

    for ai, angle in enumerate(angles):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        ix = int(np.clip(x, 0, wx - 1))
        iy = int(np.clip(y, 0, wy - 1))
        if has_color:
            bhi[:, ai, :] = volume[:, ix, iy, :]
        else:
            bhi[:, ai] = volume[:, ix, iy]

    return bhi


def apply_bhi_colormap(
    bhi_gray: np.ndarray,
    palette: str = 'standard',
) -> np.ndarray:
    """
    Apply a standard borehole image color palette.

    Parameters
    ----------
    bhi_gray : np.ndarray of shape (H, W)
        Grayscale pseudo-borehole image (0-1 range).
    palette : str
        'standard' for brown-yellow-white (resistivity-like) or
        'amplitude' for blue-green-red (acoustic-like).

    Returns
    -------
    np.ndarray of shape (H, W, 3)
        RGB image with applied colormap.
    """
    bhi = np.clip(bhi_gray, 0, 1)
    rgb = np.zeros((*bhi.shape, 3), dtype=np.float64)

    if palette == 'standard':
        # Dark brown (low) -> yellow -> white (high)
        rgb[:, :, 0] = 0.3 + 0.7 * bhi  # R
        rgb[:, :, 1] = 0.15 + 0.85 * bhi  # G
        rgb[:, :, 2] = 0.05 + 0.95 * bhi ** 1.5  # B
    elif palette == 'amplitude':
        # Blue (low) -> green -> red (high)
        rgb[:, :, 0] = np.clip(2 * bhi - 1, 0, 1)
        rgb[:, :, 1] = np.clip(1 - 2 * abs(bhi - 0.5), 0, 1)
        rgb[:, :, 2] = np.clip(1 - 2 * bhi, 0, 1)
    else:
        rgb[:, :, 0] = bhi
        rgb[:, :, 1] = bhi
        rgb[:, :, 2] = bhi

    return np.clip(rgb, 0, 1)


def rotate_bhi(bhi: np.ndarray, rotation_degrees: float) -> np.ndarray:
    """
    Rotate the borehole image azimuthally to align with field
    measurements or a reference BHI.

    Parameters
    ----------
    bhi : np.ndarray
        Borehole image (H, W, ...).
    rotation_degrees : float
        Clockwise rotation in degrees.

    Returns
    -------
    np.ndarray
        Rotated borehole image.
    """
    n_cols = bhi.shape[1]
    shift = int(round(rotation_degrees / 360.0 * n_cols)) % n_cols
    return np.roll(bhi, shift, axis=1)


def create_pseudo_borehole_image(
    outcrop_photo: np.ndarray,
    well_diameter_cm: float = 31.1,  # 12.25 in.
    photo_width_cm: float = 100.0,
    n_azimuth: int = 360,
    rotation_deg: float = 0.0,
    palette: str = 'standard',
) -> np.ndarray:
    """
    Full workflow: create a pseudo-borehole image from an outcrop photo.

    Parameters
    ----------
    outcrop_photo : np.ndarray
        Outcrop photograph.
    well_diameter_cm : float
        Well diameter in cm.
    photo_width_cm : float
        Photo width in cm.
    n_azimuth : int
        Azimuthal resolution.
    rotation_deg : float
        Azimuthal rotation to apply.
    palette : str
        Color palette.

    Returns
    -------
    np.ndarray
        Colored pseudo-borehole image (H, n_azimuth, 3).
    """
    strip = cut_photo_strip(outcrop_photo, well_diameter_cm, photo_width_cm)
    volume = extrude_to_volume(strip)
    bhi = intersect_cylinder(volume, n_azimuth=n_azimuth)
    if bhi.ndim == 3:
        bhi_gray = np.mean(bhi, axis=2)
    else:
        bhi_gray = bhi
    # Normalize
    vmin, vmax = bhi_gray.min(), bhi_gray.max()
    if vmax > vmin:
        bhi_gray = (bhi_gray - vmin) / (vmax - vmin)
    bhi_colored = apply_bhi_colormap(bhi_gray, palette)
    if abs(rotation_deg) > 0.01:
        bhi_colored = rotate_bhi(bhi_colored, rotation_deg)
    return bhi_colored


def test_all():
    """Test all functions with synthetic data."""
    print("=" * 70)
    print("Module 3: Synthetic Borehole Images (Fornero et al., 2024)")
    print("=" * 70)

    rng = np.random.RandomState(42)

    # Create a synthetic outcrop photo with fracture-like features
    h, w = 200, 300
    photo = rng.rand(h, w) * 0.4 + 0.3
    # Add some "fractures" (linear features)
    for _ in range(5):
        y0 = rng.randint(0, h)
        slope = rng.uniform(-0.5, 0.5)
        for x in range(w):
            y = int(y0 + slope * x)
            if 0 <= y < h:
                photo[max(0, y-1):min(h, y+2), x] = rng.uniform(0.7, 1.0)

    print(f"Synthetic outcrop photo: {photo.shape}")

    # Cut strip
    strip = cut_photo_strip(photo, strip_width_cm=30.0, photo_width_cm=100.0)
    print(f"Cut strip: {strip.shape}")

    # Extrude to volume
    volume = extrude_to_volume(strip)
    print(f"Extruded volume: {volume.shape}")

    # Intersect with cylinder
    bhi = intersect_cylinder(volume, n_azimuth=360)
    print(f"Raw pseudo-BHI: {bhi.shape}")

    # Apply colormap
    bhi_norm = (bhi - bhi.min()) / (bhi.max() - bhi.min() + 1e-10)
    bhi_colored = apply_bhi_colormap(bhi_norm, 'standard')
    print(f"Colored pseudo-BHI: {bhi_colored.shape}")

    # Full workflow
    full_bhi = create_pseudo_borehole_image(
        photo, well_diameter_cm=31.1, rotation_deg=45.0
    )
    print(f"Full workflow BHI: {full_bhi.shape}")
    print(f"  Value range: [{full_bhi.min():.3f}, {full_bhi.max():.3f}]")

    # Test rotation
    rotated = rotate_bhi(bhi_colored, 90.0)
    print(f"Rotated BHI: {rotated.shape}")

    print("\n[PASS] All tests completed successfully.\n")


if __name__ == "__main__":
    test_all()
