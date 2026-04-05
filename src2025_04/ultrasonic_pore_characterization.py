"""
Ultrasonic Microscopy Imaging of Carbonate Reservoir Pore Structure
====================================================================
Based on: Chen et al., "New Methodology for Ultrasonic Microscopy Imaging
of Carbonate Reservoirs' Pore Structure",
Petrophysics, Vol. 66, No. 2, April 2025, pp. 267–282.

Implements:
  - Acoustic impedance and reflection coefficient computation
  - Pore contour digitization via gradient + Otsu thresholding
  - Pore centroid calculation from boundary coordinates (Eqs. 7–10)
  - Horizontal shape descriptors: shape factor XZ, flatness BP,
    angular factor JZ, convexity TD (Eq. 12)
  - Fourier discrete descriptors for irregular pore boundaries
  - 3D pore morphology reconstruction from multi-layer scans

Reference: https://doi.org/10.30632/PJV66N2-2025a6 (SPWLA)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class PoreMorphology:
    """Quantitative descriptors for a single pore."""
    area: float
    perimeter: float
    centroid: Tuple[float, float]
    shape_factor: float       # XZ (Eq. 12)
    flatness_factor: float    # BP
    angular_factor: float     # JZ
    convexity_factor: float   # TD
    feret_max: float          # Maximum Feret diameter
    feret_min: float          # Minimum Feret diameter


def acoustic_impedance(density_g_cm3: float, velocity_m_s: float) -> float:
    """
    Compute acoustic impedance Z = rho * c (Eq. 1).

    Parameters
    ----------
    density_g_cm3 : float  Density in g/cm³
    velocity_m_s : float   Sound velocity in m/s

    Returns
    -------
    float : Acoustic impedance in MRayl (10⁶ kg/m²s)
    """
    density_kg_m3 = density_g_cm3 * 1000.0
    Z = density_kg_m3 * velocity_m_s
    return Z / 1e6  # Convert to MRayl


def reflection_coefficient(Z1: float, Z2: float) -> float:
    """
    Sound pressure reflection coefficient at normal incidence (Eq. 3).

    rP = (Z2 - Z1) / (Z2 + Z1)

    Parameters
    ----------
    Z1 : float  Acoustic impedance of medium 1 (MRayl)
    Z2 : float  Acoustic impedance of medium 2 (MRayl)

    Returns
    -------
    float : Reflection coefficient (-1 to 1)
    """
    return (Z2 - Z1) / (Z2 + Z1)


def otsu_threshold(image: np.ndarray) -> float:
    """
    Maximum inter-class variance method (Otsu) for threshold selection.

    Used in the paper to segment pore areas from the background in
    ultrasonic microscopy images.

    Parameters
    ----------
    image : np.ndarray, 2D
        Grayscale image (0–255 or float 0–1).

    Returns
    -------
    float : Optimal threshold value.
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    total = image.size
    best_thresh = 0
    best_var = 0.0
    sum_all = np.sum(np.arange(256) * hist)
    sum_bg = 0.0
    weight_bg = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = t

    return best_thresh


def sobel_gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Sobel gradient magnitude and components (Eqs. 5–6).

    Parameters
    ----------
    image : np.ndarray, 2D

    Returns
    -------
    Tuple of (Gx, Gy, magnitude)
    """
    # Sobel kernels
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    # Simple convolution
    h, w = image.shape
    Gx = np.zeros_like(image, dtype=float)
    Gy = np.zeros_like(image, dtype=float)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            patch = image[i - 1:i + 2, j - 1:j + 2].astype(float)
            Gx[i, j] = np.sum(patch * Kx)
            Gy[i, j] = np.sum(patch * Ky)

    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    return Gx, Gy, magnitude


def segment_pores(image: np.ndarray,
                  gradient_weight: float = 0.3) -> np.ndarray:
    """
    Segment pore regions using combined gradient and Otsu method.

    The paper proposes combining gradient operators with the maximum
    inter-class variance method (Otsu) for pore area identification.

    Parameters
    ----------
    image : np.ndarray, 2D grayscale
    gradient_weight : float
        Weight for gradient enhancement.

    Returns
    -------
    np.ndarray : Binary mask (1=pore, 0=matrix)
    """
    _, _, grad_mag = sobel_gradient(image)

    # Normalize
    if grad_mag.max() > 0:
        grad_norm = grad_mag / grad_mag.max()
    else:
        grad_norm = grad_mag

    # Enhanced image: lower values where gradient is high (edges)
    enhanced = image.astype(float)
    if enhanced.max() > 1.0:
        enhanced = enhanced / 255.0
    enhanced = enhanced - gradient_weight * grad_norm
    enhanced = np.clip(enhanced, 0, 1)

    threshold = otsu_threshold(enhanced) / 255.0
    binary = (enhanced < threshold).astype(int)
    return binary


def contour_centroid(boundary_x: np.ndarray,
                     boundary_y: np.ndarray) -> Tuple[float, float]:
    """
    Compute the centroid of a closed pore boundary (Eqs. 7–10).

    Uses the shoelace formula for area and centroid moments.

    Parameters
    ----------
    boundary_x, boundary_y : np.ndarray
        Coordinates of N boundary points (closed: first ≈ last).

    Returns
    -------
    Tuple[float, float] : (xz, yz) centroid coordinates.
    """
    n = len(boundary_x)
    if n < 3:
        return (np.mean(boundary_x), np.mean(boundary_y))

    # Ensure closed
    x = np.append(boundary_x, boundary_x[0])
    y = np.append(boundary_y, boundary_y[0])

    # Shoelace area (Eq. 8)
    A = 0.0
    Mx = 0.0
    My = 0.0
    for i in range(n):
        cross = x[i] * y[i + 1] - x[i + 1] * y[i]
        A += cross
        Mx += (x[i] + x[i + 1]) * cross
        My += (y[i] + y[i + 1]) * cross

    A *= 0.5
    if abs(A) < 1e-12:
        return (np.mean(boundary_x), np.mean(boundary_y))

    xz = Mx / (6.0 * A)
    yz = My / (6.0 * A)
    return (xz, yz)


def pore_shape_descriptors(boundary_x: np.ndarray,
                           boundary_y: np.ndarray) -> PoreMorphology:
    """
    Compute horizontal morphology descriptors (Eq. 12).

    Shape factor XZ:    ratio of equivalent circumference to actual perimeter
    Flatness factor BP: ratio of min to max Feret diameter
    Angular factor JZ:  ratio of equivalent ellipse perimeter to actual
    Convexity factor TD: ratio of pore area to convex hull area

    Parameters
    ----------
    boundary_x, boundary_y : np.ndarray  Boundary coordinates.

    Returns
    -------
    PoreMorphology
    """
    cx, cy = contour_centroid(boundary_x, boundary_y)

    # Perimeter PA
    dx = np.diff(np.append(boundary_x, boundary_x[0]))
    dy = np.diff(np.append(boundary_y, boundary_y[0]))
    PA = np.sum(np.sqrt(dx ** 2 + dy ** 2))

    # Area SA (shoelace)
    n = len(boundary_x)
    x = np.append(boundary_x, boundary_x[0])
    y = np.append(boundary_y, boundary_y[0])
    SA = 0.0
    for i in range(n):
        SA += x[i] * y[i + 1] - x[i + 1] * y[i]
    SA = abs(SA) * 0.5

    # Equivalent circumference (circle with same area)
    PC = 2.0 * np.pi * np.sqrt(SA / np.pi)

    # Feret diameters
    from itertools import combinations
    coords = np.column_stack([boundary_x, boundary_y])
    if len(coords) > 50:
        # Subsample for speed
        idx = np.linspace(0, len(coords) - 1, 50, dtype=int)
        coords_sub = coords[idx]
    else:
        coords_sub = coords

    dists = []
    for i in range(len(coords_sub)):
        for j in range(i + 1, len(coords_sub)):
            d = np.sqrt(np.sum((coords_sub[i] - coords_sub[j]) ** 2))
            dists.append(d)
    if dists:
        XA = max(dists)  # Max Feret
        XC = min(dists) if min(dists) > 0 else XA * 0.1  # Min Feret
    else:
        XA = XC = 1.0

    # Shape factor XZ
    XZ = PC / PA if PA > 0 else 0.0

    # Flatness factor BP
    BP = XC / XA if XA > 0 else 1.0

    # Angular factor JZ: equivalent ellipse perimeter / PA
    a = XA / 2.0  # semi-major
    b = XC / 2.0  # semi-minor
    # Ramanujan approximation for ellipse perimeter
    h = ((a - b) / (a + b)) ** 2
    PE = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
    JZ = PE / PA if PA > 0 else 0.0

    # Convexity factor TD: SA / convex hull area
    # Simple approximation using bounding rectangle
    SP = XA * XC  # rough convex hull approximation
    TD = SA / SP if SP > 0 else 1.0

    return PoreMorphology(
        area=SA,
        perimeter=PA,
        centroid=(cx, cy),
        shape_factor=XZ,
        flatness_factor=BP,
        angular_factor=JZ,
        convexity_factor=TD,
        feret_max=XA,
        feret_min=XC,
    )


def fourier_descriptors(boundary_x: np.ndarray,
                        boundary_y: np.ndarray,
                        n_harmonics: int = 16) -> np.ndarray:
    """
    Compute Fourier descriptors for pore boundary shape.

    The paper uses Fourier discrete transformation to convert
    irregular pore boundary contours into frequency-domain signals
    for shape characterization.

    Parameters
    ----------
    boundary_x, boundary_y : np.ndarray
    n_harmonics : int

    Returns
    -------
    np.ndarray : Normalized Fourier descriptor magnitudes.
    """
    # Complex representation of boundary
    z = boundary_x + 1j * boundary_y
    # FFT
    Z = np.fft.fft(z)
    # Normalize by DC component
    if abs(Z[0]) > 0:
        Z_norm = np.abs(Z) / abs(Z[0])
    else:
        Z_norm = np.abs(Z)

    # Return first n_harmonics (skip DC)
    return Z_norm[1:n_harmonics + 1]


def reconstruct_3d_porosity(layer_masks: List[np.ndarray],
                            layer_depths_mm: np.ndarray,
                            pixel_size_mm: float = 0.01) -> dict:
    """
    Reconstruct 3D pore structure from multi-layer 2D scans.

    Parameters
    ----------
    layer_masks : List[np.ndarray]
        Binary masks (1=pore) for each scanned layer.
    layer_depths_mm : np.ndarray
        Depth of each layer in mm.
    pixel_size_mm : float
        Pixel size in mm.

    Returns
    -------
    dict with 3D porosity statistics.
    """
    n_layers = len(layer_masks)
    layer_porosities = []
    total_pore_voxels = 0
    total_voxels = 0

    for mask in layer_masks:
        pore_count = np.sum(mask > 0)
        total = mask.size
        layer_porosities.append(pore_count / total)
        total_pore_voxels += pore_count
        total_voxels += total

    voxel_volume = pixel_size_mm ** 2 * np.mean(np.diff(layer_depths_mm))
    bulk_porosity = total_pore_voxels / total_voxels

    return {
        "bulk_porosity_3d": bulk_porosity,
        "layer_porosities": np.array(layer_porosities),
        "n_layers": n_layers,
        "voxel_volume_mm3": voxel_volume,
        "total_pore_volume_mm3": total_pore_voxels * voxel_volume,
    }


def test_all():
    """Test all functions with synthetic data."""
    print("=" * 70)
    print("Testing: ultrasonic_pore_characterization (Chen et al., 2025)")
    print("=" * 70)

    # Test acoustic impedance
    Z_water = acoustic_impedance(1.0, 1480.0)
    Z_limestone = acoustic_impedance(2.71, 6400.0)
    print(f"  Z_water = {Z_water:.2f} MRayl")
    print(f"  Z_limestone = {Z_limestone:.2f} MRayl")
    rp = reflection_coefficient(Z_water, Z_limestone)
    print(f"  Reflection coefficient (water→limestone): {rp:.3f}")
    assert abs(rp) < 1.0

    # Test Otsu threshold on synthetic image
    rng = np.random.RandomState(42)
    img = rng.rand(100, 100)
    # Add synthetic pores (dark regions)
    img[20:30, 20:30] = 0.1
    img[50:65, 50:60] = 0.05
    thresh = otsu_threshold(img)
    print(f"  Otsu threshold: {thresh}")

    # Test pore segmentation
    binary = segment_pores((img * 255).astype(np.uint8))
    pore_fraction = np.mean(binary)
    print(f"  Segmented pore fraction: {pore_fraction:.3f}")

    # Test shape descriptors with a synthetic elliptical pore
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    bx = 20.0 * np.cos(theta)  # semi-major = 20
    by = 10.0 * np.sin(theta)  # semi-minor = 10
    morph = pore_shape_descriptors(bx, by)
    print(f"  Elliptical pore: area={morph.area:.1f}, perimeter={morph.perimeter:.1f}")
    print(f"    XZ={morph.shape_factor:.3f}, BP={morph.flatness_factor:.3f}, "
          f"JZ={morph.angular_factor:.3f}, TD={morph.convexity_factor:.3f}")
    assert 0 < morph.shape_factor <= 1.0

    # Test Fourier descriptors
    fd = fourier_descriptors(bx, by, n_harmonics=8)
    print(f"  Fourier descriptors (first 4): {fd[:4]}")
    assert len(fd) == 8

    # Test 3D reconstruction
    masks = [rng.choice([0, 1], size=(50, 50), p=[0.9, 0.1]) for _ in range(10)]
    depths = np.linspace(0, 1.0, 10)
    result_3d = reconstruct_3d_porosity(masks, depths)
    print(f"  3D bulk porosity: {result_3d['bulk_porosity_3d']:.3f}")
    assert 0 < result_3d["bulk_porosity_3d"] < 1.0

    print("  All tests PASSED.\n")


if __name__ == "__main__":
    test_all()
