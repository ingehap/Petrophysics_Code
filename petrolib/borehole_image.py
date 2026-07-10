"""Borehole-image analysis: bed sinusoids, dip picking, thresholding, texture.

Numerics shared by the borehole-image and image-log articles: the planar-bed
sinusoid crossing a cylindrical borehole and its least-squares inversion, dip
from sinusoid amplitude and apparent-dip projection, an SVD plane fit, Otsu
thresholding with class fractions, grey-level co-occurrence texture features,
and a Sobel gradient.

Azimuths and dips are in **degrees** at the interface (converted internally);
images are indexed ``[depth, azimuth]`` (rows = depth, columns = azimuth).

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2019_04/article4_image_segmentation_uncertainty -- Article 4: Uncertainty Quantification in
  Image Segmentation for Image-Based Rock Physics in a Shaly Sandstone. Howard, Lin, Zhang (2019).
  DOI: 10.30632/PJV60N2-2019a2. Petrophysics Vol. 60 No. 2 (Apr 2019).
src2020_10/article7_multiphysics_rock_classification -- Article 7: Integrated Multiphysics Workflow
  for Automatic Rock Classification and Formation Evaluation Using Multiscale Image Analysis and
  Conventional Well Logs. Gonzalez, Kanyan, Heidari, Lopez (2020). DOI: 10.30632/PJV61N5-2020a7.
  Petrophysics Vol. 61 No. 5 (Oct 2020).
src2021_06/article6_geosteering_2d_structural -- Article 6: Maximizing Net Pay in Penta-Lateral
  Well With Advanced Proactive Geosteering and 2D Structural Analysis Using Azimuthal Resistivity
  Measurements. Antonov, Kushnir, Martakov, Pazos, Small, Tropin, Maraj, Itter, Nelson, Rabinovich
  (2021). DOI: 10.30632/PJV62N3-2021a5. Petrophysics Vol. 62 No. 3 (Jun 2021).
src2021_10/article2_image_processing_petrophysics -- Article 2: Enhanced Learning of Fundamental
  Petrophysical Concepts Through Image Processing and 3D Printing. Alyafei, Al Musleh, Bautista,
  Idris, Seers (2021). DOI: 10.30632/PJV62N5-2021a2. Petrophysics Vol. 62 No. 5 (Oct 2021).
src2021_12/article04_borehole_image_cnn_sedimentary -- Article 4: Deep-Learning-Based Automated
  Sedimentary Geometry Characterization From Borehole Images. Lefranc, Bayraktar, Kristensen,
  Driss, Le Nir, Marza, Kherroubi (2021). DOI: 10.30632/PJV62N6-2021a4. Petrophysics Vol. 62 No. 6
  (Dec 2021).
src2025_04/ultrasonic_pore_characterization -- Ultrasonic Microscopy Imaging of Carbonate Reservoir
  Pore Structure. Based on: Chen et al., "New Methodology for Ultrasonic Microscopy Imaging of
  Carbonate Reservoirs' Pore Structure", Petrophysics, Vol. 66, No. 2, April 2025, pp. 267–282.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


# --- dip geometry -------------------------------------------------------------


def bed_sinusoid(
    azimuth_deg: ArrayLike,
    z0: float,
    radius: float,
    dip_deg: float,
    dip_azimuth_deg: float,
) -> _Float:
    """Depth trace of a planar bed on the unrolled borehole wall.

    ``z(phi) = z0 - radius*tan(dip)*cos(phi - dip_azimuth)`` -- a sinusoid of
    amplitude ``radius*tan(dip)`` whose deepest point is at the dip azimuth.
    All angles in degrees.

    Sources: src2021_12/article04_borehole_image_cnn_sedimentary.
    """
    phi = np.radians(_arr(azimuth_deg))
    a = np.radians(dip_azimuth_deg)
    return np.asarray(z0 - radius * np.tan(np.radians(dip_deg)) * np.cos(phi - a))


def fit_sinusoid(
    azimuth_deg: ArrayLike, z: ArrayLike, *, mask: ArrayLike | None = None
) -> tuple[float, float, float]:
    """Least-squares sinusoid fit on ``[1, cos, sin]`` -> ``(z0, amplitude, phase_deg)``.

    Fits ``z = z0 + c1*cos(phi) + c2*sin(phi)`` (``phi`` from ``azimuth_deg``),
    returning the offset ``z0``, ``amplitude = hypot(c1, c2)`` and
    ``phase_deg = atan2(c2, c1)``.  An optional boolean ``mask`` selects valid
    samples.
    """
    phi = np.radians(_arr(azimuth_deg))
    zz = _arr(z)
    if mask is not None:
        m = np.asarray(mask, bool)
        phi = phi[m]
        zz = zz[m]
    g = np.column_stack([np.ones_like(phi), np.cos(phi), np.sin(phi)])
    coef, *_ = np.linalg.lstsq(g, zz, rcond=None)
    c0, c1, c2 = float(coef[0]), float(coef[1]), float(coef[2])
    amplitude = float(np.hypot(c1, c2))
    phase = float(np.degrees(np.arctan2(c2, c1)))
    return c0, amplitude, phase


def dip_from_amplitude(amplitude: float, radius: float, *, sample_spacing: float = 1.0) -> float:
    """True dip from a sinusoid amplitude: ``arctan(amplitude*sample_spacing/radius)`` (deg)."""
    return float(np.degrees(np.arctan(amplitude * sample_spacing / radius)))


def apparent_dip(true_dip_deg: float, section_azimuth_deg: float) -> float:
    """Apparent dip at section angle ``beta`` to true dip: ``tan(app)=tan(true)cos(beta)`` (deg).

    Sources: src2021_06/article6_geosteering_2d_structural.
    """
    return float(
        np.degrees(
            np.arctan(np.tan(np.radians(true_dip_deg)) * np.cos(np.radians(section_azimuth_deg)))
        )
    )


def fit_plane(points_enz: ArrayLike) -> tuple[float, float]:
    """SVD plane fit through ``(E, N, TVD)`` points -> ``(dip_deg, dip_azimuth_deg)``.

    The plane normal is the smallest-singular-value direction of the centred
    points; ``dip = arccos(|n_z|)`` and ``dip_azimuth = atan2(n_E, n_N) mod 360``
    (clockwise from North).

    Sources: src2021_06/article6_geosteering_2d_structural.
    """
    p = _arr(points_enz)
    centroid = p.mean(axis=0)
    _, _, vh = np.linalg.svd(p - centroid)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    dip = float(np.degrees(np.arccos(abs(n[2]))))
    az = float(np.degrees(np.arctan2(n[0], n[1])) % 360.0)
    return dip, az


# --- thresholding / segmentation ---------------------------------------------


def otsu_threshold(image: ArrayLike, *, bins: int = 256) -> float:
    """Otsu threshold on the image's native data range, returning a bin-centre value.

    Maximises the between-class variance
    ``sigma_b^2(t) = (mu_T*omega - mu)^2 / [omega(1-omega)]`` over the histogram
    of ``image`` (``bins`` bins spanning ``[min, max]``) and returns the value at
    the optimal bin centre.

    Sources: src2019_04/article4_image_segmentation_uncertainty.
    """
    img = _arr(image).ravel()
    hist, edges = np.histogram(img, bins=bins, range=(float(img.min()), float(img.max())))
    p = hist / hist.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    return float(centers[np.argmax(sigma_b2)])


def class_fractions(image: ArrayLike, thresholds: ArrayLike) -> _Float:
    """Volume fractions between ascending ``thresholds``.

    For ``thresholds = [t0, t1, ...]`` returns ``[mean(img<t0),
    mean(t0<=img<t1), ..., mean(img>=t_last)]`` -- e.g. pore / clay / grain from
    two thresholds (dark = pore convention).

    Sources: src2019_04/article4_image_segmentation_uncertainty.
    """
    img = _arr(image).ravel()
    fracs = []
    lo = -np.inf
    for t in np.asarray(thresholds, float):
        fracs.append(float(np.mean((img >= lo) & (img < t))))
        lo = t
    fracs.append(float(np.mean(img >= lo)))
    return np.asarray(fracs)


def phase_saturation(phase: ArrayLike, pore: ArrayLike) -> float:
    """Saturation of a phase within the pore space: ``|phase & pore| / |pore|``.

    Sources: src2021_10/article2_image_processing_petrophysics.
    """
    ph = np.asarray(phase)
    po = np.asarray(pore)
    return float(np.logical_and(ph, po).sum()) / float(po.sum())


def porosity_from_mask(pore: ArrayLike) -> float:
    """Porosity as the fraction of voxels flagged pore: ``pore.sum()/pore.size``.

    Sources: src2019_04/article4_image_segmentation_uncertainty.
    """
    m = np.asarray(pore)
    return float(m.sum()) / float(m.size)


# --- texture ------------------------------------------------------------------


def glcm(
    image: ArrayLike, *, levels: int = 16, offset: tuple[int, int] = (0, 1), symmetric: bool = True
) -> _Float:
    """Normalised grey-level co-occurrence matrix.

    Quantises ``image`` to ``levels`` grey levels by min-max scaling, then counts
    co-occurrences at ``offset = (drow, dcol)``.  ``symmetric=True`` counts both
    directions; the result is normalised to sum to one.

    Sources: src2020_10/article7_multiphysics_rock_classification.
    """
    img = _arr(image)
    rng = img.max() - img.min()
    q = np.clip((img - img.min()) / (rng + 1e-12) * (levels - 1), 0, levels - 1).astype(int)
    g = np.zeros((levels, levels))
    dr, dc = offset
    nr, nc = q.shape
    for r in range(nr - max(dr, 0)):
        for c in range(nc - max(dc, 0)):
            i, j = q[r, c], q[r + dr, c + dc]
            g[i, j] += 1.0
            if symmetric:
                g[j, i] += 1.0
    s = g.sum()
    return np.asarray(g / s if s > 0 else g)


def glcm_features(p: ArrayLike) -> dict[str, float]:
    """Haralick contrast / energy / correlation of a normalised GLCM ``p``.

    Sources: src2020_10/article7_multiphysics_rock_classification.
    """
    p_arr = _arr(p)
    n = p_arr.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    contrast = float(np.sum(p_arr * (i - j) ** 2))
    energy = float(np.sum(p_arr**2))
    mu_i = float(np.sum(i * p_arr))
    mu_j = float(np.sum(j * p_arr))
    var_i = float(np.sum(p_arr * (i - mu_i) ** 2))
    var_j = float(np.sum(p_arr * (j - mu_j) ** 2))
    denom = np.sqrt(var_i * var_j)
    correlation = float(np.sum(p_arr * (i - mu_i) * (j - mu_j)) / denom) if denom > 0 else 0.0
    return {"contrast": contrast, "energy": energy, "correlation": correlation}


def sobel_gradient(image: ArrayLike) -> tuple[_Float, _Float, _Float]:
    """3x3 Sobel gradient -> ``(gx, gy, magnitude)`` with zero borders.

    ``gx`` is the horizontal (column) derivative, ``gy`` the vertical (row)
    derivative; ``magnitude = sqrt(gx^2 + gy^2)``.  Interior pixels only; the
    one-pixel border stays zero.

    Sources: src2025_04/ultrasonic_pore_characterization.
    """
    img = _arr(image)
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    xstack = np.stack(
        [
            img[a : a + img.shape[0] - 2, b : b + img.shape[1] - 2] * kx[a, b]
            for a in range(3)
            for b in range(3)
        ],
        axis=-1,
    )
    ystack = np.stack(
        [
            img[a : a + img.shape[0] - 2, b : b + img.shape[1] - 2] * ky[a, b]
            for a in range(3)
            for b in range(3)
        ],
        axis=-1,
    )
    gx[1:-1, 1:-1] = np.sum(xstack, axis=-1)
    gy[1:-1, 1:-1] = np.sum(ystack, axis=-1)
    magnitude = np.sqrt(gx**2 + gy**2)
    return gx, gy, magnitude
