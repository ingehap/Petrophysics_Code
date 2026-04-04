"""
Article 1: Enhanced Reservoir Characterization Using Drill-Cuttings-Based
Image and Elemental Analysis With AI: A Vaca Muerta Formation Case Study.

Authors: Kriscautzky, Oliver, Lugo, Marchal, and Naides (2026)
DOI: 10.30632/PJV67N1-2026a1

Implements a cuttings-based reservoir characterization workflow that
integrates high-resolution imaging (white light & UV), AI-driven image
analysis, and XRF elemental data to classify drill cuttings into lithotypes
and lithofacies associations.

Key ideas implemented:
    - RGB and YUV spectral extraction from drill-cuttings images
    - XRF elemental ratio computation (Si/Ca, detrital indicators)
    - AI-based lithotype clustering using physical and geochemical parameters
    - Tuffaceous interval detection via UV mineral luminance peaks
    - Contamination assessment via Ba concentration thresholds

References
----------
Kriscautzky et al. (2026), Petrophysics, 67(1), 9-26.
Oliver and McKnight (2022); Speight et al. (2023, 2024).
Rudnick and Gao (2014) for background Ba in Upper Continental Crust (~624 ppm).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CuttingsSample:
    """A single drill-cuttings sample with image and elemental data.

    Attributes
    ----------
    depth : float
        Measured depth of the sample (m).
    rgb : np.ndarray
        Mean RGB values from white-light image, shape (3,).
    yuv : np.ndarray
        Mean YUV values from UV image, shape (3,).
    xrf_elements : dict[str, float]
        Elemental concentrations in ppm from XRF analysis.
        Typical keys: 'Si', 'Ca', 'Al', 'Fe', 'K', 'Ti', 'Ba', 'S', 'Mn', etc.
    brightness : float
        Calibrated brightness from white-light image (0-255 scale).
    uv_luminance : float
        Calibrated UV luminance intensity.
    """
    depth: float
    rgb: np.ndarray
    yuv: np.ndarray
    xrf_elements: dict
    brightness: float = 0.0
    uv_luminance: float = 0.0
    lithotype: Optional[str] = None


def assess_contamination(ba_ppm: float,
                         threshold_preferred: float = 5000.0,
                         threshold_acceptable: float = 10000.0) -> str:
    """Assess mud contamination level from barium concentration.

    Barite (BaSO4) is commonly used as a weighting agent in drilling mud.
    Natural Ba in sedimentary rocks averages ~624 ppm (Rudnick and Gao, 2014).
    Elevated Ba indicates residual mud contamination.

    Parameters
    ----------
    ba_ppm : float
        Barium concentration in ppm from XRF.
    threshold_preferred : float
        Preferred maximum Ba (ppm), default 5000.
    threshold_acceptable : float
        Acceptable maximum Ba (ppm), default 10000.

    Returns
    -------
    str
        'clean', 'acceptable', or 'contaminated'.
    """
    if ba_ppm <= threshold_preferred:
        return "clean"
    elif ba_ppm <= threshold_acceptable:
        return "acceptable"
    else:
        return "contaminated"


def compute_elemental_ratios(elements: dict) -> dict:
    """Compute key geochemical ratios from XRF elemental data.

    Ratios implemented (as used in the Vaca Muerta study):
        - Si/Ca: siliciclastic vs. carbonate content indicator
        - Si/Al: detrital silica (quartz) vs. clay indicator
        - K/Al: clay composition (illite vs. kaolinite)
        - Fe/S: pyrite and anoxia indicator
        - Ti/Al: heavy mineral / volcanic ash indicator

    Parameters
    ----------
    elements : dict
        Element concentrations in ppm. Keys: 'Si', 'Ca', 'Al', 'Fe', 'K',
        'Ti', 'S', 'Mn', etc.

    Returns
    -------
    dict
        Computed elemental ratios.
    """
    ratios = {}

    def safe_ratio(num_key, den_key):
        num = elements.get(num_key, 0.0)
        den = elements.get(den_key, 0.0)
        return num / den if den > 0 else np.nan

    ratios["Si_Ca"] = safe_ratio("Si", "Ca")
    ratios["Si_Al"] = safe_ratio("Si", "Al")
    ratios["K_Al"] = safe_ratio("K", "Al")
    ratios["Fe_S"] = safe_ratio("Fe", "S")
    ratios["Ti_Al"] = safe_ratio("Ti", "Al")
    ratios["Ca_total"] = elements.get("Ca", 0.0) / (
        elements.get("Si", 0.0) + elements.get("Ca", 0.0) + elements.get("Al", 0.0) + 1e-10
    )

    return ratios


def extract_image_features(rgb: np.ndarray, yuv: np.ndarray,
                           brightness: float, uv_luminance: float) -> np.ndarray:
    """Extract feature vector from drill-cuttings image data.

    Combines RGB spectra, YUV spectra, calibrated brightness, and UV luminance
    into a single feature vector for lithotype classification.

    Parameters
    ----------
    rgb : np.ndarray
        Mean RGB values, shape (3,).
    yuv : np.ndarray
        Mean YUV values from UV image, shape (3,).
    brightness : float
        Calibrated white-light brightness (0-255).
    uv_luminance : float
        UV luminance intensity.

    Returns
    -------
    np.ndarray
        Feature vector, shape (8,).
    """
    return np.array([
        rgb[0], rgb[1], rgb[2],
        yuv[0], yuv[1], yuv[2],
        brightness,
        uv_luminance,
    ])


def detect_tuffaceous_intervals(uv_luminance: np.ndarray,
                                depths: np.ndarray,
                                threshold_factor: float = 2.0) -> np.ndarray:
    """Detect volcanic tuff layers from UV mineral luminance peaks.

    High mineral luminance peaks in UV images are interpreted as volcanic tuff
    layers, corroborated by XRD data showing igneous minerals and clay minerals
    formed from volcanic glass weathering.

    Parameters
    ----------
    uv_luminance : np.ndarray
        UV luminance values at each depth.
    depths : np.ndarray
        Measured depths corresponding to luminance values.
    threshold_factor : float
        Peaks exceeding mean + threshold_factor * std are flagged.

    Returns
    -------
    np.ndarray
        Boolean array; True at depths identified as tuffaceous.
    """
    mean_lum = np.mean(uv_luminance)
    std_lum = np.std(uv_luminance)
    threshold = mean_lum + threshold_factor * std_lum
    return uv_luminance > threshold


def classify_lithotypes(samples: list,
                        n_clusters: int = 22,
                        random_state: int = 42) -> np.ndarray:
    """Classify drill-cuttings samples into lithotypes using k-means clustering.

    The study identified 22 distinct lithotypes based on a combination of
    physical (RGB, brightness) and geochemical (elemental volumes, ratios)
    parameters. This function uses k-means as a simplified proxy for the
    proprietary AI algorithm described in the paper.

    Parameters
    ----------
    samples : list[CuttingsSample]
        List of CuttingsSample objects with image and XRF data.
    n_clusters : int
        Number of lithotype clusters (default 22 as in the study).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Cluster labels for each sample (0 to n_clusters-1).
    """
    feature_vectors = []
    for s in samples:
        img_feats = extract_image_features(s.rgb, s.yuv, s.brightness,
                                           s.uv_luminance)
        ratios = compute_elemental_ratios(s.xrf_elements)
        ratio_vec = np.array([ratios.get(k, 0.0) for k in
                              ["Si_Ca", "Si_Al", "K_Al", "Fe_S", "Ti_Al", "Ca_total"]])
        ratio_vec = np.nan_to_num(ratio_vec, nan=0.0)
        feature_vectors.append(np.concatenate([img_feats, ratio_vec]))

    X = np.array(feature_vectors)
    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-10
    X_norm = (X - mean) / std

    # Simple k-means implementation
    rng = np.random.RandomState(random_state)
    centroids = X_norm[rng.choice(len(X_norm), n_clusters, replace=False)]

    for _ in range(100):
        dists = np.linalg.norm(X_norm[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            X_norm[labels == k].mean(axis=0) if np.any(labels == k)
            else centroids[k]
            for k in range(n_clusters)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return labels


def group_lithofacies_associations(lithotype_labels: np.ndarray,
                                   depths: np.ndarray,
                                   window_size: int = 5) -> np.ndarray:
    """Group lithotypes into lithofacies associations using a moving window.

    Lithotypes are grouped into broader associations by analyzing the dominant
    lithotype within a moving depth window, identifying sedimentary packages.

    Parameters
    ----------
    lithotype_labels : np.ndarray
        Array of lithotype labels for each sample.
    depths : np.ndarray
        Measured depths for each sample.
    window_size : int
        Number of samples in the moving window.

    Returns
    -------
    np.ndarray
        Smoothed lithofacies association labels.
    """
    n = len(lithotype_labels)
    associations = np.zeros(n, dtype=int)
    half_w = window_size // 2

    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        window = lithotype_labels[start:end]
        # Dominant lithotype in window
        counts = np.bincount(window)
        associations[i] = np.argmax(counts)

    return associations
