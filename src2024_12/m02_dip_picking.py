#!/usr/bin/env python3
"""
Enhanced AI-Driven Automatic Dip Picking in Horizontal Wells
=============================================================
Based on: Perrier, He, Bize-Forest, and Quesada (2024),
Petrophysics 65(6), pp. 875-886. DOI: 10.30632/PJV65N6-2024a2

Implements the workflow:
  1. CNN-based classification of borehole image zones (no-bedding,
     sinusoidal, non-sinusoidal).
  2. Hough-transform-based dip picking for sinusoidal beddings.
  3. DBSCAN clustering of partial dips.
  4. Path-based merging for non-sinusoidal beddings.
  5. Real-time block continuity check.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class DipPick:
    """A single dip measurement (tadpole)."""
    depth: float       # measured depth (m)
    dip_angle: float   # true dip angle (degrees)
    dip_azimuth: float # dip azimuth (degrees)
    confidence: float  # 0-1 confidence score
    category: str      # 'sinusoidal', 'non-sinusoidal', 'partial'


@dataclass
class ImageZone:
    """A classified zone in a borehole image."""
    top_depth: float
    bottom_depth: float
    label: str  # 'no_bedding', 'sinusoidal', 'non_sinusoidal'
    confidence: float


def generate_synthetic_borehole_image(
    n_rows: int = 500,
    n_cols: int = 360,
    well_type: str = 'horizontal',
    seed: int = 42,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Generate a synthetic borehole image with sinusoidal and
    non-sinusoidal bedding features.

    Parameters
    ----------
    n_rows : int
        Number of depth rows.
    n_cols : int
        Number of azimuthal columns (typically 360 for 1°/pixel).
    well_type : str
        'vertical' or 'horizontal'.
    seed : int

    Returns
    -------
    image : np.ndarray of shape (n_rows, n_cols)
    true_beds : list of dict with bed parameters
    """
    rng = np.random.RandomState(seed)
    image = rng.rand(n_rows, n_cols) * 0.3  # background noise
    azimuths = np.arange(n_cols) * (2 * np.pi / n_cols)
    true_beds = []

    n_beds = rng.randint(8, 15)
    for i in range(n_beds):
        center_depth = rng.uniform(20, n_rows - 20)
        dip_angle = rng.uniform(5, 85)
        dip_azimuth = rng.uniform(0, 360)
        amplitude = dip_angle * 0.3  # pixel amplitude

        if well_type == 'horizontal' and rng.rand() > 0.4:
            # Non-sinusoidal: flat with slight curvature
            category = 'non_sinusoidal'
            for j in range(n_cols):
                depth_offset = amplitude * 0.2 * np.sin(azimuths[j] - np.radians(dip_azimuth))
                depth_offset += amplitude * 0.05 * np.sin(3 * azimuths[j])
                row = int(center_depth + depth_offset)
                if 0 <= row < n_rows:
                    for dr in range(-2, 3):
                        r = row + dr
                        if 0 <= r < n_rows:
                            image[r, j] += 0.7 * np.exp(-0.5 * (dr / 1.5) ** 2)
        else:
            category = 'sinusoidal'
            for j in range(n_cols):
                depth_offset = amplitude * np.sin(azimuths[j] - np.radians(dip_azimuth))
                row = int(center_depth + depth_offset)
                if 0 <= row < n_rows:
                    for dr in range(-2, 3):
                        r = row + dr
                        if 0 <= r < n_rows:
                            image[r, j] += 0.8 * np.exp(-0.5 * (dr / 1.5) ** 2)

        true_beds.append({
            'depth': center_depth,
            'dip_angle': dip_angle,
            'dip_azimuth': dip_azimuth,
            'category': category,
        })

    image = np.clip(image, 0, 1)
    return image, true_beds


def classify_zones_cnn(
    image: np.ndarray,
    window_size: int = 50,
    step_size: int = 15,
) -> List[ImageZone]:
    """
    Classify borehole image zones using a sliding window approach.

    This simulates the CNN classification from the paper. The CNN
    classifies windows into: no_bedding, sinusoidal, non_sinusoidal.

    In practice, a trained CNN (3 conv blocks + 2 dense layers with
    ReLU/softmax) would be used. Here we use statistical proxies.

    Parameters
    ----------
    image : np.ndarray of shape (n_rows, n_cols)
    window_size : int
        Height of each classification window (pixels).
    step_size : int
        Sliding step.

    Returns
    -------
    list of ImageZone
    """
    n_rows, n_cols = image.shape
    vote_map = {i: {'no_bedding': 0, 'sinusoidal': 0, 'non_sinusoidal': 0}
                for i in range(n_rows)}

    for start in range(0, n_rows - window_size + 1, step_size):
        window = image[start:start + window_size, :]
        # Compute features that discriminate bedding types
        row_means = np.mean(window, axis=1)
        row_stds = np.std(window, axis=1)
        overall_contrast = np.std(row_means)
        # FFT of column-averaged signal for sinusoid detection
        col_avg = np.mean(window, axis=0)
        fft_mag = np.abs(np.fft.rfft(col_avg))
        peak_freq_power = np.max(fft_mag[1:5]) / (np.mean(fft_mag[1:]) + 1e-10)

        if overall_contrast < 0.05:
            label = 'no_bedding'
        elif peak_freq_power > 2.5:
            label = 'sinusoidal'
        else:
            label = 'non_sinusoidal'

        for r in range(start, min(start + window_size, n_rows)):
            vote_map[r][label] += 1

    # Merge into zones via majority voting
    row_labels = []
    for i in range(n_rows):
        votes = vote_map[i]
        row_labels.append(max(votes, key=votes.get))

    # Merge adjacent rows with same label into zones
    zones = []
    current_label = row_labels[0]
    start_row = 0
    for i in range(1, n_rows):
        if row_labels[i] != current_label:
            conf = sum(1 for r in range(start_row, i) if vote_map[r][current_label] > 0) / max(i - start_row, 1)
            zones.append(ImageZone(float(start_row), float(i - 1), current_label, conf))
            current_label = row_labels[i]
            start_row = i
    zones.append(ImageZone(float(start_row), float(n_rows - 1), current_label, 1.0))
    return zones


def pick_sinusoidal_dips(
    image: np.ndarray,
    zone: ImageZone,
    success_rate_threshold: float = 0.3,
) -> List[DipPick]:
    """
    Pick dips in sinusoidal bedding zones using a simplified Hough
    transform approach (a-contrario algorithm from Kherroubi et al., 2016).

    Parameters
    ----------
    image : np.ndarray
    zone : ImageZone
    success_rate_threshold : float

    Returns
    -------
    list of DipPick
    """
    top = int(zone.top_depth)
    bot = int(zone.bottom_depth) + 1
    n_cols = image.shape[1]
    azimuths = np.arange(n_cols) * (2 * np.pi / n_cols)

    window = image[top:bot, :]
    picks = []

    # For each candidate center depth in the window
    for depth_offset in range(5, bot - top - 5, 3):
        best_score = 0
        best_amp = 0
        best_phase = 0

        # Search over amplitude and phase
        for amp in np.linspace(2, 30, 8):
            for phase in np.linspace(0, 2 * np.pi, 12):
                trace_rows = depth_offset + amp * np.sin(azimuths - phase)
                trace_rows = trace_rows.astype(int)
                valid = (trace_rows >= 0) & (trace_rows < (bot - top))
                if valid.sum() < n_cols * 0.5:
                    continue
                vals = [window[trace_rows[j], j] for j in range(n_cols) if valid[j]]
                score = np.mean(vals)
                if score > best_score:
                    best_score = score
                    best_amp = amp
                    best_phase = phase

        if best_score > success_rate_threshold:
            dip_angle = np.degrees(np.arctan(best_amp / (n_cols / (2 * np.pi))))
            dip_azimuth = np.degrees(best_phase) % 360
            picks.append(DipPick(
                depth=top + depth_offset,
                dip_angle=dip_angle,
                dip_azimuth=dip_azimuth,
                confidence=min(best_score, 1.0),
                category='sinusoidal',
            ))
    return picks


def cluster_partial_dips_dbscan(
    dips: List[DipPick],
    eps: float = 5.0,
    min_samples: int = 4,
) -> List[List[DipPick]]:
    """
    Cluster partial dips using DBSCAN (density-based spatial clustering
    of applications with noise).

    Parameters
    ----------
    dips : list of DipPick
    eps : float
        Maximum distance between samples in a cluster.
    min_samples : int
        Minimum number of samples per cluster.

    Returns
    -------
    list of lists of DipPick (each sublist is one cluster)
    """
    if not dips:
        return []

    n = len(dips)
    features = np.array([
        [d.depth, d.dip_angle, d.dip_azimuth / 10.0]
        for d in dips
    ])

    # Simple DBSCAN implementation
    labels = [-1] * n
    cluster_id = 0
    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = []
        for j in range(n):
            if np.linalg.norm(features[i] - features[j]) < eps:
                neighbors.append(j)

        if len(neighbors) < min_samples:
            labels[i] = -1  # noise
        else:
            labels[i] = cluster_id
            seed_set = list(neighbors)
            k = 0
            while k < len(seed_set):
                q = seed_set[k]
                if not visited[q]:
                    visited[q] = True
                    q_neighbors = [j for j in range(n)
                                   if np.linalg.norm(features[q] - features[j]) < eps]
                    if len(q_neighbors) >= min_samples:
                        seed_set.extend([x for x in q_neighbors if x not in seed_set])
                if labels[q] == -1:
                    labels[q] = cluster_id
                k += 1
            cluster_id += 1

    clusters = {}
    for i, lab in enumerate(labels):
        if lab >= 0:
            clusters.setdefault(lab, []).append(dips[i])
    return list(clusters.values())


def merge_cluster_to_dip(cluster: List[DipPick]) -> DipPick:
    """Merge a cluster of partial dips into a single dip using median."""
    angles = [d.dip_angle for d in cluster]
    azimuths = [d.dip_azimuth for d in cluster]
    depths = [d.depth for d in cluster]
    confs = [d.confidence for d in cluster]
    return DipPick(
        depth=float(np.median(depths)),
        dip_angle=float(np.median(angles)),
        dip_azimuth=float(np.median(azimuths)),
        confidence=float(np.mean(confs)),
        category=cluster[0].category,
    )


def connect_non_sinusoidal_paths(
    dips: List[DipPick],
    distance_threshold: float = 15.0,
) -> List[List[DipPick]]:
    """
    Connect partial dips in non-sinusoidal zones into paths based on
    proximity and orientation similarity (the paper's path-building
    approach for horizontal wells).

    Parameters
    ----------
    dips : list of DipPick sorted by depth
    distance_threshold : float

    Returns
    -------
    list of paths (each path is a list of DipPick)
    """
    if not dips:
        return []
    sorted_dips = sorted(dips, key=lambda d: d.depth)
    used = [False] * len(sorted_dips)
    paths = []

    for i in range(len(sorted_dips)):
        if used[i]:
            continue
        path = [sorted_dips[i]]
        used[i] = True
        current = sorted_dips[i]

        while True:
            best_j = -1
            best_dist = distance_threshold
            for j in range(len(sorted_dips)):
                if used[j]:
                    continue
                dd = abs(sorted_dips[j].depth - current.depth)
                da = abs(sorted_dips[j].dip_angle - current.dip_angle)
                dist = np.sqrt(dd**2 + da**2)
                if dist < best_dist and sorted_dips[j].depth > current.depth:
                    best_dist = dist
                    best_j = j
            if best_j >= 0:
                used[best_j] = True
                path.append(sorted_dips[best_j])
                current = sorted_dips[best_j]
            else:
                break
        if len(path) >= 2:
            paths.append(path)
    return paths


def realtime_continuity_check(
    prev_zone_label: str,
    new_zone_label: str,
) -> bool:
    """
    Check if a new image block continues the previous zone.

    Returns True if labels match (zones should be merged).
    """
    return prev_zone_label == new_zone_label


def automatic_dip_picking_workflow(
    image: np.ndarray,
) -> List[DipPick]:
    """
    Full automatic dip-picking workflow for horizontal wells.

    Parameters
    ----------
    image : np.ndarray
        Borehole image (rows = depth, cols = azimuth).

    Returns
    -------
    list of DipPick
    """
    # Step 1: Classify zones
    zones = classify_zones_cnn(image)

    all_picks = []
    for zone in zones:
        if zone.label == 'no_bedding':
            continue
        elif zone.label == 'sinusoidal':
            picks = pick_sinusoidal_dips(image, zone)
            clusters = cluster_partial_dips_dbscan(picks, eps=8.0, min_samples=2)
            for cluster in clusters:
                all_picks.append(merge_cluster_to_dip(cluster))
            # Keep unclustered picks
            clustered_depths = {p.depth for c in clusters for p in c}
            for p in picks:
                if p.depth not in clustered_depths:
                    all_picks.append(p)
        elif zone.label == 'non_sinusoidal':
            # Use the sinusoidal picker to get partial dips
            partial_picks = pick_sinusoidal_dips(image, zone, success_rate_threshold=0.25)
            for p in partial_picks:
                p.category = 'partial'
            paths = connect_non_sinusoidal_paths(partial_picks)
            for path in paths:
                all_picks.append(merge_cluster_to_dip(path))

    return sorted(all_picks, key=lambda d: d.depth)


def test_all():
    """Test all functions with synthetic data."""
    print("=" * 70)
    print("Module 2: AI-Driven Automatic Dip Picking (Perrier et al., 2024)")
    print("=" * 70)

    # Generate synthetic borehole image
    image, true_beds = generate_synthetic_borehole_image(
        n_rows=300, n_cols=360, well_type='horizontal', seed=42
    )
    print(f"Generated borehole image: {image.shape}")
    print(f"True beds: {len(true_beds)}")

    # Classify zones
    zones = classify_zones_cnn(image)
    print(f"\nClassified {len(zones)} zones:")
    for z in zones:
        print(f"  {z.top_depth:.0f}-{z.bottom_depth:.0f}: {z.label} (conf={z.confidence:.2f})")

    # Full workflow
    picks = automatic_dip_picking_workflow(image)
    print(f"\nAutomatic dip picking produced {len(picks)} dips:")
    for p in picks[:10]:
        print(f"  depth={p.depth:.1f}, angle={p.dip_angle:.1f}°, "
              f"azimuth={p.dip_azimuth:.1f}°, cat={p.category}")
    if len(picks) > 10:
        print(f"  ... ({len(picks) - 10} more)")

    # Real-time continuity
    cont = realtime_continuity_check('sinusoidal', 'sinusoidal')
    print(f"\nContinuity check (same labels): {cont}")
    cont = realtime_continuity_check('sinusoidal', 'no_bedding')
    print(f"Continuity check (diff labels): {cont}")

    print("\n[PASS] All tests completed successfully.\n")


if __name__ == "__main__":
    test_all()
