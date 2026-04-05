#!/usr/bin/env python3
"""
Article 11: Automatic Fracture Identifications From Image Logs With
            Machine-Learning Approaches: A Contest Summary
Authors: Hyungjoo Lee, Ramin Zamani, Lei Fu, et al.
Ref: Petrophysics, Vol. 66, No. 5 (October 2025), pp. 894-914.
     DOI: 10.30632/PJV66N5-2025a11

Implements:
  - Synthetic borehole image-log generation with sinusoidal fractures
  - Feature extraction from image logs (gradient, sinusoid fitting)
  - Simple CNN-based fracture detector (NumPy)
  - Evaluation metrics: F1 score with depth-tolerance threshold
  - RMSE between predicted and true fracture depths
"""

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic borehole image generation
# ---------------------------------------------------------------------------

def generate_borehole_image(n_depths=500, n_azimuths=360, n_fractures=15,
                             noise_level=0.05, seed=42):
    """Generate a synthetic resistivity borehole image with fractures.

    Fractures appear as sinusoidal features: depth(az) = d0 + A*sin(az + phase)
    where d0 is the center depth, A is the amplitude, and phase is the dip azimuth.

    Returns
    -------
    image        : (n_depths, n_azimuths) resistivity image
    fracture_depths : list of true fracture center depths (indices)
    """
    rng = np.random.default_rng(seed)
    image = 1.0 + 0.3 * rng.standard_normal((n_depths, n_azimuths))

    # Background bedding (horizontal lines)
    for i in range(0, n_depths, 20):
        image[i, :] += 0.2

    fracture_depths = sorted(rng.choice(range(20, n_depths - 20), n_fractures,
                                        replace=False))
    az = np.linspace(0, 2 * np.pi, n_azimuths, endpoint=False)

    for d0 in fracture_depths:
        amplitude = rng.uniform(2, 10)
        phase = rng.uniform(0, 2 * np.pi)
        # Sinusoidal fracture trace
        trace = d0 + amplitude * np.sin(az + phase)
        for j, depth_j in enumerate(trace):
            di = int(round(depth_j))
            if 0 <= di < n_depths:
                for dd in range(-1, 2):
                    if 0 <= di + dd < n_depths:
                        image[di + dd, j] -= 0.8  # conductive fracture

    image += noise_level * rng.standard_normal(image.shape)
    return image, fracture_depths


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def vertical_gradient(image):
    """Vertical gradient magnitude of the image."""
    grad = np.diff(image, axis=0)
    # Pad to match original size
    grad = np.vstack([grad, grad[-1:]])
    return np.abs(grad)


def azimuthal_variance(image, window=5):
    """Variance of each depth row (captures sinusoidal anomalies)."""
    n = image.shape[0]
    var = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        var[i] = image[lo:hi].var()
    return var


def extract_features(image, window=5):
    """Extract per-depth features for fracture classification.

    Returns (n_depths, n_features) array.
    """
    grad = vertical_gradient(image)
    grad_mean = grad.mean(axis=1)
    grad_max = grad.max(axis=1)
    az_var = azimuthal_variance(image, window)
    row_min = image.min(axis=1)
    row_std = image.std(axis=1)
    return np.column_stack([grad_mean, grad_max, az_var, row_min, row_std])


# ---------------------------------------------------------------------------
# Simple threshold-based fracture detector
# ---------------------------------------------------------------------------

def detect_fractures_threshold(features, grad_threshold=0.3,
                                var_threshold=None):
    """Detect fracture depths using gradient + variance thresholds.

    Returns array of predicted fracture depth indices.
    """
    grad_max = features[:, 1]
    if var_threshold is None:
        var_threshold = np.percentile(features[:, 2], 90)

    candidates = (grad_max > grad_threshold) | (features[:, 2] > var_threshold)
    # Non-maximum suppression (merge nearby detections)
    detected = []
    in_cluster = False
    cluster_start = 0
    for i in range(len(candidates)):
        if candidates[i]:
            if not in_cluster:
                cluster_start = i
                in_cluster = True
        else:
            if in_cluster:
                center = (cluster_start + i - 1) // 2
                detected.append(center)
                in_cluster = False
    if in_cluster:
        detected.append((cluster_start + len(candidates) - 1) // 2)
    return np.array(detected)


# ---------------------------------------------------------------------------
# Simple CNN-based detector (NumPy, single-layer)
# ---------------------------------------------------------------------------

class SimpleFractureDetector:
    """1-D CNN with trainable kernel for fracture detection."""

    def __init__(self, n_features=5, kernel_size=7, seed=42):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((kernel_size, n_features)) * 0.1
        self.b = 0.0
        self.ks = kernel_size

    def predict_proba(self, features):
        """Predict fracture probability at each depth."""
        n = features.shape[0]
        pad = self.ks // 2
        fp = np.pad(features, ((pad, pad), (0, 0)), mode='edge')
        scores = np.zeros(n)
        for i in range(n):
            patch = fp[i:i + self.ks]
            scores[i] = np.sum(patch * self.W) + self.b
        return 1.0 / (1.0 + np.exp(-scores))  # sigmoid

    def detect(self, features, threshold=0.5):
        proba = self.predict_proba(features)
        return np.where(proba > threshold)[0]


# ---------------------------------------------------------------------------
# Evaluation metrics (from article)
# ---------------------------------------------------------------------------

def f1_score_with_tolerance(true_depths, pred_depths, alpha=1.0):
    """F1 score with depth tolerance alpha (in depth index units).

    A prediction is a true positive if it falls within ±alpha of a true depth.
    """
    if len(pred_depths) == 0 and len(true_depths) == 0:
        return 1.0
    if len(pred_depths) == 0 or len(true_depths) == 0:
        return 0.0

    true_set = set(true_depths)
    tp = 0
    matched = set()
    for p in pred_depths:
        for t in true_set:
            if abs(p - t) <= alpha and t not in matched:
                tp += 1
                matched.add(t)
                break

    precision = tp / len(pred_depths) if len(pred_depths) > 0 else 0
    recall = tp / len(true_depths) if len(true_depths) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def rmse_depths(true_depths, pred_depths):
    """RMSE between matched true and predicted fracture depths."""
    if len(true_depths) == 0 or len(pred_depths) == 0:
        return float('inf')
    errors = []
    for t in true_depths:
        dists = np.abs(np.array(pred_depths) - t)
        errors.append(dists.min())
    return np.sqrt(np.mean(np.array(errors) ** 2))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== Article 11: Fracture Identification Demo ===\n")

    image, true_depths = generate_borehole_image(n_depths=500, n_fractures=12)
    features = extract_features(image)

    pred = detect_fractures_threshold(features, grad_threshold=0.25)
    f1 = f1_score_with_tolerance(true_depths, pred, alpha=3)
    rmse_val = rmse_depths(true_depths, pred)

    print(f"Image shape     : {image.shape}")
    print(f"True fractures  : {len(true_depths)} at depths {true_depths[:5]}...")
    print(f"Detected        : {len(pred)}")
    print(f"F1 (alpha=3)    : {f1:.3f}")
    print(f"RMSE            : {rmse_val:.2f} depth units")

    # CNN detector
    cnn = SimpleFractureDetector(n_features=5, kernel_size=7)
    pred_cnn = cnn.detect(features, threshold=0.5)
    f1_cnn = f1_score_with_tolerance(true_depths, pred_cnn, alpha=3)
    print(f"\nCNN detector F1 : {f1_cnn:.3f} (untrained, random weights)")
    print()


if __name__ == "__main__":
    demo()
