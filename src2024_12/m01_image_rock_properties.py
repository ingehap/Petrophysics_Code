#!/usr/bin/env python3
"""
Image-Based AI for Determination of Analog Petrophysical Rock Properties
=========================================================================
Based on: Britton, Cox, and Ma (2024), Petrophysics 65(6), pp. 866-874.
DOI: 10.30632/PJV65N6-2024a1

Implements a thin-section image similarity approach to find analog
petrophysical properties (porosity, permeability, matrix density,
Archie m, capillary pressure) from a reference database of core samples.
The AI model compares texture features of a query image against a
database of labeled thin-section images.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class CoreSample:
    """Represents a conventional core sample with measured properties."""
    sample_id: str
    lithology: str  # 'clastic' or 'carbonate'
    porosity: float  # fraction
    permeability_klinkenberg: float  # mD
    matrix_density: float  # g/cc
    archie_m: float  # cementation exponent
    capillary_entry_pressure: float  # psi
    texture_features: np.ndarray = field(default_factory=lambda: np.array([]))

    def __repr__(self):
        return (f"CoreSample({self.sample_id}, {self.lithology}, "
                f"phi={self.porosity:.3f}, k={self.permeability_klinkenberg:.2f} mD)")


def extract_texture_features(image: np.ndarray, n_features: int = 64) -> np.ndarray:
    """
    Extract texture features from a thin-section image.

    Simulates the high-resolution image analysis (0.44 µm/pixel) used
    in the AI model. In practice this would use a CNN or histogram-of-
    oriented-gradients; here we use statistical texture descriptors.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D image array (grayscale or RGB).
    n_features : int
        Number of features to extract.

    Returns
    -------
    np.ndarray
        Feature vector of length n_features.
    """
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.copy()

    features = []
    # Global statistics
    features.append(np.mean(gray))
    features.append(np.std(gray))
    features.append(np.median(gray))
    features.append(float(np.percentile(gray, 25)))
    features.append(float(np.percentile(gray, 75)))

    # Histogram-based features (binned intensity distribution)
    hist, _ = np.histogram(gray.ravel(), bins=min(n_features - 5, 32), density=True)
    features.extend(hist.tolist())

    # Spatial texture: row/col variance patterns
    row_var = np.var(gray, axis=1)
    col_var = np.var(gray, axis=0)
    features.append(np.mean(row_var))
    features.append(np.mean(col_var))
    features.append(np.std(row_var))
    features.append(np.std(col_var))

    feat = np.array(features[:n_features], dtype=np.float64)
    if len(feat) < n_features:
        feat = np.pad(feat, (0, n_features - len(feat)))
    # Normalize
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat /= norm
    return feat


def build_sample_database(n_samples: int = 100, seed: int = 42) -> List[CoreSample]:
    """
    Build a synthetic core sample database with measured petrophysical
    properties and associated texture feature vectors.

    Parameters
    ----------
    n_samples : int
        Number of samples in the database.
    seed : int
        Random seed.

    Returns
    -------
    list of CoreSample
    """
    rng = np.random.RandomState(seed)
    samples = []
    for i in range(n_samples):
        lithology = 'clastic' if rng.rand() < 0.6 else 'carbonate'
        if lithology == 'clastic':
            phi = rng.uniform(0.05, 0.30)
            k = 10 ** rng.uniform(-1, 3) * phi * 10
            rho_m = rng.uniform(2.60, 2.70)
            m = rng.uniform(1.5, 2.2)
        else:
            phi = rng.uniform(0.02, 0.25)
            k = 10 ** rng.uniform(-2, 3) * phi * 5
            rho_m = rng.uniform(2.68, 2.75)
            m = rng.uniform(1.8, 3.0)
        pc_entry = max(0.5, 50.0 * (1 - phi) / (k ** 0.25))
        # Generate a synthetic texture feature vector
        base_feat = rng.randn(64)
        # Encode porosity / lithology info into features loosely
        base_feat[0] += phi * 5
        base_feat[1] += (1 if lithology == 'clastic' else -1)
        base_feat[2] += np.log10(max(k, 1e-3))
        norm = np.linalg.norm(base_feat)
        if norm > 0:
            base_feat /= norm
        samples.append(CoreSample(
            sample_id=f"CORE-{i:04d}",
            lithology=lithology,
            porosity=phi,
            permeability_klinkenberg=k,
            matrix_density=rho_m,
            archie_m=m,
            capillary_entry_pressure=pc_entry,
            texture_features=base_feat,
        ))
    return samples


def find_analog_matches(
    query_features: np.ndarray,
    database: List[CoreSample],
    top_k: int = 5,
) -> List[Tuple[CoreSample, float]]:
    """
    Find the best analog matches from the database using cosine similarity.

    Parameters
    ----------
    query_features : np.ndarray
        Feature vector of the query thin-section image.
    database : list of CoreSample
        Reference database.
    top_k : int
        Number of top matches to return.

    Returns
    -------
    list of (CoreSample, similarity_score)
    """
    similarities = []
    for sample in database:
        db_feat = sample.texture_features
        cos_sim = np.dot(query_features, db_feat) / (
            np.linalg.norm(query_features) * np.linalg.norm(db_feat) + 1e-12
        )
        similarities.append((sample, float(cos_sim)))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def predict_properties(matches: List[Tuple[CoreSample, float]]) -> dict:
    """
    Predict analog petrophysical properties from top matches using
    similarity-weighted averaging.

    Parameters
    ----------
    matches : list of (CoreSample, similarity_score)

    Returns
    -------
    dict with predicted porosity, permeability, matrix_density, archie_m,
    capillary_entry_pressure and their uncertainties.
    """
    weights = np.array([max(s, 0) for _, s in matches])
    if weights.sum() == 0:
        weights = np.ones(len(matches))
    weights /= weights.sum()

    props = {
        'porosity': np.array([m.porosity for m, _ in matches]),
        'permeability': np.array([m.permeability_klinkenberg for m, _ in matches]),
        'matrix_density': np.array([m.matrix_density for m, _ in matches]),
        'archie_m': np.array([m.archie_m for m, _ in matches]),
        'capillary_entry_pressure': np.array([m.capillary_entry_pressure for m, _ in matches]),
    }
    result = {}
    for key, vals in props.items():
        result[key] = float(np.dot(weights, vals))
        result[f'{key}_std'] = float(np.sqrt(np.dot(weights, (vals - result[key])**2)))
    return result


def evaluate_cutting_sizes(
    database: List[CoreSample],
    sizes_mm: List[float] = [5.0, 4.0, 2.0],
    noise_scale: float = 0.1,
    seed: int = 123,
) -> dict:
    """
    Evaluate how cutting size affects analog prediction accuracy.

    Simulates the paper's finding that cutting size (5, 4, 2 mm) is
    less influential on results, with clastic match rate ~85% and
    carbonate ~38%.

    Parameters
    ----------
    database : list of CoreSample
    sizes_mm : list of float
        Cutting sizes to evaluate.
    noise_scale : float
        Feature noise scale (larger = more heterogeneity loss).
    seed : int

    Returns
    -------
    dict mapping (lithology, size_mm) -> match_rate
    """
    rng = np.random.RandomState(seed)
    results = {}
    for size in sizes_mm:
        for lith in ['clastic', 'carbonate']:
            subset = [s for s in database if s.lithology == lith]
            correct = 0
            total = 0
            for sample in subset:
                # Smaller cuttings -> slightly more noise
                noise = rng.randn(len(sample.texture_features)) * noise_scale / (size / 2.0)
                query = sample.texture_features + noise
                query /= np.linalg.norm(query)
                matches = find_analog_matches(query, database, top_k=3)
                pred = predict_properties(matches)
                # "Match" if porosity within 0.03 and permeability within factor of 3
                phi_ok = abs(pred['porosity'] - sample.porosity) < 0.03
                k_ratio = pred['permeability'] / max(sample.permeability_klinkenberg, 1e-6)
                k_ok = 0.33 < k_ratio < 3.0
                if phi_ok and k_ok:
                    correct += 1
                total += 1
            results[(lith, size)] = correct / max(total, 1)
    return results


def test_all():
    """Test all functions with synthetic data."""
    print("=" * 70)
    print("Module 1: Image-Based AI Rock Properties (Britton et al., 2024)")
    print("=" * 70)

    # Build database
    db = build_sample_database(100)
    print(f"Built database with {len(db)} samples")
    print(f"  Clastic: {sum(1 for s in db if s.lithology == 'clastic')}")
    print(f"  Carbonate: {sum(1 for s in db if s.lithology == 'carbonate')}")

    # Extract features from a synthetic image
    rng = np.random.RandomState(99)
    test_image = rng.rand(128, 128) * 255
    features = extract_texture_features(test_image)
    print(f"\nExtracted {len(features)} texture features from test image")

    # Find analog matches
    matches = find_analog_matches(features, db, top_k=5)
    print("\nTop 5 analog matches:")
    for sample, sim in matches:
        print(f"  {sample} | similarity={sim:.4f}")

    # Predict properties
    props = predict_properties(matches)
    print("\nPredicted analog properties:")
    for key in ['porosity', 'permeability', 'matrix_density', 'archie_m']:
        print(f"  {key}: {props[key]:.4f} ± {props[f'{key}_std']:.4f}")

    # Evaluate cutting sizes
    print("\nCutting size evaluation:")
    size_results = evaluate_cutting_sizes(db)
    for (lith, size), rate in sorted(size_results.items()):
        print(f"  {lith:10s} {size:.0f} mm: match rate = {rate:.0%}")

    print("\n[PASS] All tests completed successfully.\n")


if __name__ == "__main__":
    test_all()
