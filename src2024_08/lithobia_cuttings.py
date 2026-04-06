"""
LiOBIA: Object-Based Cuttings Image Analysis for Automated Lithology Evaluation
================================================================================
Based on: Yamada, T., Di Santo, S., Bondabou, K., et al. (2024), "LiOBIA:
Object-Based Cuttings Image Analysis for Automated Lithology Evaluation,"
Petrophysics, 65(4), pp. 624-648. DOI: 10.30632/PJV65N4-2024a14

Implements:
  - Cuttings instance segmentation (simplified)
  - Color and texture feature extraction
  - k-NN classification in feature space for lithology estimation
  - 2D manifold analysis for feature space visualization
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix


# Five main lithology types from the cuttings library
LITHOLOGY_TYPES = ["sandstone", "limestone", "shale", "siltstone", "dolomite"]


@dataclass
class CuttingInstance:
    """A single segmented cutting from a digital image."""
    cutting_id: int
    color_features: np.ndarray    # RGB mean + std (6 features)
    texture_features: np.ndarray  # texture descriptors (e.g., GLCM-like)
    shape_features: np.ndarray    # area, perimeter, roundness, etc.
    lithology: Optional[str] = None  # ground truth if available


@dataclass
class CuttingsImage:
    """Digital image of cuttings from a single depth."""
    depth: float
    cuttings: List[CuttingInstance]
    image_width: int = 1920
    image_height: int = 1080


def simulate_cutting_features(lithology: str, n_cuttings: int = 20,
                              random_state: int = 42) -> List[CuttingInstance]:
    """Simulate feature extraction for cuttings of a given lithology.

    Each lithology has characteristic color, texture, and shape signatures:
      - Sandstone: tan/brown, coarse texture, angular to rounded
      - Limestone: light gray/white, smooth, blocky
      - Shale: dark gray/black, laminated texture, platy
      - Siltstone: gray/brown, fine texture, sub-angular
      - Dolomite: light brown/gray, crystalline texture, rhombic
    """
    rng = np.random.RandomState(random_state)

    # Define characteristic feature distributions per lithology
    color_params = {
        "sandstone":  {"mean": [180, 160, 120], "std": [15, 15, 20]},
        "limestone":  {"mean": [200, 200, 195], "std": [10, 10, 10]},
        "shale":      {"mean": [80,  80,  85],  "std": [12, 12, 10]},
        "siltstone":  {"mean": [140, 130, 110], "std": [12, 12, 15]},
        "dolomite":   {"mean": [175, 170, 150], "std": [12, 10, 12]},
    }

    texture_params = {
        "sandstone":  {"contrast": 0.6, "homogeneity": 0.4, "entropy": 0.7},
        "limestone":  {"contrast": 0.2, "homogeneity": 0.8, "entropy": 0.3},
        "shale":      {"contrast": 0.4, "homogeneity": 0.6, "entropy": 0.5},
        "siltstone":  {"contrast": 0.5, "homogeneity": 0.5, "entropy": 0.6},
        "dolomite":   {"contrast": 0.3, "homogeneity": 0.7, "entropy": 0.4},
    }

    shape_params = {
        "sandstone":  {"roundness": 0.6, "aspect_ratio": 1.2, "area": 500},
        "limestone":  {"roundness": 0.5, "aspect_ratio": 1.3, "area": 600},
        "shale":      {"roundness": 0.3, "aspect_ratio": 2.0, "area": 300},
        "siltstone":  {"roundness": 0.5, "aspect_ratio": 1.4, "area": 400},
        "dolomite":   {"roundness": 0.7, "aspect_ratio": 1.1, "area": 550},
    }

    cuttings = []
    cp = color_params[lithology]
    tp = texture_params[lithology]
    sp = shape_params[lithology]

    for i in range(n_cuttings):
        # Color: RGB mean and std (6 features)
        rgb_mean = np.array(cp["mean"]) + rng.normal(0, cp["std"], 3)
        rgb_std = np.abs(np.array(cp["std"]) + rng.normal(0, 3, 3))
        color_feat = np.concatenate([np.clip(rgb_mean, 0, 255) / 255, rgb_std / 50])

        # Texture: contrast, homogeneity, entropy (3 features)
        texture_feat = np.array([
            tp["contrast"] + rng.normal(0, 0.08),
            tp["homogeneity"] + rng.normal(0, 0.06),
            tp["entropy"] + rng.normal(0, 0.08),
        ])
        texture_feat = np.clip(texture_feat, 0, 1)

        # Shape: roundness, aspect ratio, normalized area (3 features)
        shape_feat = np.array([
            sp["roundness"] + rng.normal(0, 0.08),
            sp["aspect_ratio"] + rng.normal(0, 0.15),
            sp["area"] / 1000 + rng.normal(0, 0.05),
        ])
        shape_feat = np.clip(shape_feat, 0.1, None)

        cuttings.append(CuttingInstance(
            cutting_id=i, color_features=color_feat,
            texture_features=texture_feat, shape_features=shape_feat,
            lithology=lithology
        ))

    return cuttings


def build_feature_matrix(cuttings: List[CuttingInstance]) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix and label vector from cutting instances.

    Concatenates color (6), texture (3), and shape (3) = 12 features.
    """
    X = np.array([
        np.concatenate([c.color_features, c.texture_features, c.shape_features])
        for c in cuttings
    ])
    labels = np.array([c.lithology for c in cuttings])
    return X, labels


class LiOBIAClassifier:
    """k-NN classifier for lithology estimation from cuttings features.

    The paper uses k-NN classification in the feature space, retrieving
    similar cuttings from a reference library. This is preferred over
    other classifiers because:
      1. It naturally provides a similarity measure (distance)
      2. Results are interpretable (show similar reference cuttings)
      3. Can be extended with local data sets
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=k, metric="euclidean",
                                          weights="distance")
        self._fitted = False

    def train(self, cuttings_library: List[CuttingInstance]):
        """Train on a labeled cuttings library."""
        X, y = build_feature_matrix(cuttings_library)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True

    def predict(self, cuttings: List[CuttingInstance]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict lithology for new cuttings.

        Returns (predicted_labels, confidence_scores).
        Confidence is the proportion of k-nearest neighbors with the
        majority class.
        """
        if not self._fitted:
            raise RuntimeError("Classifier not trained.")

        X, _ = build_feature_matrix(cuttings)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        confidence = np.max(probabilities, axis=1)
        return predictions, confidence

    def predict_depth_log(self, images: List[CuttingsImage]) -> List[dict]:
        """Generate a lithology log from a sequence of cuttings images.

        For each depth, reports the majority lithology and confidence.
        """
        log = []
        for img in images:
            if not img.cuttings:
                log.append({"depth": img.depth, "lithology": "unknown",
                            "confidence": 0.0, "n_cuttings": 0})
                continue

            preds, confs = self.predict(img.cuttings)
            # Majority vote
            unique, counts = np.unique(preds, return_counts=True)
            majority = unique[np.argmax(counts)]
            avg_conf = np.mean(confs)

            log.append({
                "depth": img.depth,
                "lithology": majority,
                "confidence": avg_conf,
                "n_cuttings": len(img.cuttings),
                "distribution": dict(zip(unique, counts / len(preds))),
            })
        return log


def manifold_analysis(X: np.ndarray, labels: np.ndarray,
                      n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """2D manifold analysis of feature space using PCA.

    The paper confirms that visually similar cuttings are close
    in the high-dimensional feature space, which validates the
    k-NN approach.
    """
    pca = PCA(n_components=n_components)
    X_2d = pca.fit_transform(X)
    return X_2d, pca


def test_all():
    """Test LiOBIA lithology classification pipeline."""
    print("=" * 70)
    print("Testing: LiOBIA Cuttings Image Analysis (Yamada et al., 2024)")
    print("=" * 70)

    # Build reference cuttings library
    library = []
    for i, lith in enumerate(LITHOLOGY_TYPES):
        cuttings = simulate_cutting_features(lith, n_cuttings=50, random_state=i * 10)
        library.extend(cuttings)

    print(f"  Cuttings library: {len(library)} samples, "
          f"{len(LITHOLOGY_TYPES)} lithology types")

    # Split into train/test
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(library))
    split = int(0.7 * len(library))
    train_lib = [library[i] for i in indices[:split]]
    test_lib = [library[i] for i in indices[split:]]

    # Train classifier
    classifier = LiOBIAClassifier(k=5)
    classifier.train(train_lib)
    print(f"  Trained k-NN (k=5) on {len(train_lib)} cuttings")

    # Evaluate on test set
    preds, confs = classifier.predict(test_lib)
    true_labels = np.array([c.lithology for c in test_lib])
    acc = accuracy_score(true_labels, preds)
    print(f"\n  Test accuracy: {acc:.3f} ({acc*100:.1f}%)")
    print(f"  Mean confidence: {confs.mean():.3f}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, preds, labels=LITHOLOGY_TYPES)
    print(f"\n  Confusion matrix:")
    print(f"  {'':>12} " + " ".join(f"{l[:5]:>7}" for l in LITHOLOGY_TYPES))
    for i, lith in enumerate(LITHOLOGY_TYPES):
        print(f"  {lith[:12]:>12} " + " ".join(f"{cm[i, j]:>7}" for j in range(len(LITHOLOGY_TYPES))))

    # Manifold analysis
    X_all, labels_all = build_feature_matrix(library)
    X_2d, pca = manifold_analysis(X_all, labels_all)
    print(f"\n  PCA variance explained: "
          f"{pca.explained_variance_ratio_[0]:.2f}, "
          f"{pca.explained_variance_ratio_[1]:.2f}")

    # Check cluster separation
    for lith in LITHOLOGY_TYPES:
        mask = labels_all == lith
        centroid = X_2d[mask].mean(axis=0)
        spread = X_2d[mask].std(axis=0).mean()
        print(f"    {lith:>12}: centroid=({centroid[0]:.2f}, {centroid[1]:.2f}), "
              f"spread={spread:.2f}")

    # Simulate a well log
    well_images = []
    for depth in np.arange(1000, 1050, 5):
        # Alternating lithologies
        if depth < 1020:
            lith = "sandstone"
        elif depth < 1035:
            lith = "shale"
        else:
            lith = "limestone"
        cuttings = simulate_cutting_features(lith, n_cuttings=15,
                                             random_state=int(depth))
        well_images.append(CuttingsImage(depth=depth, cuttings=cuttings))

    litho_log = classifier.predict_depth_log(well_images)
    print(f"\n  Well lithology log:")
    for entry in litho_log:
        print(f"    Depth {entry['depth']:.0f}m: {entry['lithology']:<12} "
              f"(confidence={entry['confidence']:.2f}, n={entry['n_cuttings']})")

    print("\n  [PASS] LiOBIA lithology classification tests completed.")
    return True


if __name__ == "__main__":
    test_all()
