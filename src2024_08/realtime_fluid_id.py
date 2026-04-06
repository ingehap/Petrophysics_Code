"""
Real-Time Fluid Identification from Integrating AMG and Petrophysical Logs
==========================================================================
Based on: Kopal, M., Yerkinkyzy, G., Nygard, M.T., Cely, A., Ungar, F.,
Donnadieu, S., and Yang, T. (2024), "Real-Time Fluid Identification From
Integrating Advanced Mud Gas and Petrophysical Logs," Petrophysics, 65(4),
pp. 470-483. DOI: 10.30632/PJV65N4-2024a3

Implements:
  - Radar plot comparison of mud gas ratios against a PVT database
  - Integration of AMG-based GOR prediction with LWD petrophysical logs
  - Random Forest for GOR and Adaptive Boosting for fluid density prediction
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


@dataclass
class IntegratedLog:
    """Integrated well log combining AMG and petrophysical measurements."""
    depth: np.ndarray
    # AMG compositions (normalized C1-C5)
    c1_norm: np.ndarray
    c2_norm: np.ndarray
    c3_norm: np.ndarray
    c4_norm: np.ndarray
    c5_norm: np.ndarray
    # Petrophysical logs
    gamma_ray: np.ndarray       # GR (API)
    density: np.ndarray         # bulk density (g/cm3)
    neutron: np.ndarray         # neutron porosity (fractional)
    resistivity: np.ndarray     # deep resistivity (ohm-m)


def compute_gas_ratios(log: IntegratedLog) -> dict:
    """Compute standard mud gas ratios used for radar plot comparison.

    Ratios from the paper:
      - C1/C2, C1/C3, C1/(C4+C5)
      - (C1+C2)/(C3+C4+C5)  (balance ratio)
      - C2/C3 (diagnostic for fluid type)
      - Bernard ratio: C1 / (C2 + C3)
    """
    eps = 1e-10
    return {
        "C1_C2": log.c1_norm / (log.c2_norm + eps),
        "C1_C3": log.c1_norm / (log.c3_norm + eps),
        "C1_C4C5": log.c1_norm / (log.c4_norm + log.c5_norm + eps),
        "balance": (log.c1_norm + log.c2_norm) / (log.c3_norm + log.c4_norm + log.c5_norm + eps),
        "C2_C3": log.c2_norm / (log.c3_norm + eps),
        "bernard": log.c1_norm / (log.c2_norm + log.c3_norm + eps),
    }


def radar_plot_similarity(sample_ratios: dict, db_ratios: np.ndarray,
                          db_gor: np.ndarray, ratio_names: list) -> Tuple[int, float]:
    """Compare a single-depth mud gas ratio to a PVT database using a
    radar (star) diagram similarity measure.

    Returns (nearest_index, similarity_score).
    The paper uses this to visually match mud gas signatures to known
    PVT samples and their GOR.
    """
    sample_vec = np.array([sample_ratios[name] for name in ratio_names])

    # Normalize by database range for fair comparison
    db_min = db_ratios.min(axis=0)
    db_max = db_ratios.max(axis=0)
    db_range = db_max - db_min
    db_range[db_range == 0] = 1

    sample_norm = (sample_vec - db_min) / db_range
    db_norm = (db_ratios - db_min) / db_range

    # Euclidean distance to each database sample
    distances = np.sqrt(np.sum((db_norm - sample_norm) ** 2, axis=1))
    best_idx = np.argmin(distances)
    similarity = 1.0 / (1.0 + distances[best_idx])

    return best_idx, similarity


def density_neutron_gas_flag(density: np.ndarray, neutron: np.ndarray,
                             threshold: float = 0.05) -> np.ndarray:
    """Flag gas zones from density-neutron crossover.

    When density porosity > neutron porosity by a significant margin,
    it indicates gas effect. The paper uses this as a complementary
    indicator alongside AMG data.
    """
    # Approximate porosity from density (sandstone matrix = 2.65)
    phi_density = (2.65 - density) / (2.65 - 1.0)
    separation = phi_density - neutron
    return separation > threshold


class IntegratedFluidPredictor:
    """Predict GOR and fluid density by combining AMG with petrophysical logs.

    Uses Random Forest for GOR (more stable with noisy AMG) and
    AdaBoost for fluid density (better at capturing subtle variations).
    """

    def __init__(self, random_state: int = 42):
        self.gor_model = RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=random_state
        )
        self.density_model = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=5),
            n_estimators=100, random_state=random_state
        )
        self._fitted = False

    def _build_features(self, log: IntegratedLog) -> np.ndarray:
        """Combine AMG compositions with petrophysical features."""
        return np.column_stack([
            log.c1_norm, log.c2_norm, log.c3_norm, log.c4_norm, log.c5_norm,
            log.gamma_ray / 150.0,    # normalize GR
            log.density / 2.65,       # normalize density
            log.neutron,              # already fractional
            np.log10(log.resistivity + 0.1) / 3.0,  # normalize log resistivity
        ])

    def train(self, log: IntegratedLog, gor: np.ndarray, fluid_density: np.ndarray):
        """Train both models on a labeled data set."""
        features = self._build_features(log)
        self.gor_model.fit(features, np.log10(gor + 1))
        self.density_model.fit(features, fluid_density)
        self._fitted = True

    def predict(self, log: IntegratedLog) -> Tuple[np.ndarray, np.ndarray]:
        """Predict GOR and fluid density from integrated log.

        Returns (gor_prediction, density_prediction).
        """
        if not self._fitted:
            raise RuntimeError("Models not trained.")
        features = self._build_features(log)
        gor_pred = 10.0 ** self.gor_model.predict(features) - 1
        dens_pred = self.density_model.predict(features)
        return np.clip(gor_pred, 1, 100000), np.clip(dens_pred, 0.5, 1.1)


def classify_fluid_type(gor: np.ndarray) -> np.ndarray:
    """Classify fluid type from GOR using standard thresholds (Sm3/Sm3).

    Thresholds based on the paper's radar plot classification:
      - Black oil: GOR < 180
      - Volatile oil: 180 <= GOR < 360
      - Near-critical: 360 <= GOR < 640
      - Gas condensate: 640 <= GOR < 5000
      - Wet gas: 5000 <= GOR < 15000
      - Dry gas: GOR >= 15000
    """
    types = np.empty(len(gor), dtype="U20")
    types[gor < 180] = "black_oil"
    types[(gor >= 180) & (gor < 360)] = "volatile_oil"
    types[(gor >= 360) & (gor < 640)] = "near_critical"
    types[(gor >= 640) & (gor < 5000)] = "gas_condensate"
    types[(gor >= 5000) & (gor < 15000)] = "wet_gas"
    types[gor >= 15000] = "dry_gas"
    return types


def generate_synthetic_integrated_log(n_points: int = 150,
                                      random_state: int = 42) -> Tuple[IntegratedLog, np.ndarray, np.ndarray]:
    """Generate synthetic integrated log with corresponding GOR and fluid density."""
    rng = np.random.RandomState(random_state)
    depths = np.linspace(2500, 3500, n_points)

    # Varying fluid (oil to gas transition)
    fluid_gradient = (depths - 2500) / 1000
    log_gor_true = 1.5 + 2.5 * fluid_gradient + rng.normal(0, 0.15, n_points)
    gor_true = 10.0 ** np.clip(log_gor_true, 1.0, 4.5)
    fluid_dens = 0.85 - 0.25 * fluid_gradient + rng.normal(0, 0.02, n_points)
    fluid_dens = np.clip(fluid_dens, 0.55, 0.95)

    # AMG compositions correlated with GOR
    c1 = 0.4 + 0.5 * fluid_gradient + rng.normal(0, 0.03, n_points)
    c2 = 0.2 - 0.08 * fluid_gradient + rng.normal(0, 0.02, n_points)
    c3 = 0.15 - 0.08 * fluid_gradient + rng.normal(0, 0.015, n_points)
    c4 = 0.12 - 0.08 * fluid_gradient + rng.normal(0, 0.01, n_points)
    c5 = 0.08 - 0.05 * fluid_gradient + rng.normal(0, 0.008, n_points)
    comps = np.column_stack([c1, c2, c3, c4, c5])
    comps = np.clip(comps, 0.001, None)
    comps = comps / comps.sum(axis=1, keepdims=True)

    # Petrophysical logs
    gr = 30 + 50 * rng.random(n_points)  # shaly to clean sand
    density = 2.2 + 0.2 * (1 - fluid_gradient) + rng.normal(0, 0.02, n_points)
    neutron = 0.15 + 0.1 * (1 - fluid_gradient) + rng.normal(0, 0.02, n_points)
    resistivity = 5 + 50 * fluid_gradient ** 0.5 + rng.normal(0, 2, n_points)

    log = IntegratedLog(
        depth=depths,
        c1_norm=comps[:, 0], c2_norm=comps[:, 1], c3_norm=comps[:, 2],
        c4_norm=comps[:, 3], c5_norm=comps[:, 4],
        gamma_ray=np.clip(gr, 0, 150), density=np.clip(density, 1.8, 2.8),
        neutron=np.clip(neutron, 0, 0.5),
        resistivity=np.clip(resistivity, 0.5, 200),
    )
    return log, gor_true, fluid_dens


def test_all():
    """Test integrated fluid identification pipeline."""
    print("=" * 70)
    print("Testing: Integrated AMG + Petrophysical Fluid ID (Kopal et al., 2024)")
    print("=" * 70)

    # Generate training data
    train_log, gor_train, dens_train = generate_synthetic_integrated_log(
        n_points=300, random_state=42
    )

    # Train integrated predictor
    predictor = IntegratedFluidPredictor()
    predictor.train(train_log, gor_train, dens_train)
    print(f"  Trained on {len(train_log.depth)} depth points")

    # Generate test well
    test_log, gor_true, dens_true = generate_synthetic_integrated_log(
        n_points=100, random_state=99
    )
    gor_pred, dens_pred = predictor.predict(test_log)

    # Evaluate
    mape_gor = np.mean(np.abs((gor_true - gor_pred) / gor_true)) * 100
    mae_dens = np.mean(np.abs(dens_true - dens_pred))
    print(f"\n  GOR prediction MAPE: {mape_gor:.1f}%")
    print(f"  Fluid density MAE: {mae_dens:.4f} g/cm3")

    # Classify fluid types
    types = classify_fluid_type(gor_pred)
    unique, counts = np.unique(types, return_counts=True)
    print(f"\n  Fluid type distribution:")
    for t, c in zip(unique, counts):
        print(f"    {t}: {c} zones")

    # Gas flag from density-neutron
    gas_flags = density_neutron_gas_flag(test_log.density, test_log.neutron)
    print(f"\n  Density-neutron gas flags: {gas_flags.sum()} / {len(gas_flags)} zones")

    # Test radar plot similarity
    ratios = compute_gas_ratios(test_log)
    ratio_names = ["C1_C2", "C1_C3", "C2_C3", "balance", "bernard"]
    db_ratios = np.column_stack([ratios[name][:50] for name in ratio_names])
    db_gor = gor_true[:50]

    sample = {name: ratios[name][75] for name in ratio_names}
    best_idx, sim = radar_plot_similarity(sample, db_ratios, db_gor, ratio_names)
    print(f"\n  Radar plot match: best PVT sample idx={best_idx}, "
          f"similarity={sim:.3f}, matched GOR={db_gor[best_idx]:.0f}")

    print("\n  [PASS] Integrated fluid ID module tests completed.")
    return True


if __name__ == "__main__":
    test_all()
