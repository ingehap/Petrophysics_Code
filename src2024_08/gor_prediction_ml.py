"""
GOR Prediction from Advanced Mud Gas using Machine Learning
============================================================
Based on: Arief, I.H. and Yang, T. (2024), "A Machine-Learning Approach to
Predict Gas-Oil Ratio Based on Advanced Mud Gas Data," Petrophysics, 65(4),
pp. 433-454. DOI: 10.30632/PJV65N4-2024a1

Implements three ML models (Random Forest, MLP, Gaussian Process Regression)
to predict gas-oil ratio (GOR) from C1-C5 normalized compositions derived
from advanced mud gas (AMG) data, trained on a PVT database.

Also implements QC metrics (Wetness, Balance, Character factors) to flag
low-quality AMG data.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AMGSample:
    """Advanced Mud Gas sample with C1-C5 normalized compositions."""
    depth: float
    c1: float  # methane fraction (normalized)
    c2: float  # ethane
    c3: float  # propane
    ic4: float  # iso-butane
    nc4: float  # normal butane
    ic5: float  # iso-pentane
    nc5: float  # normal pentane

    @property
    def c4(self) -> float:
        return self.ic4 + self.nc4

    @property
    def c5(self) -> float:
        return self.ic5 + self.nc5

    @property
    def feature_vector(self) -> np.ndarray:
        """Normalized C1-C5 composition as ML feature vector."""
        total = self.c1 + self.c2 + self.c3 + self.c4 + self.c5
        if total == 0:
            return np.zeros(5)
        return np.array([self.c1, self.c2, self.c3, self.c4, self.c5]) / total


@dataclass
class QCMetrics:
    """Quality-check metrics for AMG data (Arief & Yang, 2024).

    Wetness (Wh): ratio of heavier hydrocarbons to total gas.
    Balance (Bh): ratio of C1-C2 area to C3-C5 area on a polygon.
    Character (Ch): variability measure of the gas composition.
    qc_flag: 1 = good quality, 0 = poor quality.
    """
    wetness: float
    balance: float
    character: float
    qc_flag: int


def compute_wetness(sample: AMGSample) -> float:
    """Wetness ratio Wh = (C2+C3+C4+C5) / (C1+C2+C3+C4+C5).

    Values near 0 => dry gas, near 1 => oil-associated gas.
    """
    total = sample.c1 + sample.c2 + sample.c3 + sample.c4 + sample.c5
    if total == 0:
        return 0.0
    return (sample.c2 + sample.c3 + sample.c4 + sample.c5) / total


def compute_balance(sample: AMGSample) -> float:
    """Balance factor Bh = Sa / Sb where Sa = polygon area of light
    components (C1, C2) and Sb = polygon area of heavy (C3, C4, C5).
    High values indicate lighter fluids.
    """
    sa = sample.c1 + sample.c2
    sb = sample.c3 + sample.c4 + sample.c5
    if sb == 0:
        return float("inf")
    return sa / sb


def compute_character(sample: AMGSample) -> float:
    """Character factor Ch: coefficient of variation of C1-C5 composition.
    Low values indicate uniform composition (suspicious), high values
    indicate well-differentiated components.
    """
    comps = np.array([sample.c1, sample.c2, sample.c3, sample.c4, sample.c5])
    if comps.mean() == 0:
        return 0.0
    return comps.std() / comps.mean()


def compute_qc_metrics(sample: AMGSample,
                       wetness_range: tuple = (0.05, 0.95),
                       balance_range: tuple = (0.5, 50.0),
                       character_min: float = 0.3) -> QCMetrics:
    """Compute full QC metrics for an AMG sample.

    A sample passes QC (flag=1) if wetness, balance, and character are
    within acceptable ranges derived from a PVT database.
    """
    w = compute_wetness(sample)
    b = compute_balance(sample)
    c = compute_character(sample)
    qc = int(
        wetness_range[0] <= w <= wetness_range[1]
        and balance_range[0] <= b <= balance_range[1]
        and c >= character_min
    )
    return QCMetrics(wetness=w, balance=b, character=c, qc_flag=qc)


class GORPredictor:
    """Predict GOR from C1-C5 normalized composition using ML models
    trained on a PVT database (Arief & Yang, 2024).

    Three model types:
      - 'rf'  : Random Forest (best for noisy AMG data)
      - 'mlp' : Multi-Layer Perceptron
      - 'gpr' : Gaussian Process Regression with Matern kernel
    """

    def __init__(self, model_type: str = "rf", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._build_model()
        self._is_fitted = False

    def _build_model(self):
        if self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=200, max_depth=15,
                min_samples_leaf=5, random_state=self.random_state
            )
        elif self.model_type == "mlp":
            return MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu",
                max_iter=1000, random_state=self.random_state
            )
        elif self.model_type == "gpr":
            kernel = Matern(length_scale=1.0, nu=2.5)
            return GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def train(self, compositions: np.ndarray, log_gor: np.ndarray):
        """Train on PVT database. compositions shape (n, 5) = normalized
        C1-C5. log_gor shape (n,) = log10(GOR).

        The paper trains on log(GOR) for better distribution properties.
        """
        self.model.fit(compositions, log_gor)
        self._is_fitted = True

    def predict(self, compositions: np.ndarray) -> np.ndarray:
        """Predict GOR (linear scale) from normalized C1-C5."""
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        log_gor_pred = self.model.predict(compositions)
        return 10.0 ** log_gor_pred

    def cross_validate(self, compositions: np.ndarray, log_gor: np.ndarray,
                       n_folds: int = 5) -> dict:
        """K-fold cross-validation returning MAPE statistics."""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        mapes = []
        for train_idx, val_idx in kf.split(compositions):
            self.model.fit(compositions[train_idx], log_gor[train_idx])
            pred = self.model.predict(compositions[val_idx])
            gor_pred = 10.0 ** pred
            gor_true = 10.0 ** log_gor[val_idx]
            mape = np.mean(np.abs((gor_true - gor_pred) / gor_true)) * 100
            mapes.append(mape)
        self._is_fitted = True
        return {"mean_mape": np.mean(mapes), "std_mape": np.std(mapes),
                "fold_mapes": mapes}


def generate_synthetic_pvt_database(n_samples: int = 500,
                                    random_state: int = 42) -> tuple:
    """Generate synthetic PVT database mimicking NCS fluid compositions.

    Returns (compositions, log_gor) where compositions is (n, 5) array
    of normalized C1-C5 and log_gor is log10(GOR) in Sm3/Sm3.
    """
    rng = np.random.RandomState(random_state)
    # GOR range: ~10 (heavy oil) to ~50000 (dry gas)
    log_gor = rng.uniform(1.0, 4.7, n_samples)
    gor = 10.0 ** log_gor

    # C1 fraction increases with GOR (gas systems have more methane)
    c1_frac = 0.3 + 0.65 * (log_gor - 1.0) / 3.7 + rng.normal(0, 0.03, n_samples)
    c2_frac = 0.05 + 0.15 * np.exp(-0.3 * (log_gor - 2.5) ** 2) + rng.normal(0, 0.02, n_samples)
    c3_frac = 0.03 + 0.12 * np.exp(-0.4 * (log_gor - 2.0) ** 2) + rng.normal(0, 0.015, n_samples)
    c4_frac = 0.02 + 0.08 * np.exp(-0.5 * (log_gor - 1.8) ** 2) + rng.normal(0, 0.01, n_samples)
    c5_frac = 0.01 + 0.05 * np.exp(-0.6 * (log_gor - 1.5) ** 2) + rng.normal(0, 0.008, n_samples)

    # Clip and normalize
    comps = np.column_stack([c1_frac, c2_frac, c3_frac, c4_frac, c5_frac])
    comps = np.clip(comps, 0.001, None)
    comps = comps / comps.sum(axis=1, keepdims=True)

    return comps, log_gor


def test_all():
    """Test GOR prediction pipeline with synthetic data."""
    print("=" * 70)
    print("Testing: GOR Prediction from AMG (Arief & Yang, 2024)")
    print("=" * 70)

    # Generate synthetic PVT database
    comps, log_gor = generate_synthetic_pvt_database(n_samples=400)
    print(f"  PVT database: {comps.shape[0]} samples, C1-C5 compositions")
    print(f"  GOR range: {10**log_gor.min():.0f} - {10**log_gor.max():.0f} Sm3/Sm3")

    # Test all three model types
    for model_type in ["rf", "mlp", "gpr"]:
        predictor = GORPredictor(model_type=model_type)
        cv_results = predictor.cross_validate(comps, log_gor, n_folds=5)
        print(f"\n  {model_type.upper()} Cross-Validation MAPE: "
              f"{cv_results['mean_mape']:.1f}% +/- {cv_results['std_mape']:.1f}%")

    # Train final RF model and predict on synthetic AMG data
    predictor = GORPredictor(model_type="rf")
    predictor.train(comps, log_gor)

    # Create synthetic AMG samples
    amg_samples = [
        AMGSample(depth=2000, c1=0.75, c2=0.12, c3=0.07, ic4=0.02, nc4=0.02, ic5=0.01, nc5=0.01),
        AMGSample(depth=2050, c1=0.55, c2=0.18, c3=0.13, ic4=0.05, nc4=0.04, ic5=0.03, nc5=0.02),
        AMGSample(depth=2100, c1=0.95, c2=0.03, c3=0.01, ic4=0.005, nc4=0.003, ic5=0.001, nc5=0.001),
    ]

    print("\n  AMG Predictions:")
    for s in amg_samples:
        qc = compute_qc_metrics(s)
        gor_pred = predictor.predict(s.feature_vector.reshape(1, -1))[0]
        print(f"    Depth {s.depth}m: GOR={gor_pred:.0f} Sm3/Sm3, "
              f"QC={'PASS' if qc.qc_flag else 'FAIL'} "
              f"(Wh={qc.wetness:.2f}, Bh={qc.balance:.1f}, Ch={qc.character:.2f})")

    print("\n  [PASS] GOR Prediction module tests completed.")
    return True


if __name__ == "__main__":
    test_all()
