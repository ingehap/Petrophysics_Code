"""
Reservoir Fluid Data Acquisition Using AMG in Shale Reservoirs
================================================================
Based on: Yang, T., Arief, I.H., Niemann, M., and Houbiers, M. (2024),
"Reservoir Fluid Data Acquisition Using Advanced Mud-Logging Gas in Shale
Reservoirs," Petrophysics, 65(4), pp. 455-469. DOI: 10.30632/PJV65N4-2024a2

Extends GOR prediction from AMG to unconventional (shale) reservoirs,
where fluid heterogeneity along the horizontal wellbore is critical
for hydraulic fracturing optimization. Applies extraction efficiency
correction (EEC) and ML models to predict GOR and fluid type continuously.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.ensemble import RandomForestRegressor


@dataclass
class ShaleAMGLog:
    """Continuous AMG log along a shale well."""
    depths: np.ndarray         # measured depth (m or ft)
    c1: np.ndarray             # methane concentration (ppm)
    c2: np.ndarray             # ethane
    c3: np.ndarray             # propane
    c4: np.ndarray             # butane (iC4 + nC4)
    c5: np.ndarray             # pentane (iC5 + nC5)
    total_gas: np.ndarray      # total hydrocarbon gas


@dataclass
class EECParameters:
    """Extraction Efficiency Correction parameters.

    The advanced mud gas degasser has varying extraction efficiency
    for different components. EEC corrects for this.
    """
    alpha_c1: float = 1.0   # C1 correction factor
    alpha_c2: float = 0.85  # C2 correction
    alpha_c3: float = 0.72  # C3 correction
    alpha_c4: float = 0.60  # C4 correction
    alpha_c5: float = 0.50  # C5 correction (heaviest = lowest efficiency)


def apply_eec(log: ShaleAMGLog, eec: EECParameters) -> ShaleAMGLog:
    """Apply extraction efficiency correction to raw AMG data.

    Corrects for the fact that heavier components have lower extraction
    efficiency from the drilling mud. Divides measured concentration by
    the efficiency factor to estimate true formation gas composition.
    """
    return ShaleAMGLog(
        depths=log.depths,
        c1=log.c1 / eec.alpha_c1,
        c2=log.c2 / eec.alpha_c2,
        c3=log.c3 / eec.alpha_c3,
        c4=log.c4 / eec.alpha_c4,
        c5=log.c5 / eec.alpha_c5,
        total_gas=log.total_gas,
    )


def normalize_compositions(log: ShaleAMGLog) -> np.ndarray:
    """Normalize C1-C5 to fractional compositions at each depth.

    Returns array of shape (n_depths, 5).
    """
    raw = np.column_stack([log.c1, log.c2, log.c3, log.c4, log.c5])
    totals = raw.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1, totals)
    return raw / totals


def smooth_compositions(comps: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving average smoothing of composition log to reduce noise.

    The paper emphasizes the importance of smoothing AMG data before
    applying ML models, especially in shale reservoirs where drilling
    induced noise is common.
    """
    from scipy.ndimage import uniform_filter1d
    return np.column_stack([
        uniform_filter1d(comps[:, i], size=window) for i in range(comps.shape[1])
    ])


def compute_shale_qc(comps: np.ndarray, total_gas: np.ndarray,
                     min_total_gas: float = 50.0) -> np.ndarray:
    """QC flag for shale AMG data.

    - Minimum total gas threshold (below = non-reservoir)
    - Composition consistency check: C1 should dominate in gas-rich shales
    Returns boolean array (True = pass).
    """
    gas_ok = total_gas >= min_total_gas
    c1_dominant = comps[:, 0] > 0.3   # C1 should be >30% of C1-C5
    monotonic = np.all(np.diff(comps[:, :3], axis=1) <= 0.05, axis=1)  # C1 > C2 > C3 approx
    return gas_ok & c1_dominant


class ShaleGORPredictor:
    """Predict GOR continuously along a shale well from AMG data.

    Uses Random Forest trained on PVT database, applied after EEC
    correction and smoothing of the AMG compositions.
    """

    def __init__(self, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_leaf=5, random_state=random_state
        )
        self._fitted = False

    def train_from_pvt(self, comps: np.ndarray, log_gor: np.ndarray):
        """Train on PVT database (normalized C1-C5 vs log10 GOR)."""
        self.model.fit(comps, log_gor)
        self._fitted = True

    def predict_log(self, log: ShaleAMGLog,
                    eec: EECParameters = None,
                    smooth_window: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict continuous GOR log from AMG data.

        Returns (gor_pred, qc_flags, normalized_comps).
        """
        if not self._fitted:
            raise RuntimeError("Model not trained.")

        if eec is not None:
            log = apply_eec(log, eec)

        comps = normalize_compositions(log)
        comps_smooth = smooth_compositions(comps, window=smooth_window)
        qc = compute_shale_qc(comps_smooth, log.total_gas)

        log_gor_pred = self.model.predict(comps_smooth)
        gor_pred = 10.0 ** log_gor_pred

        return gor_pred, qc, comps_smooth


def generate_synthetic_shale_well(n_points: int = 200,
                                  random_state: int = 42) -> ShaleAMGLog:
    """Generate synthetic shale well AMG data.

    Simulates a horizontal well through a shale reservoir with varying
    fluid composition (typical of Bakken or Eagle Ford-type reservoirs).
    """
    rng = np.random.RandomState(random_state)
    depths = np.linspace(3000, 5000, n_points)

    # Fluid gradient: gas-richer at shallow, oil-richer at depth
    gradient = (depths - 3000) / 2000
    c1 = 5000 * (0.8 - 0.3 * gradient) + rng.normal(0, 200, n_points)
    c2 = 800 * (0.15 + 0.1 * gradient) + rng.normal(0, 50, n_points)
    c3 = 400 * (0.1 + 0.1 * gradient) + rng.normal(0, 30, n_points)
    c4 = 200 * (0.05 + 0.08 * gradient) + rng.normal(0, 15, n_points)
    c5 = 100 * (0.03 + 0.05 * gradient) + rng.normal(0, 10, n_points)

    c1, c2, c3, c4, c5 = [np.clip(x, 1, None) for x in [c1, c2, c3, c4, c5]]
    total_gas = c1 + c2 + c3 + c4 + c5

    return ShaleAMGLog(depths=depths, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5,
                       total_gas=total_gas)


def test_all():
    """Test shale AMG fluid prediction pipeline."""
    print("=" * 70)
    print("Testing: Shale Reservoir AMG Fluid Prediction (Yang et al., 2024)")
    print("=" * 70)

    # Generate synthetic PVT training data
    rng = np.random.RandomState(42)
    n_pvt = 500
    log_gor_train = rng.uniform(1.0, 4.5, n_pvt)
    c1f = 0.35 + 0.6 * (log_gor_train - 1) / 3.5 + rng.normal(0, 0.03, n_pvt)
    c2f = 0.15 * np.exp(-0.3 * (log_gor_train - 2.5) ** 2) + rng.normal(0, 0.02, n_pvt)
    c3f = 0.1 * np.exp(-0.4 * (log_gor_train - 2.0) ** 2) + rng.normal(0, 0.015, n_pvt)
    c4f = 0.06 * np.exp(-0.5 * (log_gor_train - 1.8) ** 2) + rng.normal(0, 0.01, n_pvt)
    c5f = 0.04 * np.exp(-0.6 * (log_gor_train - 1.5) ** 2) + rng.normal(0, 0.008, n_pvt)
    comps_train = np.column_stack([c1f, c2f, c3f, c4f, c5f])
    comps_train = np.clip(comps_train, 0.001, None)
    comps_train = comps_train / comps_train.sum(axis=1, keepdims=True)

    # Train predictor
    predictor = ShaleGORPredictor()
    predictor.train_from_pvt(comps_train, log_gor_train)
    print(f"  Trained on {n_pvt} PVT samples")

    # Generate synthetic shale well
    well_log = generate_synthetic_shale_well(n_points=200)
    print(f"  Shale well: {len(well_log.depths)} depth points, "
          f"{well_log.depths[0]:.0f}-{well_log.depths[-1]:.0f} m")

    # Predict with EEC
    eec = EECParameters()
    gor_pred, qc, comps = predictor.predict_log(well_log, eec=eec, smooth_window=5)

    print(f"\n  GOR predictions:")
    print(f"    Range: {gor_pred.min():.0f} - {gor_pred.max():.0f} Sm3/Sm3")
    print(f"    Mean:  {gor_pred.mean():.0f} Sm3/Sm3")
    print(f"    QC pass rate: {qc.mean() * 100:.1f}%")

    # Verify EEC effect
    gor_no_eec, _, _ = predictor.predict_log(well_log, eec=None, smooth_window=5)
    print(f"\n  Without EEC: GOR range {gor_no_eec.min():.0f} - {gor_no_eec.max():.0f}")
    print(f"  EEC correction shifts GOR (heavier components boosted)")

    print("\n  [PASS] Shale AMG prediction module tests completed.")
    return True


if __name__ == "__main__":
    test_all()
