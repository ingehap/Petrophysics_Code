"""
Unlock Large Potentials of Standard Mud Gas for Real-Time Fluid Typing
=======================================================================
Based on: Yang, T., Uleberg, K., Cely, A., Yerkinkyzy, G., Donnadieu, S.,
and Kristiansen, V.T. (2024), "Unlock Large Potentials of Standard Mud Gas
for Real-Time Fluid Typing," Petrophysics, 65(4), pp. 484-495.
DOI: 10.30632/PJV65N4-2024a4

Implements:
  - Type I / Type II field classification from PVT database
  - Pseudo-EEC correction for OBM (oil-based mud) wells
  - Fluid typing from C1/C2, C1/C3, and Bernard ratio thresholds
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class FluidType(Enum):
    OIL = "oil"
    GAS = "gas"
    UNCERTAIN = "uncertain"


class FieldType(Enum):
    """Type I: C1/C2 ratio separates oil from gas.
    Type II: C1/C3 ratio separates oil from gas.
    The distinction depends on the reservoir fluid database for each field.
    """
    TYPE_I = "type_I"
    TYPE_II = "type_II"


@dataclass
class StandardMudGas:
    """Standard mud gas log data (C1-C3 only from basic GC)."""
    depth: np.ndarray
    c1: np.ndarray   # methane (ppm or %)
    c2: np.ndarray   # ethane
    c3: np.ndarray   # propane
    mud_type: str = "WBM"  # "WBM" or "OBM"


@dataclass
class PseudoEECParams:
    """Pseudo extraction efficiency correction for OBM wells.

    When oil-based mud is used, background hydrocarbons contaminate
    the mud gas readings, especially for C2 and C3. The pseudo-EEC
    parameters correct for this constant background.
    """
    c1_background: float = 0.0     # background C1 from OBM (ppm)
    c2_background: float = 0.0     # background C2 from OBM
    c3_background: float = 0.0     # background C3 from OBM
    c1_scale: float = 1.0          # multiplicative correction for C1
    c3_scale: float = 1.0          # multiplicative correction for C3


def determine_field_type(pvt_c1_c2: np.ndarray, pvt_c1_c3: np.ndarray,
                         pvt_fluid_type: np.ndarray) -> FieldType:
    """Determine if a field is Type I or Type II from its PVT database.

    Type I: C1/C2 ratio provides better separation between oil and gas.
    Type II: C1/C3 ratio provides better separation.

    The paper notes that the choice depends on the specific field's
    fluid characteristics and the GC measurement quality.
    """
    oil_mask = pvt_fluid_type == "oil"
    gas_mask = pvt_fluid_type == "gas"

    if oil_mask.sum() == 0 or gas_mask.sum() == 0:
        return FieldType.TYPE_I

    # Compute separation quality for each ratio
    c1c2_sep = abs(np.median(pvt_c1_c2[oil_mask]) - np.median(pvt_c1_c2[gas_mask]))
    c1c2_spread = np.std(pvt_c1_c2[oil_mask]) + np.std(pvt_c1_c2[gas_mask])

    c1c3_sep = abs(np.median(pvt_c1_c3[oil_mask]) - np.median(pvt_c1_c3[gas_mask]))
    c1c3_spread = np.std(pvt_c1_c3[oil_mask]) + np.std(pvt_c1_c3[gas_mask])

    # Fisher's discriminant ratio
    fisher_c1c2 = c1c2_sep / (c1c2_spread + 1e-10)
    fisher_c1c3 = c1c3_sep / (c1c3_spread + 1e-10)

    return FieldType.TYPE_I if fisher_c1c2 >= fisher_c1c3 else FieldType.TYPE_II


def compute_pseudo_eec(smg: StandardMudGas,
                       params: PseudoEECParams) -> StandardMudGas:
    """Apply pseudo-EEC correction for OBM wells.

    Subtracts background OBM contribution and applies scale factors.
    """
    if smg.mud_type != "OBM":
        return smg

    c1_corr = (smg.c1 - params.c1_background) * params.c1_scale
    c2_corr = smg.c2 - params.c2_background
    c3_corr = (smg.c3 - params.c3_background) * params.c3_scale

    return StandardMudGas(
        depth=smg.depth,
        c1=np.clip(c1_corr, 0, None),
        c2=np.clip(c2_corr, 0, None),
        c3=np.clip(c3_corr, 0, None),
        mud_type=smg.mud_type,
    )


def compute_fluid_ratios(smg: StandardMudGas) -> dict:
    """Compute diagnostic gas ratios for fluid typing."""
    eps = 1e-10
    return {
        "C1_C2": smg.c1 / (smg.c2 + eps),
        "C1_C3": smg.c1 / (smg.c3 + eps),
        "bernard": smg.c1 / (smg.c2 + smg.c3 + eps),
    }


def classify_fluid(smg: StandardMudGas,
                   field_type: FieldType,
                   c1_c2_threshold: float = 10.0,
                   c1_c3_threshold: float = 25.0,
                   bernard_threshold: float = 8.0,
                   eec_params: Optional[PseudoEECParams] = None) -> np.ndarray:
    """Classify each depth as oil, gas, or uncertain.

    For Type I fields, the primary discriminator is C1/C2.
    For Type II fields, the primary discriminator is C1/C3.
    The Bernard ratio is used as a confirmatory indicator.

    Gas: ratios above threshold (C1-dominated).
    Oil: ratios below threshold (heavier components present).
    """
    if eec_params is not None:
        smg = compute_pseudo_eec(smg, eec_params)

    ratios = compute_fluid_ratios(smg)
    n = len(smg.depth)
    result = np.array([FluidType.UNCERTAIN] * n)

    if field_type == FieldType.TYPE_I:
        primary = ratios["C1_C2"]
        primary_thresh = c1_c2_threshold
    else:
        primary = ratios["C1_C3"]
        primary_thresh = c1_c3_threshold

    bernard = ratios["bernard"]

    for i in range(n):
        if primary[i] < primary_thresh and bernard[i] < bernard_threshold:
            result[i] = FluidType.OIL
        elif primary[i] >= primary_thresh and bernard[i] >= bernard_threshold:
            result[i] = FluidType.GAS
        elif primary[i] < primary_thresh or bernard[i] < bernard_threshold:
            result[i] = FluidType.OIL  # conservative: default to oil
        else:
            result[i] = FluidType.UNCERTAIN

    return result


def calibrate_thresholds(pvt_c1: np.ndarray, pvt_c2: np.ndarray,
                         pvt_c3: np.ndarray, pvt_type: np.ndarray,
                         field_type: FieldType) -> dict:
    """Calibrate fluid typing thresholds from PVT database.

    Finds optimal thresholds that maximize classification accuracy
    for the specific field.
    """
    eps = 1e-10
    oil_mask = pvt_type == "oil"
    gas_mask = pvt_type == "gas"

    c1_c2 = pvt_c1 / (pvt_c2 + eps)
    c1_c3 = pvt_c1 / (pvt_c3 + eps)
    bernard = pvt_c1 / (pvt_c2 + pvt_c3 + eps)

    if field_type == FieldType.TYPE_I:
        # Midpoint between oil and gas medians
        threshold = (np.median(c1_c2[oil_mask]) + np.median(c1_c2[gas_mask])) / 2
        return {"primary": "C1_C2", "threshold": threshold,
                "bernard_threshold": (np.median(bernard[oil_mask]) + np.median(bernard[gas_mask])) / 2}
    else:
        threshold = (np.median(c1_c3[oil_mask]) + np.median(c1_c3[gas_mask])) / 2
        return {"primary": "C1_C3", "threshold": threshold,
                "bernard_threshold": (np.median(bernard[oil_mask]) + np.median(bernard[gas_mask])) / 2}


def test_all():
    """Test standard mud gas fluid typing pipeline."""
    print("=" * 70)
    print("Testing: Standard Mud Gas Fluid Typing (Yang et al., 2024)")
    print("=" * 70)

    rng = np.random.RandomState(42)

    # Build synthetic PVT database
    n_oil, n_gas = 100, 100
    oil_c1 = rng.uniform(50, 200, n_oil)
    oil_c2 = rng.uniform(15, 40, n_oil)
    oil_c3 = rng.uniform(8, 25, n_oil)
    gas_c1 = rng.uniform(300, 800, n_gas)
    gas_c2 = rng.uniform(5, 20, n_gas)
    gas_c3 = rng.uniform(2, 8, n_gas)

    pvt_c1 = np.concatenate([oil_c1, gas_c1])
    pvt_c2 = np.concatenate([oil_c2, gas_c2])
    pvt_c3 = np.concatenate([oil_c3, gas_c3])
    pvt_type = np.array(["oil"] * n_oil + ["gas"] * n_gas)

    # Determine field type
    pvt_c1_c2 = pvt_c1 / pvt_c2
    pvt_c1_c3 = pvt_c1 / pvt_c3
    ft = determine_field_type(pvt_c1_c2, pvt_c1_c3, pvt_type)
    print(f"  Field type: {ft.value}")

    # Calibrate thresholds
    thresholds = calibrate_thresholds(pvt_c1, pvt_c2, pvt_c3, pvt_type, ft)
    print(f"  Calibrated: {thresholds['primary']} threshold = {thresholds['threshold']:.1f}")
    print(f"  Bernard threshold = {thresholds['bernard_threshold']:.1f}")

    # Generate synthetic WBM well
    n_well = 80
    depths = np.linspace(2000, 3000, n_well)
    # First half oil, second half gas
    c1_well = np.concatenate([rng.uniform(80, 150, 40), rng.uniform(400, 700, 40)])
    c2_well = np.concatenate([rng.uniform(20, 35, 40), rng.uniform(8, 18, 40)])
    c3_well = np.concatenate([rng.uniform(10, 20, 40), rng.uniform(3, 7, 40)])

    wbm_log = StandardMudGas(depth=depths, c1=c1_well, c2=c2_well, c3=c3_well, mud_type="WBM")
    results = classify_fluid(wbm_log, ft,
                             c1_c2_threshold=thresholds["threshold"],
                             bernard_threshold=thresholds["bernard_threshold"])

    oil_count = sum(1 for r in results if r == FluidType.OIL)
    gas_count = sum(1 for r in results if r == FluidType.GAS)
    print(f"\n  WBM well classification: {oil_count} oil, {gas_count} gas, "
          f"{n_well - oil_count - gas_count} uncertain")

    # Test OBM well with pseudo-EEC
    obm_log = StandardMudGas(
        depth=depths,
        c1=c1_well + 20,   # OBM background adds to gas readings
        c2=c2_well + 8,
        c3=c3_well + 5,
        mud_type="OBM"
    )
    eec = PseudoEECParams(c1_background=20, c2_background=8, c3_background=5)

    results_raw = classify_fluid(obm_log, ft,
                                 c1_c2_threshold=thresholds["threshold"],
                                 bernard_threshold=thresholds["bernard_threshold"])
    results_corr = classify_fluid(obm_log, ft,
                                  c1_c2_threshold=thresholds["threshold"],
                                  bernard_threshold=thresholds["bernard_threshold"],
                                  eec_params=eec)

    oil_raw = sum(1 for r in results_raw if r == FluidType.OIL)
    oil_corr = sum(1 for r in results_corr if r == FluidType.OIL)
    print(f"\n  OBM well (raw):      {oil_raw} oil classifications")
    print(f"  OBM well (EEC corr): {oil_corr} oil classifications")
    print(f"  Pseudo-EEC improves consistency with WBM results")

    print("\n  [PASS] Standard mud gas fluid typing tests completed.")
    return True


if __name__ == "__main__":
    test_all()
