"""
Fluid Quantification and Kerogen Assessment Using ¹³C and ¹H MR
=================================================================

Implements the ideas of:

    Zamiri, M.S., Ansaribaranghar, N., Ramírez Aguilera, A., Marica, F.,
    and Balcom, B.J., 2025a,
    "Fluid Quantification and Kerogen Assessment in Shales Using ¹³C and
    ¹H Magnetic Resonance Measurements",
    Petrophysics, 66(6), 1090–1100.
    DOI: 10.30632/PJV66N6-2025a12

Key ideas
---------
* ¹³C T₂ resolves kerogen (short T₂) from oil (long T₂) without water
  interference.
* Signal calibration vs. a known reference (decane) gives moles of
  carbon in kerogen and volume of oil.
* Combining ¹³C (carbon content) and ¹H (hydrogen content) yields the
  H/C atomic ratio for plotting on the Van Krevelen diagram.
* Van Krevelen H/C identifies kerogen class (I–IV) and thermal maturity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. NMR Signal Calibration
# ──────────────────────────────────────────────────────────────────────
def calibrate_signal(
    sample_signal: float,
    reference_signal: float,
    reference_quantity: float,
) -> float:
    """Calibrate NMR signal intensity to a physical quantity (moles or mL).

    quantity = (sample_signal / reference_signal) × reference_quantity

    Parameters
    ----------
    sample_signal : float
        Integrated signal of the component of interest.
    reference_signal : float
        Integrated signal of the reference standard.
    reference_quantity : float
        Known quantity of the reference (moles or mL).

    Returns
    -------
    float
        Calibrated quantity of the sample component.
    """
    if reference_signal <= 0:
        raise ValueError("Reference signal must be positive.")
    return (sample_signal / reference_signal) * reference_quantity


# ──────────────────────────────────────────────────────────────────────
# 2. T₂ Component Assignment (kerogen vs oil)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class T2Component:
    """A resolved ¹³C T₂ component.

    Attributes
    ----------
    label : str     'kerogen', 'oil', or 'other'.
    T2_ms : float   Log-mean T₂ (ms).
    amplitude : float  Integrated signal amplitude (a.u.).
    """
    label: str
    T2_ms: float
    amplitude: float


def assign_components(
    T2_values: np.ndarray,
    amplitudes: np.ndarray,
    kerogen_cutoff_ms: float = 1.0,
) -> Tuple[T2Component, T2Component]:
    """Separate ¹³C T₂ distribution into kerogen and oil components.

    Components with T₂ < cutoff are assigned to kerogen (semi-solid);
    components above are assigned to oil.

    Parameters
    ----------
    T2_values : array_like
        T₂ centres of each component (ms).
    amplitudes : array_like
        Corresponding signal amplitudes.
    kerogen_cutoff_ms : float
        T₂ threshold separating kerogen from mobile oil.

    Returns
    -------
    kerogen, oil : T2Component
    """
    T2 = np.asarray(T2_values, float)
    A = np.asarray(amplitudes, float)

    ker_mask = T2 < kerogen_cutoff_ms
    oil_mask = ~ker_mask

    def _summarise(mask, label):
        if not mask.any():
            return T2Component(label, 0.0, 0.0)
        a = A[mask]
        t = T2[mask]
        lm = float(np.exp(np.sum(a * np.log(t)) / np.sum(a)))
        return T2Component(label, lm, float(a.sum()))

    return _summarise(ker_mask, "kerogen"), _summarise(oil_mask, "oil")


# ──────────────────────────────────────────────────────────────────────
# 3. H/C Ratio from Combined ¹H and ¹³C Measurements
# ──────────────────────────────────────────────────────────────────────
def hydrogen_carbon_ratio(
    H_moles: float,
    C_moles: float,
) -> float:
    """Molar H/C ratio for kerogen.

    Parameters
    ----------
    H_moles : float
        Moles of hydrogen in kerogen (from ¹H measurement).
    C_moles : float
        Moles of carbon in kerogen (from ¹³C measurement,
        corrected for natural abundance).

    Returns
    -------
    float
        H/C molar ratio.
    """
    if C_moles <= 0:
        raise ValueError("Carbon moles must be positive.")
    return H_moles / C_moles


def carbon_moles_from_13C(
    signal_13C: float,
    ref_signal_13C: float,
    ref_moles_C: float,
    natural_abundance: float = 0.011,
) -> float:
    """Total carbon moles from ¹³C signal (accounting for natural abundance).

    n_C_total = (signal / ref_signal) × ref_moles / natural_abundance × natural_abundance
              = calibrate_signal(...)

    Because the reference is also at natural abundance, the abundance
    cancels — the calibrated value is directly total moles of C.
    """
    return calibrate_signal(signal_13C, ref_signal_13C, ref_moles_C)


# ──────────────────────────────────────────────────────────────────────
# 4. Van Krevelen Diagram Classification
# ──────────────────────────────────────────────────────────────────────
def kerogen_class_from_HC(HC: float) -> str:
    """Approximate kerogen class from H/C ratio.

    Kerogen class I:   H/C > 1.5
    Kerogen class II:  1.0 < H/C ≤ 1.5
    Kerogen class III: 0.5 < H/C ≤ 1.0
    Kerogen class IV:  H/C ≤ 0.5  (over-mature / inert)

    (Simplified; the full Van Krevelen diagram also uses O/C.)
    """
    if HC > 1.5:
        return "I"
    elif HC > 1.0:
        return "II"
    elif HC > 0.5:
        return "III"
    else:
        return "IV (over-mature)"


def maturity_from_HC(HC: float) -> str:
    """Qualitative maturity assessment from H/C ratio.

    Low H/C → high maturity (hydrogen lost during thermal cracking).
    """
    if HC > 1.2:
        return "immature / early mature"
    elif HC > 0.7:
        return "mid-mature (oil window)"
    elif HC > 0.4:
        return "late mature (wet gas / condensate)"
    else:
        return "over-mature (dry gas)"


@dataclass
class VanKrevelenPoint:
    """A point on the Van Krevelen diagram."""
    sample_id: str
    HC: float
    OC: float = 0.0  # optional — requires ¹⁷O measurement

    @property
    def kerogen_class(self) -> str:
        return kerogen_class_from_HC(self.HC)

    @property
    def maturity(self) -> str:
        return maturity_from_HC(self.HC)


# ──────────────────────────────────────────────────────────────────────
# 5. T₁/T₂ Ratio Summary
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ShaleSpeciesRelaxation:
    """Relaxation times of a shale species from ¹³C and ¹H MR.

    Directly inspired by Tables 1-2 in the paper.
    """
    species: str          # 'kerogen', 'oil', 'water'
    T1_13C_ms: float
    T2_13C_ms: float
    T1_1H_ms: float
    T2_1H_ms: float

    @property
    def T1_T2_13C(self) -> float:
        return self.T1_13C_ms / self.T2_13C_ms if self.T2_13C_ms > 0 else float("inf")

    @property
    def T1_T2_1H(self) -> float:
        return self.T1_1H_ms / self.T2_1H_ms if self.T2_1H_ms > 0 else float("inf")


# ──────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Mimic measurements for Sample E10 and H2 from the paper
    # Signal calibration
    ker_C_E10 = carbon_moles_from_13C(signal_13C=120, ref_signal_13C=500,
                                       ref_moles_C=0.50)
    oil_vol_E10 = calibrate_signal(sample_signal=350, reference_signal=500,
                                    reference_quantity=2.0)  # mL
    print(f"Sample E10: kerogen C = {ker_C_E10:.3f} mol, oil = {oil_vol_E10:.2f} mL")

    # H/C from combined measurements
    # Suppose ¹H gives 0.13 mol H in kerogen for E10 (immature, H/C ~ 1.1)
    HC_E10 = hydrogen_carbon_ratio(0.13, ker_C_E10)
    vk_E10 = VanKrevelenPoint("E10", HC_E10)
    print(f"  H/C = {HC_E10:.2f}  → class {vk_E10.kerogen_class}, "
          f"{vk_E10.maturity}")

    ker_C_H2 = carbon_moles_from_13C(300, 500, 0.50)
    # H2 is mature, low H/C ~ 0.4
    HC_H2 = hydrogen_carbon_ratio(0.12, ker_C_H2)
    vk_H2 = VanKrevelenPoint("H2", HC_H2)
    print(f"Sample H2: H/C = {HC_H2:.2f}  → class {vk_H2.kerogen_class}, "
          f"{vk_H2.maturity}")

    # Relaxation summary
    ker = ShaleSpeciesRelaxation("kerogen", 300, 0.4, 92, 0.1)
    oil = ShaleSpeciesRelaxation("oil", 1400, 10, 360, 10)
    for sp in [ker, oil]:
        print(f"  {sp.species:8s}  T₁/T₂(¹³C)={sp.T1_T2_13C:8.0f}  "
              f"T₁/T₂(¹H)={sp.T1_T2_1H:6.0f}")
