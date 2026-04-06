#!/usr/bin/env python3
"""
Reconstructing In-Situ Saturation from Dean-Stark Measurements
================================================================
Based on: Zhang et al. (2024), "A Method for Reconstructing In-Situ
Saturation Based on Dean-Stark Saturation Measured in Laboratory",
Petrophysics, Vol. 65, No. 5, pp. 682-698.

Implements:
- Pore-volume expansion (PVE) correction  (Eqs. 2-3)
- Clay dehydration correction              (Eq. 7)
- Degasification correction                (Eqs. 8-11)
- Full saturation-correction workflow
- Normalization to 100% total saturation   (Eqs. 12-13)

Reference: DOI:10.30632/PJV65N5-2024a2
"""

import numpy as np
from typing import Tuple


# -----------------------------------------------------------------------
# 1. Pore-Volume Expansion (PVE) Correction
# -----------------------------------------------------------------------
def pve_correction(Sws: float, Sos: float,
                   phi_s: float, phi_f: float
                   ) -> Tuple[float, float]:
    """
    Correct measured saturations for pore-volume expansion.

    When a core is brought to surface, overburden is released and pore
    volume increases. Laboratory-measured saturations (at surface porosity)
    underestimate true in-situ values.

    Swf1 = Sws * (phi_s / phi_f)
    Sof1 = Sos * (phi_s / phi_f)

    Parameters
    ----------
    Sws : float
        Dean-Stark water saturation at surface conditions (fraction).
    Sos : float
        Dean-Stark oil saturation at surface conditions (fraction).
    phi_s : float
        Surface (lab) porosity (fraction).
    phi_f : float
        In-situ (formation) porosity (fraction).

    Returns
    -------
    Swf1 : float
        PVE-corrected water saturation.
    Sof1 : float
        PVE-corrected oil saturation.
    """
    ratio = phi_s / phi_f
    return Sws * ratio, Sos * ratio


# -----------------------------------------------------------------------
# 2. Clay-Dehydration Correction
# -----------------------------------------------------------------------
def clay_dehydration_correction(Sw_pve: float, delta: float,
                                Vcl: float, phi_e: float
                                ) -> float:
    """
    Correct water saturation for clay dehydration during Dean-Stark
    distillation.

    Montmorillonite releases interlayer water at high temperatures,
    inflating measured water volume.  Eq. 7:

        Swf2 = Sw_pve - delta * Vcl / phi_e

    Parameters
    ----------
    Sw_pve : float
        Water saturation after PVE correction (fraction).
    delta : float
        Average volume ratio of dehydrated water to total plug volume.
        Typical value: 0.201 for B-field (Zhang et al.).
    Vcl : float
        Clay volume fraction (fraction of bulk volume).
    phi_e : float
        Effective porosity (fraction).

    Returns
    -------
    float
        Clay-dehydration-corrected water saturation (Swf2).
    """
    correction = delta * Vcl / phi_e
    return Sw_pve - correction


# -----------------------------------------------------------------------
# 3. Degasification Correction
# -----------------------------------------------------------------------
def degasification_water(Sw_after: float, kw: float, bw: float) -> float:
    """
    Correct water saturation for degasification (Eq. 8):

        Swf3 = kw * ln(Sw_after) + bw

    Parameters
    ----------
    Sw_after : float
        Water saturation after PVE + clay-dehydration correction (%).
    kw : float
        Degasification water-saturation slope.
    bw : float
        Degasification water-saturation intercept.

    Returns
    -------
    float
        Pre-degasification water saturation (%).
    """
    if Sw_after <= 0:
        return 0.0
    return kw * np.log(Sw_after) + bw


def degasification_oil(So_after: float, ko: float) -> float:
    """
    Correct oil saturation for degasification (Eq. 9):

        Sof3 = ko * So_after

    Parameters
    ----------
    So_after : float
        Oil saturation after PVE correction (%).
    ko : float
        Degasification oil-saturation coefficient.

    Returns
    -------
    float
        Pre-degasification oil saturation (%).
    """
    return ko * So_after


def estimate_kw_bw(measured_Sw: np.ndarray, measured_So: np.ndarray
                   ) -> Tuple[float, float, float]:
    """
    Estimate degasification correction coefficients by minimizing
    the total-fluid-saturation error (Eq. 10).

    Uses the empirical linear relationship between kw and bw (Eq. 11):
        bw = 3.8967 * kw - 57.707

    and a 1-D optimization over kw.

    Parameters
    ----------
    measured_Sw : np.ndarray
        Post-PVE, post-clay-dehydration water saturations (%).
    measured_So : np.ndarray
        Post-PVE oil saturations (%).

    Returns
    -------
    kw : float
    bw : float
    ko : float
    """
    from scipy.optimize import minimize_scalar

    def total_error(kw_val):
        bw_val = 3.8967 * kw_val - 57.707
        Sw_corrected = np.array([
            degasification_water(sw, kw_val, bw_val) for sw in measured_Sw
        ])
        # ko chosen so sum of corrected saturations ≈ 100%
        # Sof3 = ko * So  and  Swf3 + Sof3 ≈ 100
        residual_oil = 100.0 - Sw_corrected
        mask = measured_So > 0
        if mask.sum() == 0:
            return 1e6
        ko_est = np.mean(residual_oil[mask] / measured_So[mask])
        So_corrected = ko_est * measured_So
        total_sat = Sw_corrected + So_corrected
        return np.sum((total_sat - 100.0) ** 2)

    result = minimize_scalar(total_error, bounds=(20, 50), method='bounded')
    kw = result.x
    bw = 3.8967 * kw - 57.707
    Sw_corrected = np.array([
        degasification_water(sw, kw, bw) for sw in measured_Sw
    ])
    residual_oil = 100.0 - Sw_corrected
    mask = measured_So > 0
    ko = float(np.mean(residual_oil[mask] / measured_So[mask]))
    return kw, bw, ko


# -----------------------------------------------------------------------
# 4. Normalization (Eqs. 12-13)
# -----------------------------------------------------------------------
def normalize_saturations(Sw_corr: float, So_corr: float
                          ) -> Tuple[float, float]:
    """
    Normalize corrected saturations to sum to 100%.

    Soc = So_corr / (So_corr + Sw_corr) * 100
    Swc = Sw_corr / (So_corr + Sw_corr) * 100

    Parameters
    ----------
    Sw_corr : float
        Corrected water saturation (%).
    So_corr : float
        Corrected oil saturation (%).

    Returns
    -------
    Swc, Soc : (float, float)
        Normalized water and oil saturations (%).
    """
    total = Sw_corr + So_corr
    if total <= 0:
        return 50.0, 50.0
    Swc = 100.0 * Sw_corr / total
    Soc = 100.0 * So_corr / total
    return Swc, Soc


# -----------------------------------------------------------------------
# 5. Full Workflow
# -----------------------------------------------------------------------
def correct_saturation_workflow(
        Sws: float, Sos: float,
        phi_s: float, phi_f: float,
        delta: float, Vcl: float, phi_e: float,
        kw: float, bw: float, ko: float
) -> Tuple[float, float]:
    """
    Apply the full saturation correction workflow:
    PVE → Clay dehydration → Degasification → Normalization.

    All saturations in percent (0–100).

    Returns
    -------
    Swc, Soc : normalized corrected saturations (%).
    """
    # Step 1: PVE correction
    Swf1, Sof1 = pve_correction(Sws, Sos, phi_s, phi_f)

    # Step 2: Clay dehydration on water saturation
    Swf2 = clay_dehydration_correction(Swf1, delta, Vcl, phi_e)

    # Step 3: Degasification
    Swf3 = degasification_water(Swf2, kw, bw)
    Sof3 = degasification_oil(Sof1, ko)

    # Step 4: Normalize
    Swc, Soc = normalize_saturations(Swf3, Sof3)
    return Swc, Soc


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Dean-Stark Saturation Correction Demo ===\n")

    # Synthetic core plug data
    Sws, Sos = 35.0, 45.0       # measured saturations (%)
    phi_s, phi_f = 0.33, 0.30    # surface vs. in-situ porosity
    delta = 0.201                 # clay dehydration ratio
    Vcl = 0.10                    # clay volume fraction
    phi_e = 0.30                  # effective porosity

    # PVE
    Swf1, Sof1 = pve_correction(Sws, Sos, phi_s, phi_f)
    print(f"After PVE: Sw = {Swf1:.2f}%, So = {Sof1:.2f}%")

    # Clay dehydration
    Swf2 = clay_dehydration_correction(Swf1, delta, Vcl, phi_e)
    print(f"After clay dehydration: Sw = {Swf2:.2f}%")

    # Degasification coefficients (typical values)
    kw, bw, ko = 30.0, 3.8967 * 30.0 - 57.707, 1.15
    Swf3 = degasification_water(Swf2, kw, bw)
    Sof3 = degasification_oil(Sof1, ko)
    print(f"After degasification: Sw = {Swf3:.2f}%, So = {Sof3:.2f}%")

    # Normalize
    Swc, Soc = normalize_saturations(Swf3, Sof3)
    print(f"Normalized: Sw = {Swc:.2f}%, So = {Soc:.2f}%")
    print(f"Total = {Swc + Soc:.2f}%")

    # Full workflow
    Swc2, Soc2 = correct_saturation_workflow(
        Sws, Sos, phi_s, phi_f, delta, Vcl, phi_e, kw, bw, ko
    )
    print(f"\nFull workflow: Sw = {Swc2:.2f}%, So = {Soc2:.2f}%")
