#!/usr/bin/env python3
"""
Bulk Saturation Measurement Using 13C and 1H Magnetic Resonance.

Reference: Ansaribaranghar et al., 2025, Petrophysics 66(1), 155-168. DOI:10.30632/PJV66N1-2025a11

Implements:
  - CPMG signal decay model (Eq. 1)
  - Oil volume from 13C MR measurement (Eq. 3)
  - Water volume from 1H + 13C combination (Eq. 2)
  - Water volume from 23Na measurement (Eq. 4)
  - Saturation computation workflow (Fig. 1)
  - Comparison with Dean-Stark method
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class MRCalibration:
    """Reference sample calibration data."""
    H1_signal_per_vol_oil: float = 100.0   # signal per mL oil
    H1_signal_per_vol_brine: float = 110.0  # signal per mL brine
    C13_signal_per_vol_oil: float = 1.1     # signal per mL oil (1.1% natural abundance)
    Na23_signal_per_vol_brine: float = 5.0  # signal per mL brine

def cpmg_decay(t, S0_components, T2_components):
    """Multi-exponential CPMG decay (Eq. 1): S(t) = Σ Si·exp(-t/T2i)."""
    signal = np.zeros_like(t, dtype=float)
    for S0, T2 in zip(S0_components, T2_components):
        signal += S0 * np.exp(-t / T2)
    return signal

def oil_volume_from_C13(C13_signal_sample, cal: MRCalibration):
    """Oil volume from 13C MR measurement (Eq. 3).

    V_oil = C13_signal_sample / (C13_signal_ref / V_ref)_oil
    """
    return C13_signal_sample / cal.C13_signal_per_vol_oil

def water_volume_from_H1_C13(H1_signal_sample, V_oil, cal: MRCalibration):
    """Water volume from 1H signal and known oil volume (Eq. 2).

    1H_signal = V_oil * (1H/vol)_oil + V_w * (1H/vol)_brine
    => V_w = (1H_signal - V_oil*(1H/vol)_oil) / (1H/vol)_brine
    """
    return (H1_signal_sample - V_oil * cal.H1_signal_per_vol_oil) / cal.H1_signal_per_vol_brine

def water_volume_from_Na23(Na23_signal_sample, cal: MRCalibration):
    """Water volume from 23Na MR measurement (Eq. 4)."""
    return Na23_signal_sample / cal.Na23_signal_per_vol_brine

def compute_saturation(V_oil, V_water, pore_volume=None):
    """Compute saturations from fluid volumes.

    If pore_volume unknown: PV = V_oil + V_water (assumes 100% liquid-filled).
    """
    if pore_volume is None:
        pore_volume = V_oil + V_water
    if pore_volume <= 0:
        return dict(Sw=0, So=0, PV=0)
    Sw = V_water / pore_volume
    So = V_oil / pore_volume
    return dict(Sw=Sw, So=So, PV=pore_volume)

def saturation_workflow(H1_signal, C13_signal, cal: MRCalibration,
                        gravimetric_PV=None, Na23_signal=None):
    """Full saturation determination workflow (Fig. 1).

    Step 1: 13C → V_oil
    Step 2: If gravimetric PV known → V_w = PV - V_oil
            Else: 1H → V_w using Eq. 2
    Step 3: Compute Sw, So
    """
    V_oil = oil_volume_from_C13(C13_signal, cal)

    if gravimetric_PV is not None:
        V_water = gravimetric_PV - V_oil
        method = "gravimetric"
    elif Na23_signal is not None:
        V_water = water_volume_from_Na23(Na23_signal, cal)
        method = "23Na"
    else:
        V_water = water_volume_from_H1_C13(H1_signal, V_oil, cal)
        method = "1H+13C"

    sat = compute_saturation(V_oil, V_water, gravimetric_PV)
    sat['method'] = method
    sat['V_oil'] = V_oil
    sat['V_water'] = V_water
    return sat

def compare_with_dean_stark(Sw_mr, Sw_ds):
    """Compare MR saturation with Dean-Stark reference."""
    diff = abs(Sw_mr - Sw_ds)
    return dict(Sw_MR=Sw_mr, Sw_DS=Sw_ds, difference=diff, agreement=diff<0.02)

if __name__ == "__main__":
    cal = MRCalibration()
    # Simulate measurement on a partially saturated core plug
    V_oil_true = 3.5; V_water_true = 4.2; PV = V_oil_true + V_water_true
    H1_sig = V_oil_true*cal.H1_signal_per_vol_oil + V_water_true*cal.H1_signal_per_vol_brine
    C13_sig = V_oil_true*cal.C13_signal_per_vol_oil
    sat = saturation_workflow(H1_sig, C13_sig, cal)
    ds = compare_with_dean_stark(sat['Sw'], V_water_true/PV)
    # CPMG decay
    t = np.linspace(0, 500, 100)  # ms
    sig = cpmg_decay(t, [60, 40], [100, 10])  # two T2 components
    print(f"MR Bulk Saturation — Sw_MR={sat['Sw']:.4f}, Sw_DS={ds['Sw_DS']:.4f}, diff={ds['difference']:.4f}")
    print(f"Method: {sat['method']}, V_oil={sat['V_oil']:.2f}, V_water={sat['V_water']:.2f}")
    print(f"Agreement with Dean-Stark: {ds['agreement']}")
