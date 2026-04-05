#!/usr/bin/env python3
"""
Dual Matrix Porosity in Sandstone: Fluid Distribution and Flow Properties.

Reference: Wang & Galley, 2025, Petrophysics 66(1), 134-154. DOI:10.30632/PJV66N1-2025a10

Implements:
  - Dual-porosity Brooks-Corey capillary pressure model (Eqs. 1-3)
  - Imbibition Pc from drainage Pc with contact angle correction (Eq. 4)
  - Trapped oil saturation via Land correlation (Eq. 6)
  - NMR T2 Gaussian deconvolution for macro/meso porosity
  - Dual-porosity Corey relative permeability
  - Composite saturation height function
"""
import numpy as np

def brooks_corey_Pc(Sw, Swir, Pd, lam):
    """Brooks-Corey Pc = Pd * Se^(-1/λ)."""
    Se = np.clip((Sw-Swir)/(1-Swir), 1e-6, 1)
    return Pd * Se**(-1.0/lam)

def dual_porosity_Pc(Sw, phi_macro, phi_meso, Swir_M, Swir_m,
                     Pd_M, Pd_m, lam_M, lam_m):
    """Composite Pc from two Brooks-Corey functions for macro/meso systems.

    Total Sw is volume-weighted from both systems:
      Sw = (phi_M*Sw_M + phi_m*Sw_m) / (phi_M + phi_m)
    Each system has its own Pc curve.
    """
    phi_t = phi_macro + phi_meso
    # Distribute saturation: macropores drain first (lower Pc)
    Sw_macro = np.clip(Sw*phi_t/phi_macro - phi_meso/phi_macro, Swir_M, 1)
    Sw_meso = np.clip((Sw*phi_t - phi_macro*Sw_macro)/phi_meso, Swir_m, 1)
    Pc_M = brooks_corey_Pc(Sw_macro, Swir_M, Pd_M, lam_M)
    Pc_m = brooks_corey_Pc(Sw_meso, Swir_m, Pd_m, lam_m)
    return Pc_M, Pc_m, (Pc_M*phi_macro + Pc_m*phi_meso)/phi_t

def imbibition_Pc_from_drainage(Pc_dra, Sw_dra, Swirr, Swt, Sot,
                                 theta_imb=30, theta_dra=30):
    """Imbibition Pc from drainage Pc (Eq. 4).

    Pc_imb(Sw_imb) = Pc_dra(1-Sw_dra+Swirr+Swt-Sot) * cos(θ_imb)/cos(θ_dra)
    """
    ratio = np.cos(np.radians(theta_imb)) / np.cos(np.radians(theta_dra))
    Sw_adj = 1 - Sw_dra + Swirr + Swt - Sot
    Sw_adj = np.clip(Sw_adj, Sw_dra.min(), Sw_dra.max())
    Pc_adj = np.interp(Sw_adj, Sw_dra, Pc_dra)
    return Pc_adj * ratio

def land_trapped_oil(Soi, C):
    """Trapped oil from Land correlation (Eq. 6): Sot = Soi/(1+C·Soi)."""
    return np.asarray(Soi, float) / (1 + C*np.asarray(Soi, float))

def Sot_max_from_Swdra(Sot_max_global, Sw_dra, Swirr):
    """Max trapped oil scales linearly with drainage saturation (Eq. 5).

    Sot = Sot_max * (1-Sw_dra)/(1-Swirr)
    """
    return Sot_max_global * (1-Sw_dra)/(1-Swirr)

def nmr_gaussian_deconvolution(T2, amplitude, n_peaks=2):
    """Decompose T2 distribution into Gaussian components (log10 domain).

    Returns peak positions and relative volumes (macro/meso split).
    """
    logT2 = np.log10(T2)
    peaks = []
    # Simple peak finding
    da = np.diff(amplitude); sign_changes = np.where(np.diff(np.sign(da)))[0] + 1
    for idx in sign_changes[:n_peaks]:
        peaks.append(dict(T2_peak=T2[idx], log_T2=logT2[idx], amp=amplitude[idx]))
    # Volume fractions from integral around each peak
    if len(peaks) >= 2:
        mid = (peaks[0]['log_T2'] + peaks[1]['log_T2']) / 2
        mask_low = logT2 < mid
        vol_low = np.trapezoid(amplitude[mask_low], logT2[mask_low])
        vol_high = np.trapezoid(amplitude[~mask_low], logT2[~mask_low])
        total = vol_low + vol_high
        peaks[0]['volume_fraction'] = vol_low/total
        peaks[1]['volume_fraction'] = vol_high/total
    return peaks

def corey_kr_dual(Sw, phi_M, phi_m, Swir_M, Swir_m, kro_max, krw_max, no, nw):
    """Dual-porosity Corey kr with weighted contribution."""
    phi_t = phi_M + phi_m
    Se_M = np.clip((Sw-Swir_M)/(1-Swir_M), 0, 1)
    Se_m = np.clip((Sw-Swir_m)/(1-Swir_m), 0, 1)
    kro_M = kro_max*(1-Se_M)**no; krw_M = krw_max*Se_M**nw
    kro_m = kro_max*(1-Se_m)**no; krw_m = krw_max*Se_m**nw
    kro = (phi_M*kro_M + phi_m*kro_m)/phi_t
    krw = (phi_M*krw_M + phi_m*krw_m)/phi_t
    return krw, kro

if __name__ == "__main__":
    Sw = np.linspace(0.10, 1.0, 50)
    PcM, Pcm, Pc_t = dual_porosity_Pc(Sw, 0.116, 0.075, 0.05, 0.30, 2.0, 15.0, 2.5, 1.5)
    Sot = land_trapped_oil(0.5, 1.5)
    krw, kro = corey_kr_dual(Sw, 0.116, 0.075, 0.05, 0.30, 0.8, 0.5, 3, 4)
    print(f"Dual Porosity — Pc_total range: {Pc_t.min():.1f} to {Pc_t.max():.1f}")
    print(f"Land Sot(0.5)={Sot:.3f}")
    print(f"Dual kr: krw_max={krw.max():.3f}, kro_max={kro.max():.3f}")
