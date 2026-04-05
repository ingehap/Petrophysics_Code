#!/usr/bin/env python3
"""
Enhanced Gas Recovery (EGR) by CO2 Injection.

Reference: Jones et al., 2025, Petrophysics 66(1), 54-66. DOI:10.30632/PJV66N1-2025a4

Implements:
  - Land trapping correlation for CH4 and CO2
  - EGR displacement efficiency
  - Burdine capillary pressure model
  - LET relative permeability model
  - Gravity-stable flood criterion
  - ISSM saturation from X-ray attenuation
"""
import numpy as np

def land_trapping(Sgi, C):
    """Sgt = Sgi / (1 + C*Sgi)  (Land, 1968)."""
    Sgi = np.asarray(Sgi, float)
    return Sgi / (1.0 + C * Sgi)

def land_coefficient(Sgi_max, Sgt_max):
    return 1.0/Sgt_max - 1.0/Sgi_max

def compare_land_ch4_co2(C_ch4, Sgt_max_ch4, C_co2, Sgt_max_co2):
    red = 1.0 - Sgt_max_co2/Sgt_max_ch4
    return dict(C_ch4=C_ch4, C_co2=C_co2, reduction=red, co2_partial_wetting=C_co2>C_ch4)

def burdine_Pc(Sw, Swir, Pd, lam):
    Se = np.clip((Sw-Swir)/(1-Swir), 1e-6, 1)
    return Pd * Se**(-1.0/lam)

def let_kr_gas(Sw, Swir, krg_max, L, E, T):
    Swn = np.clip((Sw-Swir)/(1-Swir), 0, 1)
    n = (1-Swn)**L; d = n + E*Swn**T; d = np.where(d==0, 1e-30, d)
    return krg_max * n / d

def egr_efficiency(V_produced, V_initial):
    rf = V_produced/V_initial if V_initial>0 else 0
    return dict(recovery_factor=rf, complete=rf>0.95)

def issm_saturation(I_wet, I_dry, I_meas):
    d = np.where(np.abs(I_wet-I_dry)<1e-10, 1e-10, I_wet-I_dry)
    return np.clip((I_meas-I_dry)/d, 0, 1)

def gravity_stable(rho_inj, rho_disp, k, kr, mu, Q, A, angle=0):
    Qc = k*kr*A*abs(rho_disp-rho_inj)*9.81*np.cos(np.radians(angle))/mu
    return Q < Qc

if __name__ == "__main__":
    C4 = land_coefficient(0.90, 0.545); C2 = land_coefficient(0.78, 0.34)
    comp = compare_land_ch4_co2(C4, 0.545, C2, 0.34)
    egr = egr_efficiency(19.1, 18.6)
    Sw = np.linspace(0.15, 1, 30)
    Pc = burdine_Pc(Sw, 0.10, 5.0, 2.0)
    print(f"EGR — C_CH4={C4:.3f}, C_CO2={C2:.3f}, reduction={comp['reduction']:.1%}")
    print(f"Recovery={egr['recovery_factor']:.1%}, complete={egr['complete']}")
    print(f"Burdine Pc range: {Pc.min():.1f} to {Pc.max():.1f}")
