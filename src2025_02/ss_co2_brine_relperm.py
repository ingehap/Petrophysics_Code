#!/usr/bin/env python3
"""
Steady-State scCO2-Brine Relative Permeability at Different Pressures.

Reference: Richardson et al., 2025, Petrophysics 66(1), 44-53. DOI:10.30632/PJV66N1-2025a3

Implements:
  - Steady-state kr computation from fw/dP/Sw data
  - Pressure-effect comparison between different pore pressures
  - Hysteresis assessment (drainage vs imbibition)
  - Material balance validation
  - Wettability indicator from kr curve shape
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class SSTest:
    Pp: float = 12.4; T: float = 77.0; NCS: float = 17.4
    L: float = 0.307; D: float = 0.0381; phi: float = 0.165; k_mD: float = 51.8
    @property
    def A(self): return np.pi*(self.D/2)**2
    @property
    def PV(self): return self.A*self.L*self.phi

def compute_ss_kr(fw, dP, Sw, Q, mu_b, mu_c, k, L, A):
    dP = np.maximum(dP, 1e-10); f = Q*L/(k*A)
    return Sw, np.clip(fw*f*mu_b/dP, 0, 2), np.clip((1-fw)*f*mu_c/dP, 0, 2)

def pressure_effect(kr_lp, kr_hp, tol=0.05):
    lo, hi = max(kr_lp[0].min(), kr_hp[0].min()), min(kr_lp[0].max(), kr_hp[0].max())
    Sw = np.linspace(lo, hi, 20)
    rw = np.sqrt(np.mean((np.interp(Sw,kr_lp[0],kr_lp[1])-np.interp(Sw,kr_hp[0],kr_hp[1]))**2))
    rg = np.sqrt(np.mean((np.interp(Sw,kr_lp[0],kr_lp[2])-np.interp(Sw,kr_hp[0],kr_hp[2]))**2))
    return dict(rmse_krw=rw, rmse_krg=rg, effect_krw=rw>tol, effect_krg=rg>tol)

def hysteresis_check(Sw_d, kr_d, Sw_i, kr_i, tol=0.05):
    lo, hi = max(Sw_d.min(),Sw_i.min()), min(Sw_d.max(),Sw_i.max())
    Sw = np.linspace(lo, hi, 20)
    r = np.sqrt(np.mean((np.interp(Sw,Sw_d,kr_d)-np.interp(Sw,Sw_i,kr_i))**2))
    return dict(rmse=r, significant=r>tol)

def material_balance(Swi, Swf, Vb_prod, PV, tol=0.03):
    err = abs(Vb_prod/PV - (Swi-Swf))
    return dict(error=err, passes=err<tol)

def wettability_from_kr(Sw_cross, krw_Sor, krg_Swir):
    if Sw_cross > 0.5 and krw_Sor < krg_Swir: return "strongly water-wet"
    elif Sw_cross > 0.5: return "moderately water-wet"
    elif Sw_cross < 0.4: return "CO2-wet"
    return "intermediate"

if __name__ == "__main__":
    t = SSTest()
    np.random.seed(42)
    fw = np.array([1,.9,.7,.5,.3,.1,0.])
    Sw = np.array([1,.85,.65,.50,.38,.28,.22])
    dP = np.array([500,600,900,1200,1800,3000,5000.])
    k = t.k_mD*9.869e-16
    s,krw,krg = compute_ss_kr(fw, dP, Sw, 1e-8, 7e-4, 5e-5, k, t.L, t.A)
    eff = pressure_effect((s,krw,krg), (s,krw*(1+.03*np.random.randn(7)),krg*(1+.03*np.random.randn(7))))
    w = wettability_from_kr(0.55, 0.3, 0.6)
    print(f"SS scCO2-Brine — {t.Pp} MPa, {t.T}°C")
    print(f"Pressure effect: krw_rmse={eff['rmse_krw']:.4f}, krg_rmse={eff['rmse_krg']:.4f}")
    print(f"Wettability: {w}")
