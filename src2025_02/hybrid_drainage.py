#!/usr/bin/env python3
"""
Hybrid Drainage Technique (HDT) on Bimodal Limestone.

Reference: Fernandes et al., 2025, Petrophysics 66(1), 94-109. DOI:10.30632/PJV66N1-2025a7

Implements:
  - Bimodal pore-system characterisation (macro + meso)
  - Hybrid Drainage Technique: viscous flood + capillary steps
  - Viscous Oilflood (VOF) protocol
  - NMR T2 bimodal distribution generation
  - Saturation profile homogeneity assessment
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class BimodalPores:
    phi_macro: float = 0.18; phi_meso: float = 0.10
    r_macro: float = 50.0; r_meso: float = 5.0; connectivity: float = 0.3
    @property
    def phi_total(self): return self.phi_macro + self.phi_meso

def Pc_entry(r_um, ift_mNm, theta_deg=0):
    return 2*ift_mNm*1e-3*np.cos(np.radians(theta_deg))/(r_um*1e-6)

def bimodal_t2(pores: BimodalPores, rho_s=10.0, n=200):
    T2 = np.logspace(-1, 4, n)
    T2m = pores.r_macro/(3*rho_s)*1000; T2s = pores.r_meso/(3*rho_s)*1000
    amp = (pores.phi_macro*np.exp(-0.5*((np.log10(T2)-np.log10(T2m))/0.5)**2) +
           pores.phi_meso*np.exp(-0.5*((np.log10(T2)-np.log10(T2s))/0.4)**2))
    return T2, amp/np.trapezoid(amp, np.log10(T2))

def hdt_protocol(pores: BimodalPores, target_Swi, ift=25.0, n_steps=5):
    Sw_vof = pores.phi_meso/pores.phi_total + 0.05
    Pc_ma = Pc_entry(pores.r_macro, ift); Pc_me = Pc_entry(pores.r_meso, ift)
    steps = [Sw_vof]
    for i in range(n_steps):
        frac = min((i+1)/n_steps, 1)*pores.connectivity
        dSw = frac*pores.phi_meso/pores.phi_total/n_steps
        steps.append(max(steps[-1]-dSw, target_Swi))
    return dict(method="HDT", final_Swi=steps[-1], target_ok=steps[-1]<=target_Swi+0.02, Sw=np.array(steps))

def vof_protocol(pores: BimodalPores, target_Swi, n_bumps=3):
    Sw_min = pores.phi_meso/pores.phi_total + (1-pores.connectivity)*0.1
    Sw = 1.0; steps = []
    for _ in range(n_bumps+1):
        Sw -= (Sw-Sw_min)*0.6; steps.append(Sw)
    return dict(method="VOF", final_Swi=steps[-1], target_ok=steps[-1]<=target_Swi+0.02, limited=steps[-1]>target_Swi)

def profile_homogeneity(Sw_profile):
    m = np.mean(Sw_profile); cv = np.std(Sw_profile)/(m+1e-30)
    return dict(mean=m, cv=cv, homogeneous=cv<0.05)

if __name__ == "__main__":
    p = BimodalPores()
    hdt = hdt_protocol(p, 0.20); vof = vof_protocol(p, 0.20)
    T2, amp = bimodal_t2(p)
    np.random.seed(42)
    hom = profile_homogeneity(hdt['final_Swi']+0.01*np.random.randn(20))
    print(f"HDT Bimodal — φ_tot={p.phi_total:.2f}, macro={p.r_macro:.0f}µm, meso={p.r_meso:.0f}µm")
    print(f"HDT: Swi={hdt['final_Swi']:.3f} (ok={hdt['target_ok']})")
    print(f"VOF: Swi={vof['final_Swi']:.3f} (ok={vof['target_ok']}, limited={vof['limited']})")
    print(f"Profile CV={hom['cv']:.4f}, homogeneous={hom['homogeneous']}")
