#!/usr/bin/env python3
"""
Digital Rock Physics: Relative Permeability & Wettability Assessment.

Reference: Regaieg et al., 2025, Petrophysics 66(1), 80-92. DOI:10.30632/PJV66N1-2025a6

Implements:
  - Pore-network model generation from pore-size distributions
  - Contact angle assignment for wettability anchoring
  - Simplified invasion-percolation kr simulation
  - ESRGAN resolution enhancement metrics
  - DRP vs SCAL comparison framework
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class WettabilityAnchor:
    theta_adv: float = 40; theta_rec: float = 20; theta_std: float = 15
    frac_oil_wet: float = 0.3; label: str = "mixed-wet"

def pore_size_dist(n, mean_r, std_r=None):
    if std_r is None: std_r = 0.3*mean_r
    sig = np.sqrt(np.log(1+(std_r/mean_r)**2)); mu = np.log(mean_r)-0.5*sig**2
    return np.random.lognormal(mu, sig, n)

def assign_contacts(n, anchor: WettabilityAnchor):
    a = np.zeros(n); nw = int(n*anchor.frac_oil_wet)
    a[:n-nw] = np.random.normal(anchor.theta_rec, anchor.theta_std, n-nw)
    a[n-nw:] = np.random.normal(180-anchor.theta_adv, anchor.theta_std, nw)
    return np.clip(a, 0, 180)

def invasion_drainage_kr(pore_r, throat_r, theta, ift=30.0, n_steps=20):
    Pc_e = 2*ift*np.cos(np.radians(theta[:len(throat_r)]))/(throat_r+1e-6)
    order = np.argsort(-Pc_e)
    total = np.sum(throat_r**4); nt = len(throat_r)
    Sw = np.linspace(1, 0.1, n_steps); krw = np.zeros(n_steps); kro = np.zeros(n_steps)
    for i, sw in enumerate(Sw):
        ni = int((1-sw)*nt); inv = order[:ni]; ninv = order[ni:]
        if len(ninv)>0: krw[i] = np.sum(throat_r[ninv]**4)/total
        if len(inv)>0: kro[i] = np.sum(throat_r[inv]**4)/total
    return Sw, krw, kro

def esrgan_metrics(res_orig, res_enh):
    f = res_orig/res_enh
    return dict(linear=f, volumetric=f**3, orig_um=res_orig, enh_um=res_enh)

def compare_drp_scal(Sw_d, krw_d, kro_d, Sw_s, krw_s, kro_s):
    lo = max(Sw_d.min(), Sw_s.min()); hi = min(Sw_d.max(), Sw_s.max())
    Sc = np.linspace(lo, hi, 20)
    d_w = np.interp(Sc, np.sort(Sw_d), krw_d[np.argsort(Sw_d)])
    s_w = np.interp(Sc, np.sort(Sw_s), krw_s[np.argsort(Sw_s)])
    d_o = np.interp(Sc, np.sort(Sw_d), kro_d[np.argsort(Sw_d)])
    s_o = np.interp(Sc, np.sort(Sw_s), kro_s[np.argsort(Sw_s)])
    return dict(rmse_krw=np.sqrt(np.mean((d_w-s_w)**2)), rmse_kro=np.sqrt(np.mean((d_o-s_o)**2)),
                corr_krw=np.corrcoef(d_w, s_w)[0,1])

if __name__ == "__main__":
    np.random.seed(42)
    anchor = WettabilityAnchor()
    pr = pore_size_dist(5000, 25); tr = pore_size_dist(10000, 10)
    ca = assign_contacts(10000, anchor)
    Sw, krw, kro = invasion_drainage_kr(pr, tr, ca)
    enh = esrgan_metrics(10.0, 2.5)
    print(f"DRP — {anchor.label}, θ_mean={ca.mean():.1f}°, oil-wet frac={anchor.frac_oil_wet:.0%}")
    print(f"kr: Sw={Sw[0]:.2f}→{Sw[-1]:.2f}, krw_max={krw.max():.3f}, kro_max={kro.max():.3f}")
    print(f"ESRGAN: {enh['linear']:.0f}x linear, {enh['volumetric']:.0f}x volumetric")
