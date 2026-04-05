#!/usr/bin/env python3
"""
Impact of Dopants (NaI) on SCAL Experiments.

Reference: Pairoys et al., 2025, Petrophysics 66(1), 123-133. DOI:10.30632/PJV66N1-2025a9

Implements:
  - X-ray attenuation contrast with/without NaI doping
  - Amott wettability index computation
  - Oil recovery comparison: doped vs undoped brine
  - Spontaneous imbibition rate analysis
  - Residual oil saturation dopant impact
"""
import numpy as np

def xray_attenuation(fluid, nai_mol=0.0):
    base = {'brine': 0.0208, 'oil': 0.0165, 'doped_brine': 0.058}
    if fluid=='brine' and nai_mol>0: return 0.0208+(0.058-0.0208)*min(nai_mol/0.4, 1)
    return base.get(fluid, 0.02)

def contrast_ratio(att_brine, att_oil):
    return abs(att_brine-att_oil)/att_oil

def amott_index(Vw_sp, Vw_fo, Vo_sp, Vo_fo):
    Iw = Vw_sp/(Vw_sp+Vw_fo) if Vw_sp+Vw_fo>0 else 0
    Io = Vo_sp/(Vo_sp+Vo_fo) if Vo_sp+Vo_fo>0 else 0
    IAH = Iw-Io
    if IAH>0.3: c="water-wet"
    elif IAH>0.1: c="weakly water-wet"
    elif IAH>-0.1: c="intermediate"
    elif IAH>-0.3: c="weakly oil-wet"
    else: c="oil-wet"
    return dict(Iw=Iw, Io=Io, IAH=IAH, classification=c)

def recovery_comparison(rec_undoped, rec_doped, pvi):
    fu = rec_undoped[-1]; fd = rec_doped[-1]; d = fu-fd
    return dict(undoped=fu, doped=fd, diff=d, rel=d/(fu+1e-10), significant=abs(d)>0.05)

def imbibition_rate(time, recovery):
    return np.gradient(recovery, np.sqrt(time+1e-10))

def sor_impact(Sor_undoped, Sor_doped):
    mu = np.mean(Sor_undoped); md = np.mean(Sor_doped)
    return dict(mean_undoped=mu, mean_doped=md, delta=md-mu, increases_Sor=md>mu)

if __name__ == "__main__":
    au = xray_attenuation('brine'); ad = xray_attenuation('brine', 0.4); ao = xray_attenuation('oil')
    cr_u = contrast_ratio(au, ao); cr_d = contrast_ratio(ad, ao)
    am_u = amott_index(0.15, 0.10, 0.02, 0.08)
    am_d = amott_index(0.05, 0.15, 0.04, 0.08)
    pvi = np.linspace(0,5,50)
    rc = recovery_comparison(0.55*(1-np.exp(-1.5*pvi)), 0.45*(1-np.exp(-1.2*pvi)), pvi)
    si = sor_impact(np.array([.25,.22,.28]), np.array([.32,.35,.30]))
    print(f"Dopant Impact — contrast: undoped={cr_u:.2f}, doped={cr_d:.2f} ({cr_d/cr_u:.0f}x)")
    print(f"Amott: undoped={am_u['IAH']:.3f}({am_u['classification']}), doped={am_d['IAH']:.3f}({am_d['classification']})")
    print(f"Recovery diff: {rc['diff']:.1%}, Sor increase: {si['delta']:.3f}")
