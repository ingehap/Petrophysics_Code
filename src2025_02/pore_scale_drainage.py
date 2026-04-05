#!/usr/bin/env python3
"""
Pore-Scale Comparisons of Primary Drainage Techniques on Non-Water-Wet Rocks.

Reference: Nono et al., 2025, Petrophysics 66(1), 110-122. DOI:10.30632/PJV66N1-2025a8

Implements:
  - Porous-plate (PP) vs oilflood (OF) invasion simulation
  - Pore occupancy analysis by size class
  - Effective permeability from pore occupancy
  - Wettability-dependent artifact identification
"""
import numpy as np

def classify_pores(radii, thresholds=(5.0, 30.0)):
    v = radii**3; vt = v.sum()
    mi = radii<thresholds[0]; me = (radii>=thresholds[0])&(radii<thresholds[1]); ma = radii>=thresholds[1]
    return dict(micro=dict(mask=mi, vf=v[mi].sum()/vt), meso=dict(mask=me, vf=v[me].sum()/vt),
                macro=dict(mask=ma, vf=v[ma].sum()/vt))

def pp_invasion(radii, ift, theta, Pc_applied):
    Pc_e = 2*ift*1e-3*np.cos(np.radians(theta))/(radii*1e-6)
    return Pc_applied >= Pc_e

def of_invasion(radii, ift, theta, dp, L, positions=None):
    if positions is None: positions = np.random.uniform(0, L, len(radii))
    Pc_e = 2*ift*1e-3*np.cos(np.radians(theta))/(radii*1e-6)
    local_Pc = dp*(1-positions/L)
    return local_Pc >= Pc_e

def pore_occupancy(radii, inv_pp, inv_of, thresholds=(5., 30.)):
    cls = classify_pores(radii, thresholds); v = radii**3; vt = v.sum()
    res = {}
    for name, c in cls.items():
        m = c['mask']; n = m.sum()
        if n == 0: continue
        res[name] = dict(pp=inv_pp[m].sum()/n, of=inv_of[m].sum()/n)
    Swi_pp = 1-v[inv_pp].sum()/vt; Swi_of = 1-v[inv_of].sum()/vt
    res['Swi_pp'] = Swi_pp; res['Swi_of'] = Swi_of; res['pp_lower'] = Swi_pp < Swi_of
    return res

def keff_from_occupancy(radii, invaded, phase='oil'):
    t = np.sum(radii**4)
    if phase=='oil': return np.sum(radii[invaded]**4)/t
    return np.sum(radii[~invaded]**4)/t

def wettability_artifacts(theta, inv_pp, inv_of):
    ww = theta<90; ow = theta>=90
    res = {}
    for lbl, m in [('water_wet', ww), ('oil_wet', ow)]:
        n = m.sum()
        if n>0: res[lbl] = dict(pp=inv_pp[m].sum()/n, of=inv_of[m].sum()/n, diff=(inv_pp[m].sum()-inv_of[m].sum())/n)
    return res

if __name__ == "__main__":
    np.random.seed(42); n = 3000
    r = np.clip(np.random.lognormal(np.log(15), 0.6, n), 1, 200)
    theta = np.where(np.random.rand(n)<0.4, np.random.normal(120,15,n), np.random.normal(50,15,n))
    theta = np.clip(theta, 0, 180)
    inv_pp = pp_invasion(r, 25, theta, 15000)
    inv_of = of_invasion(r, 25, theta, 15000, 0.05)
    occ = pore_occupancy(r, inv_pp, inv_of)
    art = wettability_artifacts(theta, inv_pp, inv_of)
    print(f"Pore-Scale Drainage — Swi_PP={occ['Swi_pp']:.3f}, Swi_OF={occ['Swi_of']:.3f}, PP lower={occ['pp_lower']}")
    for k,v in art.items(): print(f"  {k}: PP={v['pp']:.2%}, OF={v['of']:.2%}")
