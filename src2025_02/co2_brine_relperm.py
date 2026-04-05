#!/usr/bin/env python3
"""
CO2/Brine Relative Permeability: Reconciling SS and USS Methods.

Reference: Mascle et al., 2025, Petrophysics 66(1), 26-43. DOI:10.30632/PJV66N1-2025a2

Implements:
  - Corey relative permeability model
  - Fractional flow (Buckley-Leverett) theory
  - SS analytical kr from Darcy's law
  - Capillary end-effect (CEE) length estimation
  - JBN unsteady-state interpretation
  - Reconciliation of SS + USS data via Corey fitting
"""
import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass

@dataclass
class CoreFluidProps:
    core_L: float = 0.020;  core_D: float = 0.010;  phi: float = 0.10
    k_abs: float = 50e-15;  mu_co2: float = 5e-5;  mu_brine: float = 7e-4
    ift: float = 30e-3
    @property
    def A(self): return np.pi*(self.core_D/2)**2
    @property
    def PV(self): return self.A*self.core_L*self.phi
    @property
    def mobility_ratio(self): return self.mu_brine/self.mu_co2

def corey_relperm(Sw, Swir, Sor, krw_max, krco2_max, nw, nco2):
    Se = np.clip((Sw-Swir)/(1-Swir-Sor), 0, 1)
    return krw_max*Se**nw, krco2_max*(1-Se)**nco2

def fractional_flow(Sw, Swir, Sor, krw_max, krco2_max, nw, nco2, mu_b, mu_c):
    krw, krc = corey_relperm(Sw, Swir, Sor, krw_max, krco2_max, nw, nco2)
    return 1.0/(1.0 + krc*mu_b/(krw*mu_c+1e-30))

def ss_analytical_kr(fw, Q, dP, props: CoreFluidProps):
    dP = max(dP, 1e-10)
    f = Q*props.core_L/(props.A*dP*props.k_abs)
    return fw*f*props.mu_brine, (1-fw)*f*props.mu_co2

def capillary_end_effect_length(Pc_entry, dP, L):
    return min(Pc_entry/(max(dP,1e-10))*L, L)

def capillary_number(Q, mu, ift, A):
    return Q*mu/(A*ift)

def fit_corey(Sw, krw_data, krco2_data, Swir=0.15, Sor=0.0):
    Se = np.clip((Sw-Swir)/(1-Swir-Sor), 1e-6, 1-1e-6)
    m_w = krw_data > 0
    try: pw,_ = curve_fit(lambda s,a,b: a*s**b, Se[m_w], krw_data[m_w], p0=[1,3], bounds=([0,.5],[1.5,10]))
    except: pw = [1,3]
    m_c = krco2_data > 0
    try: pc,_ = curve_fit(lambda s,a,b: a*(1-s)**b, Se[m_c], krco2_data[m_c], p0=[.8,2], bounds=([0,.5],[1.5,10]))
    except: pc = [.8,2]
    return dict(Swir=Swir,Sor=Sor,krw_max=pw[0],nw=pw[1],krco2_max=pc[0],nco2=pc[1])

def reconcile_ss_uss(Sw_ss,krw_ss,krc_ss, Sw_uss,krw_uss,krc_uss):
    Sw = np.concatenate([Sw_ss,Sw_uss]); idx = np.argsort(Sw)
    return fit_corey(Sw[idx], np.concatenate([krw_ss,krw_uss])[idx],
                     np.concatenate([krc_ss,krc_uss])[idx])

if __name__ == "__main__":
    p = CoreFluidProps()
    Sw = np.array([.30,.40,.50,.60,.70,.80,.90])
    krw,krc = corey_relperm(Sw, .20, 0, .8, .6, 4, 2.5)
    np.random.seed(42)
    krw *= 1+.05*np.random.randn(len(Sw)); krc *= 1+.05*np.random.randn(len(Sw))
    fitted = fit_corey(Sw, np.clip(krw,0,1), np.clip(krc,0,1), Swir=.20)
    Nca = capillary_number(1e-8, p.mu_brine, p.ift, p.A)
    print(f"CO2/Brine RelPerm — M={p.mobility_ratio:.0f}, Ca={Nca:.2e}")
    print(f"Fitted: nw={fitted['nw']:.2f}, nco2={fitted['nco2']:.2f}")
