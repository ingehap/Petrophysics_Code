#!/usr/bin/env python3
"""
SCAL Model for CCS: LET Correlations for Relative Permeability and Capillary Pressure.

Reference: Ebeltoft et al., 2025, Petrophysics 66(1), 10-25. DOI:10.30632/PJV66N1-2025a1

Implements:
  - LET correlation for relative permeability (Eq. 1)
  - LET correlation for capillary pressure (Eq. 2)
  - Normalized water saturation (Eq. 3)
  - Land trapping correlation for residual CO2
  - Leverett J-function scaling between fluid systems
  - CO2 storage capacity estimation
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class LETRelPermParams:
    """LET relative-permeability parameters (Lomeland et al., 2005, 2008)."""
    Lg: float = 2.0; Eg: float = 1.0; Tg: float = 1.5
    Lw: float = 3.0; Ew: float = 1.0; Tw: float = 1.5
    krg_max: float = 0.8; krw_max: float = 1.0; Swr: float = 0.15

@dataclass
class LETCapPresParams:
    """LET capillary-pressure parameters."""
    Lfp: float = 2.0; Efp: float = 1.0; Tfp: float = 1.5
    Pcgw_fp: float = 25.0; Pcgw_tp: float = 0.5; Swr: float = 0.15

def normalized_water_saturation(Sw, Swr):
    """Swn = (Sw - Swr) / (1 - Swr)  [Eq. 3]."""
    return np.clip((np.asarray(Sw, float) - Swr) / (1.0 - Swr), 0, 1)

def let_relative_permeability(Sw, params: LETRelPermParams):
    """LET relative permeability (Eq. 1). Returns (krg, krw)."""
    Swn = normalized_water_saturation(Sw, params.Swr)
    ng = (1-Swn)**params.Lg;  dg = ng + params.Eg * Swn**params.Tg
    nw = Swn**params.Lw;      dw = nw + params.Ew * (1-Swn)**params.Tw
    dg = np.where(dg==0, 1e-30, dg); dw = np.where(dw==0, 1e-30, dw)
    return np.clip(params.krg_max*ng/dg, 0, 1), np.clip(params.krw_max*nw/dw, 0, 1)

def let_capillary_pressure(Sw, params: LETCapPresParams):
    """LET capillary pressure (Eq. 2)."""
    Swn = normalized_water_saturation(Sw, params.Swr)
    n = (1-Swn)**params.Lfp; d = n + params.Efp * Swn**params.Tfp
    d = np.where(d==0, 1e-30, d)
    return (params.Pcgw_fp - params.Pcgw_tp) * n / d + params.Pcgw_tp

def leverett_j_scaling(Pc, ift_orig, ift_target, k_orig=None, k_tgt=None, phi_orig=None, phi_tgt=None):
    """Leverett J-function scaling of Pc between fluid systems."""
    s = ift_target / ift_orig
    if all(v is not None for v in (k_orig, k_tgt, phi_orig, phi_tgt)):
        s *= np.sqrt((k_orig * phi_tgt) / (k_tgt * phi_orig))
    return Pc * s

def land_trapping(Sgi, C):
    """Land (1968) trapping: Sgt = Sgi / (1 + C·Sgi)."""
    return np.asarray(Sgi, float) / (1.0 + C * np.asarray(Sgi, float))

def land_coefficient(Sgi_max, Sgt_max):
    """C = 1/Sgt_max - 1/Sgi_max."""
    return 1.0/Sgt_max - 1.0/Sgi_max

def co2_storage_capacity(pore_volume, Swr, Sgt, rho_co2=700.0):
    """Structural and residual CO2 storage [tonnes]."""
    struct = pore_volume * (1-Swr) * rho_co2 / 1000
    resid  = pore_volume * Sgt     * rho_co2 / 1000
    return struct, resid

@dataclass
class SCALModelCCS:
    """SCAL model with base / optimistic / pessimistic cases."""
    base: LETRelPermParams = field(default_factory=LETRelPermParams)
    optimistic: LETRelPermParams = field(default_factory=lambda: LETRelPermParams(Lg=3,Eg=1.5,Tg=1,Lw=2,Ew=0.5,Tw=2,krg_max=0.5,Swr=0.10))
    pessimistic: LETRelPermParams = field(default_factory=lambda: LETRelPermParams(Lg=1.5,Eg=0.5,Tg=2,Lw=4,Ew=2,Tw=1,krg_max=1.0,Swr=0.20))
    def evaluate(self, Sw, case="base"):
        return let_relative_permeability(Sw, getattr(self, case))

if __name__ == "__main__":
    Sw = np.linspace(0.15, 1.0, 50)
    p = LETRelPermParams(Lg=2.5,Eg=1.2,Tg=1.3,Lw=3.5,Ew=1,Tw=1.5,krg_max=0.7,Swr=0.15)
    krg, krw = let_relative_permeability(Sw, p)
    pc_p = LETCapPresParams(Lfp=2,Efp=1.5,Tfp=1.5,Pcgw_fp=20,Pcgw_tp=0.3,Swr=0.15)
    Pc = let_capillary_pressure(Sw, pc_p)
    C = land_coefficient(0.85, 0.35)
    Sgt = land_trapping(0.85, C)
    s, r = co2_storage_capacity(1e6, 0.15, 0.35)
    print(f"SCAL Model CCS — krg(Swr)={krg[0]:.3f}, krw(1)={krw[-1]:.3f}, Pc(Swr)={Pc[0]:.1f}")
    print(f"Land C={C:.3f}, Sgt(0.85)={Sgt:.3f}")
    print(f"Storage: structural={s:.0f} t, residual={r:.0f} t")
