"""
Analog Two-Phase Relative Permeability for CO₂/Brine Estimation
================================================================

Implements the workflow of:

    Schembre-McCabe, J., Akbarabadi, M., Burger, J., Rauschhuber, M.,
    and Richardson, W., 2025,
    "Use of Analog Two-Phase Relative Permeability Data to Estimate
    Drainage CO₂/Brine Relative Permeability Curves",
    Petrophysics, 66(6), 969–981.
    DOI: 10.30632/PJV66N6-2025a4

Key ideas
---------
* Capillary-number-based comparisons across fluid pairs (Eq. 1).
* Total and phase mobility (Eqs. 2-3).
* Fractional flow (Buckley–Leverett, Eq. 4).
* Use of analog fluids (N₂/mineral oil, mineral oil/brine) to estimate
  CO₂/brine drainage kr curves at matching dimensionless conditions.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────
# 1. Capillary Number  (Eq. 1, Saffman & Taylor 1958)
# ──────────────────────────────────────────────────────────────────────
def capillary_number(
    mu: float,
    v: float,
    sigma: float,
) -> float:
    """Capillary number  Nca = μ·v / σ   [Eq. 1].

    Parameters
    ----------
    mu : float
        Displacing-fluid dynamic viscosity (Pa·s).
    v : float
        Darcy velocity (m/s).
    sigma : float
        Interfacial tension (N/m).

    Returns
    -------
    float
        Capillary number (dimensionless).
    """
    return mu * v / sigma


# ──────────────────────────────────────────────────────────────────────
# 2. Phase and Total Mobility  (Eqs. 2-3)
# ──────────────────────────────────────────────────────────────────────
def phase_mobility(kr: np.ndarray, mu: float) -> np.ndarray:
    """Mobility of phase α:  λα = krα / μα   [Eq. 3].

    Parameters
    ----------
    kr : array_like
        Relative permeability of the phase.
    mu : float
        Dynamic viscosity of the phase (Pa·s).
    """
    return np.asarray(kr, float) / mu


def total_mobility(
    kr_nw: np.ndarray, mu_nw: float,
    kr_w: np.ndarray, mu_w: float,
) -> np.ndarray:
    """Total mobility  λ_t = λ_nw + λ_w   [Eq. 2].

    Parameters
    ----------
    kr_nw, kr_w : array_like
        Non-wetting and wetting phase relative permeabilities.
    mu_nw, mu_w : float
        Viscosities (Pa·s).

    Returns
    -------
    np.ndarray
        Total mobility at each saturation point.
    """
    return phase_mobility(kr_nw, mu_nw) + phase_mobility(kr_w, mu_w)


# ──────────────────────────────────────────────────────────────────────
# 3. Fractional Flow  (Eq. 4, Buckley & Leverett 1942)
# ──────────────────────────────────────────────────────────────────────
def fractional_flow(
    kr_inj: np.ndarray, mu_inj: float,
    kr_disp: np.ndarray, mu_disp: float,
) -> np.ndarray:
    """Fractional flow of injected phase [Eq. 4].

    f_inj = λ_inj / (λ_inj + λ_disp)

    Gravity and capillary terms neglected (1-D horizontal,
    incompressible, immiscible).

    Parameters
    ----------
    kr_inj, kr_disp : array_like
        Relative permeabilities of injected and displaced phases.
    mu_inj, mu_disp : float
        Viscosities (Pa·s).

    Returns
    -------
    np.ndarray
        Fractional flow of the injected phase.
    """
    lam_inj = phase_mobility(kr_inj, mu_inj)
    lam_disp = phase_mobility(kr_disp, mu_disp)
    denom = lam_inj + lam_disp
    # Avoid division by zero at endpoints
    return np.where(denom > 0, lam_inj / denom, 0.0)


# ──────────────────────────────────────────────────────────────────────
# 4. Viscosity Ratio
# ──────────────────────────────────────────────────────────────────────
def viscosity_ratio(mu_nw: float, mu_w: float) -> float:
    """Non-wetting to wetting phase viscosity ratio."""
    return mu_nw / mu_w


# ──────────────────────────────────────────────────────────────────────
# 5. Analog Fluid Kr Scaling Workflow
# ──────────────────────────────────────────────────────────────────────
class FluidPair:
    """Represent a fluid-pair system (e.g., N₂/brine, mineral oil/brine).

    Attributes
    ----------
    name : str
    mu_nw : float   Non-wetting phase viscosity (Pa·s).
    mu_w : float    Wetting phase viscosity (Pa·s).
    ift : float     Interfacial tension (N/m).
    """

    def __init__(self, name: str, mu_nw: float, mu_w: float, ift: float):
        self.name = name
        self.mu_nw = mu_nw
        self.mu_w = mu_w
        self.ift = ift

    @property
    def visc_ratio(self) -> float:
        return viscosity_ratio(self.mu_nw, self.mu_w)

    def Nca(self, v: float) -> float:
        """Capillary number at Darcy velocity *v*."""
        return capillary_number(self.mu_nw, v, self.ift)

    def __repr__(self) -> str:
        return (f"FluidPair({self.name!r}, μ_nw={self.mu_nw:.2e}, "
                f"μ_w={self.mu_w:.2e}, IFT={self.ift:.4f})")


# Typical fluid systems (Table based on Figs. 1-2 of the paper)
CO2_BRINE = FluidPair("CO2/brine", mu_nw=3e-5, mu_w=5e-4, ift=0.030)
N2_BRINE = FluidPair("N2/brine", mu_nw=1.8e-5, mu_w=1e-3, ift=0.060)
N2_MINERAL_OIL = FluidPair("N2/mineral_oil", mu_nw=1.8e-5, mu_w=5e-3, ift=0.025)
OIL_BRINE = FluidPair("mineral_oil/brine", mu_nw=5e-3, mu_w=1e-3, ift=0.040)


def select_analog_pair(
    target: FluidPair,
    candidates: list[FluidPair],
    weight_visc: float = 0.5,
    weight_ift: float = 0.5,
) -> FluidPair:
    """Choose the best analog fluid pair by matching viscosity ratio and IFT.

    Uses a weighted Euclidean distance in log-space.
    """
    def _distance(a: FluidPair, b: FluidPair) -> float:
        d_vr = (np.log10(a.visc_ratio) - np.log10(b.visc_ratio)) ** 2
        d_ift = (np.log10(a.ift) - np.log10(b.ift)) ** 2
        return float(weight_visc * d_vr + weight_ift * d_ift)

    return min(candidates, key=lambda c: _distance(target, c))


def rescale_kr_to_target_viscosity(
    Sw: np.ndarray,
    kr_nw_analog: np.ndarray,
    kr_w_analog: np.ndarray,
    analog: FluidPair,
    target: FluidPair,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute fractional flow and mobility using analog kr curves
    but target viscosities (Figs. 8-10 of the paper).

    Returns total mobility and fractional flow at the target field
    viscosity ratio.
    """
    lam_t = total_mobility(kr_nw_analog, target.mu_nw,
                           kr_w_analog, target.mu_w)
    f_nw = fractional_flow(kr_nw_analog, target.mu_nw,
                           kr_w_analog, target.mu_w)
    return Sw, lam_t, f_nw


def buckley_leverett_displacement(
    Sw: np.ndarray,
    f_nw: np.ndarray,
    pore_volumes_injected: np.ndarray,
) -> np.ndarray:
    """Wetting-phase saturation vs PVI using Welge tangent construction.

    Parameters
    ----------
    Sw : 1-D array
        Water saturation values (monotonically decreasing).
    f_nw : 1-D array
        Non-wetting fractional flow at each Sw.
    pore_volumes_injected : 1-D array
        PVI values at which to evaluate average saturation.

    Returns
    -------
    np.ndarray
        Average wetting-phase saturation at each PVI.
    """
    # Derivative df/dSw (using *wetting* saturation)
    Sw = np.asarray(Sw, float)
    f_w = 1.0 - np.asarray(f_nw, float)
    df_dSw = np.gradient(f_w, Sw)

    # Welge: PVI = 1 / (df_w/dSw) at the shock front
    pvi = pore_volumes_injected
    Sw_avg = np.zeros_like(pvi)
    for i, pv in enumerate(pvi):
        if pv <= 0:
            Sw_avg[i] = Sw[0]
            continue
        # Find front: df/dSw = 1/PVI
        target_slope = 1.0 / pv
        idx = np.argmin(np.abs(df_dSw - target_slope))
        Sw_front = Sw[idx]
        f_front = f_w[idx]
        # Average saturation behind front
        Sw_avg[i] = Sw_front + (1.0 - f_front) * pv
    return Sw_avg


# ──────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from petrophysics_v66n6.pgs_rock_typing import corey_curves

    # Generate analog kr (mineral oil / brine)
    Sw, kro, krw = corey_curves(Swir=0.15, Sorw=0.20,
                                kro0=1.0, krw0=0.25,
                                no=2.5, nw=3.5)

    # Compare fractional flow at analog vs target viscosities
    f_analog = fractional_flow(kro, OIL_BRINE.mu_nw, krw, OIL_BRINE.mu_w)
    f_target = fractional_flow(kro, CO2_BRINE.mu_nw, krw, CO2_BRINE.mu_w)

    print("Sw     f_nw(analog)   f_nw(CO2/brine)")
    for i in range(0, len(Sw), 20):
        print(f"{Sw[i]:.3f}  {f_analog[i]:.4f}         {f_target[i]:.4f}")

    # Capillary numbers
    v = 1e-5  # m/s
    for fp in [CO2_BRINE, N2_BRINE, N2_MINERAL_OIL, OIL_BRINE]:
        print(f"{fp.name:20s}  Nca = {fp.Nca(v):.2e}  "
              f"visc_ratio = {fp.visc_ratio:.3f}")
