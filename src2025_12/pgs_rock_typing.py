"""
Pore Geometry Structure (PGS) Rock Typing and Corey-Parameter
Relative Permeability Trend Modelling
=============================================================

Implements the workflow of:

    Akbar, M.N.A., Putra, A.P., and Reppert, M.G., 2025,
    "A Versatile Workflow of Pore Geometry Structure Rock Typing and
    Corey Parameter-Based Relative Permeability Trend Modeling",
    Petrophysics, 66(6), 924–938.
    DOI: 10.30632/PJV66N6-2025a1

Key ideas
---------
* Kozeny-equation-based *pore geometry structure* (PGS) rock typing
  using (k/φ)^0.5 vs (k/φ³) on log–log scale.
* Corey relative-permeability parameterisation with wettability-
  consistent trend models linking Swi to Sorw, endpoint krw, and
  Corey exponents.

References
----------
Wibowo and Permadi, 2013 — PGS rock-type curve concept.
Lomeland et al., 2012; Ebeltoft et al., 2014 — SCAL trend modelling.
Corey, A.T., 1954 — Corey relative-permeability model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1.  PGS Rock-Type Classification  (Eqs. 1-3, Akbar et al.)
# ──────────────────────────────────────────────────────────────────────
def pore_geometry(k: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Mean hydraulic radius proxy  (k/φ)^0.5   [Eq. 1].

    Parameters
    ----------
    k : array_like
        Absolute permeability (mD).
    phi : array_like
        Porosity (fraction, 0-1).

    Returns
    -------
    np.ndarray
        (k / phi) ** 0.5
    """
    k, phi = np.asarray(k, float), np.asarray(phi, float)
    return np.sqrt(k / phi)


def pore_structure(k: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Pore structure parameter  k/φ³   [Eq. 2].

    Parameters
    ----------
    k : array_like
        Absolute permeability (mD).
    phi : array_like
        Porosity (fraction, 0-1).

    Returns
    -------
    np.ndarray
        k / phi**3
    """
    k, phi = np.asarray(k, float), np.asarray(phi, float)
    return k / phi**3


def pgs_rock_type_number(k: np.ndarray, phi: np.ndarray,
                         b: float = 0.5) -> np.ndarray:
    """Assign PGS rock-type numbers following the power-law relation [Eq. 3].

    The PGS plot is (k/φ)^0.5 vs (k/φ³) on log-log axes.  Parallel type
    curves are defined by constant *a* in::

        (k/phi)^0.5 = a * (k/phi^3)^b

    *b* is typically ≤ 0.5; for ideal capillary tubes b = 0.5.
    The intercept *a* increases with pore-system complexity and decreasing
    internal surface area — higher *a* ⇒ better reservoir quality.

    We discretise *a* on a log scale to assign integer rock-type numbers.
    The convention in the paper is RT-4 … RT-15.

    Parameters
    ----------
    k : array_like
        Permeability (mD).
    phi : array_like
        Porosity (fraction).
    b : float, optional
        Slope on PGS log-log plot (default 0.5).

    Returns
    -------
    np.ndarray of int
        Rock-type number for each sample.
    """
    pg = pore_geometry(k, phi)
    ps = pore_structure(k, phi)
    # Solve for a:  pg = a * ps^b  =>  a = pg / ps^b
    a = pg / ps**b
    # Map log10(a) to integer RT via equally spaced bins.
    # We use bins centred on half-integer log-a values
    # (following the Wibowo & Permadi 2013 convention).
    log_a = np.log10(a)
    rt = np.clip(np.round(log_a * 2 + 9).astype(int), 4, 15)
    return rt


# ──────────────────────────────────────────────────────────────────────
# 2.  Corey Relative Permeability  (Eqs. 4-6)
# ──────────────────────────────────────────────────────────────────────
def corey_kro(Sw: np.ndarray, Swir: float, Sorw: float,
              kro0: float = 1.0, no: float = 2.0) -> np.ndarray:
    """Oil relative permeability — Corey model [Eq. 4].

    kro = kro0 * ((1 - Sw - Sorw) / (1 - Swir - Sorw)) ^ no

    Parameters
    ----------
    Sw : array_like
        Water saturation (fraction).
    Swir : float
        Irreducible water saturation.
    Sorw : float
        Residual oil saturation to water.
    kro0 : float
        Endpoint oil relative permeability at Sw = Swir.
    no : float
        Corey exponent for oil.

    Returns
    -------
    np.ndarray
        Oil relative permeability.
    """
    Sw = np.asarray(Sw, float)
    denom = 1.0 - Swir - Sorw
    if denom <= 0:
        return np.zeros_like(Sw)
    Sn = np.clip((1.0 - Sw - Sorw) / denom, 0.0, 1.0)
    return kro0 * Sn**no


def corey_krw(Sw: np.ndarray, Swir: float, Sorw: float,
              krw0: float = 0.3, nw: float = 3.0) -> np.ndarray:
    """Water relative permeability — Corey model [Eq. 5].

    krw = krw0 * ((Sw - Swir) / (1 - Swir - Sorw)) ^ nw

    Parameters
    ----------
    Sw : array_like
        Water saturation (fraction).
    Swir : float
        Irreducible water saturation.
    Sorw : float
        Residual oil saturation to water.
    krw0 : float
        Endpoint water relative permeability at Sw = 1 - Sorw.
    nw : float
        Corey exponent for water.

    Returns
    -------
    np.ndarray
        Water relative permeability.
    """
    Sw = np.asarray(Sw, float)
    denom = 1.0 - Swir - Sorw
    if denom <= 0:
        return np.zeros_like(Sw)
    Sn = np.clip((Sw - Swir) / denom, 0.0, 1.0)
    return krw0 * Sn**nw


def corey_curves(Swir: float, Sorw: float,
                 kro0: float = 1.0, krw0: float = 0.3,
                 no: float = 2.0, nw: float = 3.0,
                 n_points: int = 100
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a complete pair of Corey kr curves.

    Returns
    -------
    Sw, kro, krw : 1-D arrays of length *n_points*.
    """
    Sw = np.linspace(Swir, 1.0 - Sorw, n_points)
    kro = corey_kro(Sw, Swir, Sorw, kro0, no)
    krw = corey_krw(Sw, Swir, Sorw, krw0, nw)
    return Sw, kro, krw


# ──────────────────────────────────────────────────────────────────────
# 3.  Wettability-Based Trend Models  (Eqs. 7-8)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TrendModelParams:
    """Calibration parameters for the Sorw–Swi and krw–Swi trend models.

    Attributes
    ----------
    A_Sorw : float
        Amplitude for Sorw–Swi trend [Eq. 7].
    B_Sorw : float
        Rate parameter for Sorw–Swi trend [Eq. 7].
    C_Sorw : float
        Offset for Sorw–Swi trend [Eq. 7].
    Swi_max_Sorw : float
        Swi at which Sorw reaches its maximum [Eq. 7].
    Akrw : float
        Maximum endpoint krw [Eq. 8].
    Bkrw : float
        Rate of decline of krw with Swi [Eq. 8].
    Ckrw : float
        Minimum endpoint krw [Eq. 8].
    """
    A_Sorw: float = 0.35
    B_Sorw: float = 5.0
    C_Sorw: float = 0.05
    Swi_max_Sorw: float = 0.15
    Akrw: float = 0.60
    Bkrw: float = 6.0
    Ckrw: float = 0.05


def sorw_trend(Swi: np.ndarray, p: TrendModelParams | None = None) -> np.ndarray:
    """Residual oil saturation as a function of Swi [Eq. 7].

    The trend first increases then decreases with Swi, consistent
    with Spiteri et al. (2008).

    Sorw(Swi) = A * Swi * exp(-B * (Swi - Swi_max)^2) + C
    """
    if p is None:
        p = TrendModelParams()
    Swi = np.asarray(Swi, float)
    return (p.A_Sorw * Swi
            * np.exp(-p.B_Sorw * (Swi - p.Swi_max_Sorw) ** 2)
            + p.C_Sorw)


def krw_endpoint_trend(Swi: np.ndarray,
                       p: TrendModelParams | None = None) -> np.ndarray:
    """Endpoint water relative permeability as a function of Swi [Eq. 8].

    krw(Sorw)(Swi) = Akrw * exp(-Bkrw * Swi) + Ckrw

    Higher Swi → more water-wet → lower krw endpoint.
    """
    if p is None:
        p = TrendModelParams()
    Swi = np.asarray(Swi, float)
    return p.Akrw * np.exp(-p.Bkrw * Swi) + p.Ckrw


def corey_exponent_oil_trend(Swi: np.ndarray,
                             no_min: float = 1.5,
                             no_max: float = 5.0,
                             rate: float = 8.0) -> np.ndarray:
    """Corey exponent for oil vs Swi (Fig. 5c).

    As Swi increases → more water-wet → higher no.
    """
    Swi = np.asarray(Swi, float)
    return no_min + (no_max - no_min) * (1.0 - np.exp(-rate * Swi))


def corey_exponent_water_trend(Swi: np.ndarray,
                               nw_min: float = 1.5,
                               nw_max: float = 6.0,
                               rate: float = 6.0) -> np.ndarray:
    """Corey exponent for water vs Swi (Fig. 5d).

    As Swi increases → more water-wet → higher nw (more depressed krw curve).
    """
    Swi = np.asarray(Swi, float)
    return nw_min + (nw_max - nw_min) * (1.0 - np.exp(-rate * Swi))


# ──────────────────────────────────────────────────────────────────────
# 4.  Full Workflow: from Rock Type to kr Curves
# ──────────────────────────────────────────────────────────────────────
@dataclass
class RockTypeKr:
    """Container for a rock-type's representative saturation functions."""
    rock_type: int
    Swi: float
    Sorw: float
    kro0: float
    krw0: float
    no: float
    nw: float
    Sw: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    kro: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    krw: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def generate_kr_for_rock_types(
    Swi_per_rt: dict[int, float],
    trend_params: TrendModelParams | None = None,
    n_points: int = 100,
) -> dict[int, RockTypeKr]:
    """Produce Corey kr curves for each rock type using trend models.

    Parameters
    ----------
    Swi_per_rt : dict
        Mapping of rock-type number → average initial water saturation.
    trend_params : TrendModelParams, optional
        Calibrated trend-model parameters.
    n_points : int
        Resolution of output curves.

    Returns
    -------
    dict mapping rock-type number to a ``RockTypeKr`` object.
    """
    if trend_params is None:
        trend_params = TrendModelParams()

    results: dict[int, RockTypeKr] = {}
    for rt, swi_val in Swi_per_rt.items():
        swi_arr = np.array([swi_val])
        Sorw = float(sorw_trend(swi_arr, trend_params)[0])
        krw0 = float(krw_endpoint_trend(swi_arr, trend_params)[0])
        no = float(corey_exponent_oil_trend(swi_arr)[0])
        nw = float(corey_exponent_water_trend(swi_arr)[0])
        kro0 = 1.0  # endpoint oil kr at Swir is conventionally 1
        Sw, kro, krw = corey_curves(swi_val, Sorw, kro0, krw0, no, nw,
                                    n_points)
        results[rt] = RockTypeKr(
            rock_type=rt, Swi=swi_val, Sorw=Sorw,
            kro0=kro0, krw0=krw0, no=no, nw=nw,
            Sw=Sw, kro=kro, krw=krw,
        )
    return results


# ──────────────────────────────────────────────────────────────────────
# 5.  Recovery Factor  (Eq. 9)
# ──────────────────────────────────────────────────────────────────────
def recovery_factor(cum_oil: float, ooip: float) -> float:
    """Oil Recovery Factor [Eq. 9].

    RF = Cumulative Oil Production / OOIP
    """
    if ooip <= 0:
        raise ValueError("OOIP must be positive")
    return cum_oil / ooip


# ──────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: assign rock types for synthetic data
    rng = np.random.default_rng(42)
    k_samples = 10 ** rng.uniform(0, 3, 200)   # 1–1000 mD
    phi_samples = rng.uniform(0.05, 0.35, 200)

    rt = pgs_rock_type_number(k_samples, phi_samples)
    print("Rock-type distribution:", dict(zip(*np.unique(rt, return_counts=True))))

    # Generate kr curves for groups 2–6 (Table 1 in the paper)
    Swi_map = {2: 0.08, 3: 0.11, 4: 0.15, 5: 0.20, 6: 0.30}
    kr_results = generate_kr_for_rock_types(Swi_map)
    for rt_num, rk in kr_results.items():
        print(f"RT-{rt_num}: Swi={rk.Swi:.2f}  Sorw={rk.Sorw:.3f}  "
              f"krw0={rk.krw0:.3f}  no={rk.no:.2f}  nw={rk.nw:.2f}")
