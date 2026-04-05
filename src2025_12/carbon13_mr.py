"""
¹³C Magnetic Resonance Relaxation-Time Wettability Characterisation
====================================================================

Implements the ideas of:

    Ansaribaranghar, N., Zamiri, M.S., Pairoys, F., Fernandes, V.,
    Romero-Zerón, L., Marica, F., Ramírez Aguilera, A., Green, D.,
    Nicot, B., and Balcom, B.J., 2025,
    "Characterizing Wettability of Core Plugs Using ¹³C Magnetic
    Resonance Relaxation Times",
    Petrophysics, 66(6), 1073–1089.
    DOI: 10.30632/PJV66N6-2025a11

Key ideas
---------
* ¹³C MR sees *only* hydrocarbons — no water signal overlap.
* T₂ and T₁/T₂ ratio of ¹³C are sensitive to oil–surface interaction
  (wettability proxy).
* Surface-relaxivity model relating T₂ to pore-surface area / volume.
* Comparison of ¹³C vs ¹H relaxation for wettability interpretation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. Surface Relaxation Model
# ──────────────────────────────────────────────────────────────────────
def surface_relaxation_T2(
    rho2: float,
    S_over_V: float,
    T2_bulk: float = 1.0,
) -> float:
    """Observed T₂ with surface relaxation (fast-diffusion regime).

    1/T₂_obs = 1/T₂_bulk + ρ₂ (S/V)

    Parameters
    ----------
    rho2 : float
        Surface relaxivity for T₂ (m/s or µm/ms).
    S_over_V : float
        Surface-to-volume ratio of the pore (1/m or 1/µm).
    T2_bulk : float
        Bulk fluid T₂ relaxation time (s or ms — match units).

    Returns
    -------
    float
        Observed T₂ (same time unit as T2_bulk).
    """
    return 1.0 / (1.0 / T2_bulk + rho2 * S_over_V)


def surface_relaxation_T1(
    rho1: float,
    S_over_V: float,
    T1_bulk: float = 2.0,
) -> float:
    """Observed T₁ with surface relaxation.

    1/T₁_obs = 1/T₁_bulk + ρ₁ (S/V)
    """
    return 1.0 / (1.0 / T1_bulk + rho1 * S_over_V)


def T1_over_T2(
    rho1: float, rho2: float,
    S_over_V: float,
    T1_bulk: float = 2.0,
    T2_bulk: float = 1.0,
) -> float:
    """T₁/T₂ ratio — a key wettability indicator.

    Large T₁/T₂ → strong surface interaction → more oil-wet.
    Small T₁/T₂ → weak surface interaction → more water-wet.
    """
    T1 = surface_relaxation_T1(rho1, S_over_V, T1_bulk)
    T2 = surface_relaxation_T2(rho2, S_over_V, T2_bulk)
    return T1 / T2


# ──────────────────────────────────────────────────────────────────────
# 2. ¹³C vs ¹H Sensitivity Comparison
# ──────────────────────────────────────────────────────────────────────
GAMMA_1H = 2.675e8       # rad/(s·T)
GAMMA_13C = 6.728e7      # rad/(s·T)
ABUNDANCE_13C = 0.011    # 1.1 %

def relative_sensitivity_13C_1H(B0: float = 9.4) -> float:
    """Relative MR sensitivity of ¹³C to ¹H at field B₀.

    Sensitivity ∝ γ³ × abundance × I(I+1)

    Both are spin-½ → I(I+1) cancels.  Factor:
        (γ_13C / γ_1H)^3 × (abundance_13C / 1.0)

    Parameters
    ----------
    B0 : float
        Static magnetic field (T).  Not used in ratio but noted
        because higher B₀ improves absolute ¹³C sensitivity.

    Returns
    -------
    float
        Relative sensitivity (dimensionless).
    """
    return (GAMMA_13C / GAMMA_1H) ** 3 * ABUNDANCE_13C


def snr_improvement_factor(n_averages: int) -> float:
    """SNR improvement from signal averaging.

    SNR ∝ √(n_averages)
    """
    return math.sqrt(n_averages)


# ──────────────────────────────────────────────────────────────────────
# 3. T₂ Distribution Analysis (multi-exponential)
# ──────────────────────────────────────────────────────────────────────
def multi_exponential_decay(
    t: np.ndarray,
    amplitudes: np.ndarray,
    T2_values: np.ndarray,
) -> np.ndarray:
    """Simulate CPMG echo-train decay with multiple T₂ components.

    S(t) = Σᵢ Aᵢ exp(-t / T₂ᵢ)

    Parameters
    ----------
    t : array_like
        Echo times (ms).
    amplitudes : array_like
        Signal amplitudes for each component.
    T2_values : array_like
        T₂ relaxation times for each component (ms).
    """
    t = np.asarray(t, float)[:, np.newaxis]
    A = np.asarray(amplitudes, float)
    T2 = np.asarray(T2_values, float)
    return np.sum(A * np.exp(-t / T2), axis=1)


def log_mean_T2(amplitudes: np.ndarray, T2_values: np.ndarray) -> float:
    """Logarithmic mean T₂.

    T₂_LM = exp( Σ Aᵢ ln(T₂ᵢ) / Σ Aᵢ )
    """
    A = np.asarray(amplitudes, float)
    T2 = np.asarray(T2_values, float)
    return float(np.exp(np.sum(A * np.log(T2)) / np.sum(A)))


# ──────────────────────────────────────────────────────────────────────
# 4. Wettability Index from ¹³C T₁/T₂
# ──────────────────────────────────────────────────────────────────────
@dataclass
class WettabilityMR:
    """Wettability assessment from ¹³C MR measurements.

    Attributes
    ----------
    sample_id : str
    T2_logmean_13C : float   (ms)
    T1_T2_ratio_13C : float
    wettability_class : str
    """
    sample_id: str
    T2_logmean_13C: float
    T1_T2_ratio_13C: float
    wettability_class: str = ""

    def classify(self,
                 water_wet_threshold: float = 3.0,
                 oil_wet_threshold: float = 10.0) -> str:
        """Classify wettability based on T₁/T₂ ratio.

        Low T₁/T₂  → water-wet (oil far from surfaces).
        High T₁/T₂ → oil-wet (oil adsorbed on surfaces).
        """
        r = self.T1_T2_ratio_13C
        if r < water_wet_threshold:
            self.wettability_class = "water-wet"
        elif r > oil_wet_threshold:
            self.wettability_class = "oil-wet"
        else:
            self.wettability_class = "mixed-wet"
        return self.wettability_class


# ──────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"¹³C / ¹H relative sensitivity: {relative_sensitivity_13C_1H():.2e}")
    print(f"  → need ~{1/relative_sensitivity_13C_1H()**2:.0f}× more averages "
          f"for equal SNR")

    # Simulate T₂ distributions for WW vs OW plugs
    t = np.linspace(0.1, 500, 200)  # ms

    # Water-wet: oil T₂ long (far from surface)
    S_ww = multi_exponential_decay(t, [0.7, 0.3], [200, 50])
    T2_lm_ww = log_mean_T2(np.array([0.7, 0.3]), np.array([200, 50]))

    # Oil-wet: oil T₂ short (adsorbed on surface)
    S_ow = multi_exponential_decay(t, [0.5, 0.5], [30, 8])
    T2_lm_ow = log_mean_T2(np.array([0.5, 0.5]), np.array([30, 8]))

    print(f"\nWater-wet plug: T₂_LM = {T2_lm_ww:.1f} ms")
    print(f"Oil-wet plug:   T₂_LM = {T2_lm_ow:.1f} ms")

    # Classify
    ww = WettabilityMR("Plug-WW", T2_lm_ww, 2.5)
    ow = WettabilityMR("Plug-OW", T2_lm_ow, 15.0)
    print(f"\n{ww.sample_id}: T₁/T₂ = {ww.T1_T2_ratio_13C}  → {ww.classify()}")
    print(f"{ow.sample_id}: T₁/T₂ = {ow.T1_T2_ratio_13C}  → {ow.classify()}")
