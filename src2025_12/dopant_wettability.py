"""
Pore-Scale Investigation of Dopant Impact on Wettability Alteration
====================================================================

Implements the ideas of:

    Nono, F., Faisal, T.F., Pairoys, F., Regaieg, M., and Caubit, C., 2025,
    "Pore-Scale Investigation of Dopant Impact on Wettability Alteration",
    Petrophysics, 66(6), 1032–1042.
    DOI: 10.30632/PJV66N6-2025a8

Key ideas
---------
* Comparison of dopant-free vs. doped brine wettability restoration
  protocols for DRP studies.
* Pore-scale trapping analysis: residual oil (trapped) vs. connected
  oil after imbibition.
* Quantification of how dopants (NaI, KI) enhance water-wetness and
  alter trapping behaviour relative to undoped brines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. Pore-Occupancy Analysis
# ──────────────────────────────────────────────────────────────────────
@dataclass
class PoreOccupancy:
    """Pore-scale fluid distribution snapshot from µCT segmentation.

    Attributes
    ----------
    n_pores : int
        Total number of identified pores.
    oil_filled : np.ndarray
        Boolean mask — True where the pore is oil-filled.
    pore_volumes : np.ndarray
        Volume of each pore (voxel units or physical).
    """
    n_pores: int
    oil_filled: np.ndarray
    pore_volumes: np.ndarray

    @property
    def oil_saturation(self) -> float:
        """Bulk oil saturation (fraction of pore volume)."""
        vol = np.asarray(self.pore_volumes, float)
        return float(vol[self.oil_filled].sum() / vol.sum())

    @property
    def water_saturation(self) -> float:
        return 1.0 - self.oil_saturation


def trapped_oil_fraction(
    before_imbibition: PoreOccupancy,
    after_imbibition: PoreOccupancy,
) -> float:
    """Fraction of oil that became trapped (disconnected) after imbibition.

    Trapped oil = oil remaining after imbibition that was connected before.
    """
    vol = np.asarray(before_imbibition.pore_volumes, float)
    oil_before = before_imbibition.oil_filled
    oil_after = after_imbibition.oil_filled

    trapped_vol = vol[oil_after].sum()
    initial_oil_vol = vol[oil_before].sum()
    if initial_oil_vol <= 0:
        return 0.0
    return float(trapped_vol / initial_oil_vol)


def recovery_efficiency(
    before_imbibition: PoreOccupancy,
    after_imbibition: PoreOccupancy,
) -> float:
    """Oil recovery efficiency from imbibition (fraction).

    RE = 1 − So_after / So_before
    """
    So_before = before_imbibition.oil_saturation
    So_after = after_imbibition.oil_saturation
    if So_before <= 0:
        return 0.0
    return 1.0 - So_after / So_before


# ──────────────────────────────────────────────────────────────────────
# 2. Dopant-Free vs. Doped Protocol Comparison
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ProtocolResult:
    """Summary of a wettability-restoration protocol result.

    Attributes
    ----------
    protocol_name : str
        E.g. "Dopant-free", "NaI-doped".
    Swi : float
        Initial water saturation after primary drainage.
    Sor : float
        Residual oil saturation after imbibition.
    trapped_fraction : float
        Fraction of initial oil that is trapped.
    amott_wettability_index : float
        Amott wettability index (-1 oil-wet to +1 water-wet).
    """
    protocol_name: str
    Swi: float
    Sor: float
    trapped_fraction: float
    amott_wettability_index: float


def amott_wettability_index(
    Swi: float,
    Sos: float,
    Sw_spont_imb: float,
    So_spont_drain: float,
) -> float:
    """Amott (Harvey) wettability index.

    Iw = δ_w − δ_o

    where:
        δ_w = (Sw_spont_imb − Swi) / (1 − Sor − Swi)
        δ_o = (So_spont_drain − Sor) / (1 − Sor − Swi)

    Parameters
    ----------
    Swi : float
        Initial water saturation.
    Sos : float
        Residual oil saturation.
    Sw_spont_imb : float
        Water saturation reached by spontaneous imbibition.
    So_spont_drain : float
        Oil saturation reached by spontaneous drainage (from 1-Sos).

    Returns
    -------
    float
        Amott index Iw ∈ [-1, 1].
    """
    denom = 1.0 - Sos - Swi
    if denom <= 0:
        return 0.0
    delta_w = (Sw_spont_imb - Swi) / denom
    delta_o = (So_spont_drain - Sos) / denom
    return delta_w - delta_o


def compare_protocols(results: List[ProtocolResult]) -> None:
    """Print a comparison table of protocol results."""
    header = (f"{'Protocol':<20s} {'Swi':>6s} {'Sor':>6s} "
              f"{'Trapped':>8s} {'Iw':>6s}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.protocol_name:<20s} {r.Swi:6.3f} {r.Sor:6.3f} "
              f"{r.trapped_fraction:8.3f} {r.amott_wettability_index:6.3f}")


# ──────────────────────────────────────────────────────────────────────
# 3. Wettability Alteration Quantification
# ──────────────────────────────────────────────────────────────────────
def wettability_shift(Iw_undoped: float, Iw_doped: float) -> float:
    """Change in wettability index due to dopant.

    Positive shift → dopant makes system more water-wet.
    """
    return Iw_doped - Iw_undoped


def dopant_bias_summary(
    results: List[ProtocolResult],
    baseline_name: str = "Dopant-free",
) -> Dict[str, float]:
    """Return dopant-induced Iw shifts relative to the baseline."""
    baseline = next(r for r in results if r.protocol_name == baseline_name)
    return {r.protocol_name: wettability_shift(baseline.amott_wettability_index,
                                                r.amott_wettability_index)
            for r in results if r.protocol_name != baseline_name}


# ──────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic comparison (inspired by the paper's observations)
    protocols = [
        ProtocolResult("Dopant-free", 0.20, 0.28, 0.35, 0.10),
        ProtocolResult("NaI-doped",   0.20, 0.22, 0.45, 0.30),
        ProtocolResult("KI-doped",    0.20, 0.23, 0.42, 0.25),
    ]
    compare_protocols(protocols)
    shifts = dopant_bias_summary(protocols)
    print("\nDopant-induced Iw shifts vs. dopant-free:")
    for name, s in shifts.items():
        print(f"  {name}: ΔIw = {s:+.3f}")
