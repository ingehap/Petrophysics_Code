"""
Wettability-Based Pore Partitioning and Its Effects on Oil Recovery
and Formation Damage in Unconventional Reservoirs

Reference:
    Aljishi, M.K., Chitrala, Y., Dang, S.T., and Rai, C. (2026).
    Wettability-Based Pore Partitioning and Its Effects on Oil Recovery
    and Formation Damage in Unconventional Reservoirs.
    Petrophysics, 67(2), 263–279. DOI: 10.30632/PJV67N2-2026a2

Implements:
  - NMR T₂-based wettability index (Looyestijn & Hofman 2006, Eq. 1)
  - Pore partitioning into water-wet / oil-wet / mixed-wet fractions
  - Dual-fluid displacement simulation (Sequences A–D)
  - Oil-saturation choking threshold estimation
  - Mineralogy-based wettability classification
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Wettability index (NMR-based, Looyestijn & Hofman 2006)
# ---------------------------------------------------------------------------

def wettability_index(nmr_Sw: float, nmr_Sdo: float) -> float:
    """
    Compute the NMR-based wettability index (Eq. 1 of paper).

    Parameters
    ----------
    nmr_Sw  : Pore-volume fraction occupied by brine after Sequence A
              (brine-only forced imbibition to max pressure).
    nmr_Sdo : Pore-volume fraction occupied by dodecane (oil proxy) after
              Sequence B (oil-only forced imbibition to max pressure).

    Returns
    -------
    Iw : Wettability index in [-1, +1]
         +1 → strongly water-wet
         -1 → strongly oil-wet
          0 → mixed-wet

    Notes
    -----
    Iw = NMR(Sw) - NMR(Sdo)   (normalised pore-volume fractions)
    Reference: Looyestijn, W.J. and Hofman, J., 2006, SPE-93624.
    """
    return nmr_Sw - nmr_Sdo


def classify_wettability(Iw: float) -> str:
    """Human-readable classification of the wettability index."""
    if Iw > 0.3:
        return "Water-wet"
    elif Iw < -0.3:
        return "Oil-wet"
    else:
        return "Mixed-wet"


# ---------------------------------------------------------------------------
# 2. Pore partitioning fractions
# ---------------------------------------------------------------------------

@dataclass
class PorePartition:
    """
    Volumetric fractions of water-wet, oil-wet, and mixed-wet pores.
    Fractions should sum to approximately 1.0.

    Attributes
    ----------
    f_ww  : water-wet pore fraction
    f_ow  : oil-wet pore fraction
    f_mw  : mixed-wet pore fraction
    """
    f_ww: float   # water-wet
    f_ow: float   # oil-wet
    f_mw: float   # mixed-wet

    def total(self) -> float:
        return self.f_ww + self.f_ow + self.f_mw

    def is_valid(self, tol: float = 0.05) -> bool:
        return abs(self.total() - 1.0) < tol


def compute_pore_partition(phi_brine_seqA: float,
                            phi_oil_seqB:   float,
                            phi_total:      float) -> PorePartition:
    """
    Estimate water-wet, oil-wet, and mixed-wet pore fractions from
    single-fluid NMR saturations (methodology from Mukherjee et al. 2020,
    extended in this paper).

    Parameters
    ----------
    phi_brine_seqA : Connected porosity fraction filled by brine alone
                     (end of Sequence A, forced imbibition).
    phi_oil_seqB   : Connected porosity fraction filled by oil alone
                     (end of Sequence B, forced imbibition).
    phi_total      : Total porosity fraction (NMR + helium pycnometry).

    Returns
    -------
    PorePartition
    """
    # Pores that accept only brine → water-wet
    # Pores that accept only oil   → oil-wet
    # Pores that accept both       → mixed-wet
    f_ww  = max(phi_brine_seqA - phi_oil_seqB, 0.0)
    f_ow  = max(phi_oil_seqB   - phi_brine_seqA, 0.0)
    f_mw  = min(phi_brine_seqA, phi_oil_seqB)
    total = f_ww + f_ow + f_mw
    # Normalise to total porosity
    if total > 0 and phi_total > 0:
        scale = phi_total / total
        f_ww  *= scale
        f_ow  *= scale
        f_mw  *= scale
    return PorePartition(f_ww=f_ww, f_ow=f_ow, f_mw=f_mw)


# ---------------------------------------------------------------------------
# 3. Dual-fluid displacement (Sequences C and D)
# ---------------------------------------------------------------------------

@dataclass
class DisplacementState:
    """
    Tracks fluid volumes during a dual-fluid displacement sequence.

    Attributes
    ----------
    phi_brine : Brine-filled porosity fraction
    phi_oil   : Oil-filled porosity fraction
    phi_iso   : Isolated (unconnected) porosity fraction
    pressure  : Current imbibition pressure, psi
    """
    phi_brine: float = 0.0
    phi_oil:   float = 0.0
    phi_iso:   float = 0.0
    pressure:  float = 0.0


def simulate_sequence_C(phi_brine_spont: float,
                         phi_oil_spont_counter: float,
                         phi_oil_forced: float,
                         phi_iso: float,
                         pressure_steps: np.ndarray) -> list:
    """
    Sequence C: Brine imbibition → dodecane counter-imbibition + forced.

    Mimics the observation that brine resists displacement by oil;
    oil uptake occurs mainly between 250–1,000 psi (Fig. 6).

    Parameters
    ----------
    phi_brine_spont        : Brine pore fraction after spontaneous imbibition.
    phi_oil_spont_counter  : Oil fraction entering spontaneously during
                             counter-imbibition (small).
    phi_oil_forced         : Final oil fraction after forced imbibition.
    phi_iso                : Isolated (unconnected) pore fraction.
    pressure_steps         : Array of pressure values (psi) for forced stage.

    Returns
    -------
    List[DisplacementState] at each pressure step.
    """
    states = []

    # Step a→b: spontaneous brine imbibition
    states.append(DisplacementState(phi_brine=phi_brine_spont,
                                    phi_oil=0.0,
                                    phi_iso=phi_iso,
                                    pressure=0.0))

    # Step b→c: spontaneous dodecane counter-imbibition
    phi_brine_after_spont = phi_brine_spont - phi_oil_spont_counter
    states.append(DisplacementState(phi_brine=phi_brine_after_spont,
                                    phi_oil=phi_oil_spont_counter,
                                    phi_iso=phi_iso,
                                    pressure=0.0))

    # Step c→d: forced dodecane imbibition (main oil entry 250–1000 psi)
    oil_forced_range = phi_oil_forced - phi_oil_spont_counter
    for p in pressure_steps:
        # Sigmoid-shaped oil entry, concentrated between 250–1000 psi
        frac_complete = _sigmoid_entry(p, p_low=250.0, p_high=1000.0)
        phi_oil_now  = phi_oil_spont_counter + frac_complete * oil_forced_range
        phi_brine_now = phi_brine_after_spont - (phi_oil_now - phi_oil_spont_counter)
        states.append(DisplacementState(phi_brine=max(phi_brine_now, 0.0),
                                        phi_oil=phi_oil_now,
                                        phi_iso=phi_iso,
                                        pressure=p))
    return states


def simulate_sequence_D(phi_oil_spont: float,
                         phi_brine_spont_counter: float,
                         phi_brine_forced: float,
                         phi_iso: float,
                         pressure_steps: np.ndarray) -> list:
    """
    Sequence D: Oil imbibition → brine counter-imbibition + forced.

    Brine displaces a significant fraction of oil spontaneously (Fig. 8).

    Returns
    -------
    List[DisplacementState] at each pressure step.
    """
    states = []

    # Step a→b: spontaneous oil imbibition fills connected pores
    states.append(DisplacementState(phi_brine=0.0,
                                    phi_oil=phi_oil_spont,
                                    phi_iso=phi_iso,
                                    pressure=0.0))

    # Step b→c: spontaneous brine counter-imbibition (major displacement)
    phi_oil_after_spont  = phi_oil_spont - phi_brine_spont_counter
    states.append(DisplacementState(phi_brine=phi_brine_spont_counter,
                                    phi_oil=phi_oil_after_spont,
                                    phi_iso=phi_iso,
                                    pressure=0.0))

    # Step c→d: forced brine imbibition (limited additional displacement)
    brine_forced_increment = phi_brine_forced - phi_brine_spont_counter
    for p in pressure_steps:
        frac_complete = _sigmoid_entry(p, p_low=0.0, p_high=500.0)
        phi_brine_now = phi_brine_spont_counter + frac_complete * brine_forced_increment
        phi_oil_now   = phi_oil_spont - phi_brine_now
        states.append(DisplacementState(phi_brine=phi_brine_now,
                                        phi_oil=max(phi_oil_now, 0.0),
                                        phi_iso=phi_iso,
                                        pressure=p))
    return states


def _sigmoid_entry(p: float, p_low: float, p_high: float) -> float:
    """Smoothly ramp from 0 to 1 between p_low and p_high."""
    if p <= p_low:
        return 0.0
    if p >= p_high:
        return 1.0
    x = (p - p_low) / (p_high - p_low)
    return 3 * x**2 - 2 * x**3  # smoothstep


# ---------------------------------------------------------------------------
# 4. Oil-saturation choking threshold
# ---------------------------------------------------------------------------

def choking_threshold(phi_oil_seqC_final: float,
                       phi_total: float) -> float:
    """
    Estimate the oil-saturation choking threshold: the residual oil-filled
    porosity fraction at which no further oil is displaced by brine forced
    imbibition (Sequence C result).

    The paper reports a range of 26–71 % of pore volume across samples.

    Parameters
    ----------
    phi_oil_seqC_final : Oil pore fraction remaining at end of Sequence C.
    phi_total          : Total porosity fraction.

    Returns
    -------
    threshold : Choking threshold as a percent of pore volume (%).
    """
    if phi_total <= 0:
        return float("nan")
    return (phi_oil_seqC_final / phi_total) * 100.0


# ---------------------------------------------------------------------------
# 5. Mineralogy-based wettability tendency
# ---------------------------------------------------------------------------

def mineralogy_wettability_tendency(quartz_wt: float,
                                     carbonate_wt: float,
                                     clay_wt: float,
                                     toc_wt: float) -> str:
    """
    Qualitative wettability tendency based on mineralogical composition.

    Key findings from the paper:
      - Silica- and clay-rich samples → predominantly mixed-wet
      - Carbonate-rich samples → higher oil-wet fraction
      - High TOC → oil-wet organics

    Parameters
    ----------
    quartz_wt    : Quartz + feldspar weight fraction (0–1)
    carbonate_wt : Carbonate weight fraction (0–1)
    clay_wt      : Clay weight fraction (0–1)
    toc_wt       : Total organic carbon weight fraction (0–1)

    Returns
    -------
    tendency : 'Oil-wet tendency', 'Mixed-wet', or 'Water-wet tendency'
    """
    # Score: positive = oil-wet tendency
    oil_wet_score = 2.0 * carbonate_wt + 3.0 * toc_wt - 1.5 * quartz_wt
    if oil_wet_score > 0.5:
        return "Oil-wet tendency"
    elif oil_wet_score < -0.3:
        return "Water-wet tendency"
    else:
        return "Mixed-wet"


# ---------------------------------------------------------------------------
# 6. Production strategy recommendation
# ---------------------------------------------------------------------------

def recommend_production_strategy(partition: PorePartition) -> str:
    """
    Based on pore wettability partition, recommend a production strategy
    consistent with the paper's conclusions.

    Parameters
    ----------
    partition : PorePartition from compute_pore_partition()

    Returns
    -------
    str : Production strategy recommendation
    """
    if partition.f_mw >= 0.5:
        return ("Mixed-wet zone: Use moderate drawdown pressure. "
                "Brine-based choking prior to flowback is recommended "
                "to displace oil from mixed-wet pores.")
    elif partition.f_ow >= 0.5:
        return ("Oil-wet zone: Higher drawdown or wettability-altering "
                "agents (surfactants, CO₂) are needed to mobilise trapped "
                "hydrocarbons. Brine injection alone is insufficient.")
    else:
        return ("Water-wet zone: Standard waterflooding is effective. "
                "Monitor for water blockage in tight pores.")


# ---------------------------------------------------------------------------
# 7. Example workflow
# ---------------------------------------------------------------------------

def example_workflow():
    print("=" * 60)
    print("NMR Wettability Pore Partitioning")
    print("Ref: Aljishi et al., Petrophysics 67(2) 2026")
    print("=" * 60)

    # Illustrative values mimicking Sample 1 (mixed-wet, slight oil-wet bias)
    phi_total      = 0.082   # 8.2% total porosity
    phi_brine_A    = 0.076   # brine fills ~92% of connected pores in Seq A
    phi_oil_B      = 0.080   # oil fills ~98% of connected pores in Seq B

    Iw = wettability_index(phi_brine_A / phi_total,
                             phi_oil_B   / phi_total)
    print(f"\nWettability index Iw = {Iw:+.3f}  → {classify_wettability(Iw)}")

    partition = compute_pore_partition(phi_brine_A, phi_oil_B, phi_total)
    print(f"\nPore partitioning (fraction of total porosity):")
    print(f"  Water-wet  : {partition.f_ww:.3f}")
    print(f"  Mixed-wet  : {partition.f_mw:.3f}")
    print(f"  Oil-wet    : {partition.f_ow:.3f}")

    pressures = np.array([0, 50, 100, 250, 500, 1000, 2000, 4500])

    seqC = simulate_sequence_C(
        phi_brine_spont=phi_brine_A,
        phi_oil_spont_counter=0.002,
        phi_oil_forced=0.018,
        phi_iso=0.006,
        pressure_steps=pressures,
    )
    choking = choking_threshold(seqC[-1].phi_oil, phi_total)
    print(f"\nSequence C – oil saturation choking threshold: {choking:.1f}%")

    seqD = simulate_sequence_D(
        phi_oil_spont=phi_oil_B,
        phi_brine_spont_counter=0.80 * phi_oil_B,
        phi_brine_forced=0.82 * phi_oil_B,
        phi_iso=0.006,
        pressure_steps=pressures,
    )
    residual_oil = seqD[-1].phi_oil / phi_total * 100.0
    print(f"Sequence D – residual oil after brine: {residual_oil:.1f}% PV")

    print(f"\nMineralogy test (carbonate-rich sample):")
    tendency = mineralogy_wettability_tendency(quartz_wt=0.12,
                                               carbonate_wt=0.75,
                                               clay_wt=0.05,
                                               toc_wt=0.02)
    print(f"  → {tendency}")

    print(f"\nProduction recommendation: {recommend_production_strategy(partition)}")
    return partition, seqC, seqD


if __name__ == "__main__":
    example_workflow()
