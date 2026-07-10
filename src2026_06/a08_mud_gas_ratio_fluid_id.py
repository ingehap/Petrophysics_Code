"""
An Improved Mud Gas Ratio Method for Enhanced Fluid Identification
While Drilling

Reference:
    Luo, P., Li, W., Lu, P., and Qubaisi, K. (2026). An Improved Mud Gas
    Ratio Method for Enhanced Fluid Identification While Drilling.
    Petrophysics, 67(3), 582-593.
    DOI: 10.30632/PJV67N3-2026a8

The paper improves the classic mud-gas-ratio interpretation by combining
three key gas-ratio parameters to discriminate eight distinct fluid types in
real time while drilling (automated in the authors' GeochemLog software).

This module implements the standard Haworth (1985) gas-ratio framework that
underlies the method:
    - Wetness ratio  Wh = (C2+C3+C4+C5) / (C1+C2+C3+C4+C5) * 100
    - Balance ratio  Bh = (C1+C2) / (C3+C4+C5)
    - Character ratio Ch = (C4+C5) / C3
plus an eight-class fluid-type classifier built on those three parameters,
and a normalisation / quality-control front end for the C1-C5 gas readings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------------------------------------
# 1. The three key gas-ratio parameters (Haworth ratios)
# ---------------------------------------------------------------------------

@dataclass
class GasReading:
    """Chromatographic mud-gas components (any consistent unit, e.g. ppm)."""
    c1: float  # methane
    c2: float  # ethane
    c3: float  # propane
    c4: float  # butanes (iC4 + nC4)
    c5: float  # pentanes (iC5 + nC5)

    @property
    def total(self) -> float:
        return self.c1 + self.c2 + self.c3 + self.c4 + self.c5


def wetness_ratio(g: GasReading) -> float:
    """
    Wetness ratio (Wh), in percent:

        Wh = (C2+C3+C4+C5) / (C1+C2+C3+C4+C5) * 100
    """
    return petrolib.integrity_drilling.haworth_ratios(g.c1, g.c2, g.c3, g.c4, g.c5)[0]


def balance_ratio(g: GasReading) -> float:
    """
    Balance ratio (Bh):

        Bh = (C1+C2) / (C3+C4+C5)
    """
    return petrolib.integrity_drilling.haworth_ratios(g.c1, g.c2, g.c3, g.c4, g.c5)[1]


def character_ratio(g: GasReading) -> float:
    """
    Character ratio (Ch):

        Ch = (C4+C5) / C3
    """
    return petrolib.integrity_drilling.haworth_ratios(g.c1, g.c2, g.c3, g.c4, g.c5)[2]


# ---------------------------------------------------------------------------
# 2. Eight-class fluid-type classifier
# ---------------------------------------------------------------------------

FLUID_TYPES = [
    "very dry gas",        # 0
    "dry gas",             # 1
    "wet gas",             # 2
    "gas condensate",      # 3
    "volatile oil",        # 4
    "light oil",           # 5
    "medium/black oil",    # 6
    "heavy/residual oil",  # 7
]


def classify_fluid(g: GasReading) -> str:
    """
    Classify into one of eight fluid types using the three ratios.

    The Wetness ratio sets the primary increasing-density progression, while
    the Balance and Character ratios refine the gas/condensate and oil
    sub-classes (Haworth rules: Bh >> Wh and Ch < 0.5 => gas; Bh < Wh and
    Ch > 0.5 => oil; very high Wh with low Bh => residual/heavy).
    """
    wh = wetness_ratio(g)
    bh = balance_ratio(g)
    ch = character_ratio(g)
    return petrolib.integrity_drilling.classify_fluid_haworth(wh, bh, ch, n_classes=8)


def fluid_density_index(g: GasReading) -> float:
    """
    Monotonic 0-1 fluid-density index derived from the wetness ratio
    (0 ~ driest gas, 1 ~ heavy oil), useful for continuous logging tracks.
    """
    wh = wetness_ratio(g)
    if np.isnan(wh):
        return float("nan")
    return float(min(max(wh / 50.0, 0.0), 1.0))


# ---------------------------------------------------------------------------
# 3. QC / normalisation front end
# ---------------------------------------------------------------------------

def normalize_reading(raw: Dict[str, float]) -> GasReading:
    """
    Build a GasReading from a raw component dict, summing isomers and
    clamping negatives to zero (basic QC before ratio computation).
    """
    def g(*keys):
        return sum(max(raw.get(k, 0.0), 0.0) for k in keys)
    return GasReading(
        c1=g("C1", "c1"),
        c2=g("C2", "c2"),
        c3=g("C3", "c3"),
        c4=g("iC4", "nC4", "C4", "c4"),
        c5=g("iC5", "nC5", "C5", "c5"),
    )


def classify_log(readings: List[GasReading]) -> List[str]:
    """Classify a depth series of gas readings (real-time-while-drilling)."""
    return [classify_fluid(g) for g in readings]


# ---------------------------------------------------------------------------
# 4. Convenience: full workflow example
# ---------------------------------------------------------------------------

def example_workflow():
    """Run a complete example and print key results."""
    print("=" * 64)
    print("Improved Mud Gas Ratio Method (8 fluid types)")
    print("Ref: Luo, Li, Lu & Qubaisi, Petrophysics 67(3) 2026")
    print("=" * 64)

    samples = {
        "dry gas":      GasReading(9800, 120, 40, 25, 15),
        "wet gas":      GasReading(8000, 600, 300, 40, 20),
        "condensate":   GasReading(9000, 700, 400, 500, 300),
        "light oil":    GasReading(4000, 600, 500, 400, 214),
        "heavy oil":    GasReading(2500, 1600, 1400, 1800, 2700),
    }

    print(f"\n  {'label':<12s}{'Wh%':>8s}{'Bh':>8s}{'Ch':>7s}  classified")
    for label, g in samples.items():
        wh, bh, ch = wetness_ratio(g), balance_ratio(g), character_ratio(g)
        ftype = classify_fluid(g)
        print(f"  {label:<12s}{wh:>8.1f}{bh:>8.1f}{ch:>7.2f}  {ftype}")

    print(f"\nAll {len(FLUID_TYPES)} fluid types in the scheme:")
    for i, t in enumerate(FLUID_TYPES):
        print(f"  {i}: {t}")

    # QC / normalisation front end.
    raw = {"C1": 5000, "C2": 1500, "C3": 900, "iC4": 300, "nC4": 400,
           "iC5": 200, "nC5": 200}
    g = normalize_reading(raw)
    print(f"\nNormalised reading -> {classify_fluid(g)} "
          f"(density index {fluid_density_index(g):.2f})")

    return samples


if __name__ == "__main__":
    example_workflow()
