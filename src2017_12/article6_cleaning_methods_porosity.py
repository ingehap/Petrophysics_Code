"""
Article 6: Impact of Different Cleaning Methods on Petrophysical Measurements
Gupta, Rai, Tinni, Sondergeld (2017)
Reference: Petrophysics Vol. 58, No. 6 (December 2017), pp. 613-622
DOI: none assigned (this issue predates SPWLA DOI assignment)

Removing residual hydrocarbons/bitumen by solvent cleaning opens up pore space,
so the measured (helium) porosity, free hydrocarbon content, and surface area
all change with the cleaning method and time.  This module implements the
crushed-sample helium porosity (bulk and grain volume from weights and
densities) and the porosity gain from cleaning, plus a simple solvent-efficiency
ranking.

Implements:

  - Bulk volume  BV = W/bulk_density  and grain volume  GV = W/grain_density
  - Helium porosity  phi = (BV - GV)/BV
  - Porosity gain after cleaning (grain volume falls as bitumen is removed)
  - Solvent-efficiency ranking

Note: this issue's PDF has a text layer; the porosity equation survived
(decoded), and the relations below are transcribed/standard-form.  Cleaning is
reported to raise porosity up to ~50%, cut Rock-Eval S1 up to ~90%, and raise
BET surface area by ~450%.  Weights in g, densities in g/cm^3.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

SOLVENT_EFFICIENCY = {"toluene": 1.0, "DCM": 0.98, "chloroform": 0.95, "n-heptane": 0.6}


# ---------------------------------------------- volumes / porosity --------------

def bulk_volume(weight, bulk_density):
    """Sample bulk volume  BV = W/bulk_density."""
    return np.asarray(weight, float) / bulk_density


def grain_volume(weight, grain_density):
    """Sample grain (solid) volume  GV = W/grain_density."""
    return np.asarray(weight, float) / grain_density


def helium_porosity(bv, gv):
    """Helium porosity on a crushed sample  phi = (BV - GV)/BV  (Eq. 1)."""
    return petrolib.porosity_lithology.porosity_from_volumes(bv, gv)


def porosity_after_cleaning(weight_before, weight_after, bulk_density, grain_density):
    """Porosity from before/after weights: cleaning removes bitumen, lowering GV.

    weight_before sets the bulk volume; weight_after (after solvent removal of
    pore-filling bitumen) sets the grain volume.
    """
    bv = bulk_volume(weight_before, bulk_density)
    gv = grain_volume(weight_after, grain_density)
    return helium_porosity(bv, gv)


def rank_solvents(names):
    """Rank solvents by cleaning efficiency (most effective first)."""
    return sorted(names, key=lambda s: SOLVENT_EFFICIENCY.get(s, 0.0), reverse=True)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Cleaning Methods & Porosity")
    print("=" * 60)

    # Helium porosity from bulk and grain volumes
    bv = bulk_volume(10.0, 2.30)
    gv = grain_volume(9.0, 2.65)
    phi = helium_porosity(bv, gv)
    print(f"  helium porosity        = {phi:.3f}")
    assert 0.0 < phi < 1.0

    # Cleaning removes bitumen (lower post-clean weight -> lower GV -> higher phi)
    phi_dirty = porosity_after_cleaning(10.0, 9.6, 2.30, 2.65)
    phi_clean = porosity_after_cleaning(10.0, 9.0, 2.30, 2.65)
    print(f"  porosity dirty/clean   = {phi_dirty:.3f} / {phi_clean:.3f}")
    assert phi_clean > phi_dirty

    # Solvent-efficiency ranking: toluene/DCM/chloroform beat n-heptane
    ranked = rank_solvents(["n-heptane", "chloroform", "toluene", "DCM"])
    print(f"  solvent ranking        = {ranked}")
    assert ranked[0] == "toluene" and ranked[-1] == "n-heptane"
    print("  PASS")
    return {"phi_clean": float(phi_clean)}


if __name__ == "__main__":
    test_all()
