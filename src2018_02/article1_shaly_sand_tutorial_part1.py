"""
Article 1 (Tutorial): What Is It About Shaly Sands? Shaly Sand Tutorial 1 of 3
Thomas (2018)
DOI: 10.30632/petro_059_1_t1

The first shaly-sand tutorial sets the Archie baseline and its two validity
conditions - the rock is homogeneous and the only electrical conductor is the
brine in the pore network - then motivates shaly-sand corrections as departures
from the second condition: clay minerals (the "Big Four": kaolinite, smectite,
illite, chlorite) add conductivity and have specific surface areas 10^5-10^6
times that of quartz.  A "shaly sand" is defined as a siliciclastic with clay
between 5% and 50% of grain volume.

Implements:

  - Archie water saturation  Sw = (a*Rw/(phi^m*Rt))^(1/n)
  - Archie formation factor  F = a/phi^m
  - Shaly-sand classification (5% < clay < 50% of grain volume)
  - Clay specific-surface ratio relative to quartz

Note: this installment is conceptual (no closed-form shaly-sand equations yet),
so the relations below capture the Archie baseline the tutorial builds on; the
DOI is the authoritative SPWLA/CrossRef value for this issue (the older
`petro_059_1_*` scheme).  Fractions and consistent resistivity units.
"""

import numpy as np


# ---------------------------------------------- Archie --------------

def formation_factor(phi, a=1.0, m=2.0):
    """Archie formation factor  F = a/phi^m."""
    return a / np.asarray(phi, float) ** m


def archie_water_saturation(rw, rt, phi, a=1.0, m=2.0, n=2.0):
    """Archie water saturation  Sw = (a*Rw/(phi^m*Rt))^(1/n).

    Valid when the rock is homogeneous and brine is the only conductor - the two
    conditions the tutorial says clay violates.
    """
    sw = (a * rw / (np.asarray(phi, float) ** m * rt)) ** (1.0 / n)
    return np.clip(sw, 0.0, 1.0)


# ---------------------------------------------- shaly-sand --------------

def is_shaly_sand(clay_volume_fraction):
    """True if the clay fraction of grain volume is between 5% and 50%."""
    c = np.asarray(clay_volume_fraction, float)
    return (c > 0.05) & (c < 0.50)


def specific_surface_ratio(clay_surface, quartz_surface):
    """Clay-to-quartz specific-surface ratio (the tutorial cites 1e5-1e6)."""
    return clay_surface / quartz_surface


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1 (Tutorial): Shaly Sand 1 of 3")
    print("=" * 60)

    # Formation factor rises as porosity falls
    assert formation_factor(0.10) > formation_factor(0.30)

    # Archie: lower true resistivity -> higher water saturation
    sw_wet = archie_water_saturation(0.05, 2.0, 0.25)
    sw_dry = archie_water_saturation(0.05, 20.0, 0.25)
    print(f"  Sw (Rt 2 / 20)         = {sw_wet:.3f} / {sw_dry:.3f}")
    assert sw_wet > sw_dry and 0.0 <= sw_dry <= 1.0

    # Shaly-sand window: 3% clean, 25% shaly, 60% shale
    flags = is_shaly_sand([0.03, 0.25, 0.60])
    print(f"  shaly-sand 3/25/60%    = {flags.tolist()}")
    assert flags.tolist() == [False, True, False]

    # Clay has vastly larger specific surface than quartz
    assert specific_surface_ratio(1e6, 1.0) >= 1e5
    print("  PASS")
    return {"Sw_wet": float(sw_wet), "F": float(formation_factor(0.25))}


if __name__ == "__main__":
    test_all()
