"""
Article 7: Investigation of Salt-Bearing Sediments Through Digital Rock
           Technology Together With Experimental Core Analysis
Rydzy, Anger, Hertel, Dietderich, Patino, Appel (2018)
DOI: 10.30632/petro_059_1_a6

Halite-cemented sandstones are studied by combining micro-CT imaging with
single-phase lattice-Boltzmann flow simulation.  Digitally removing the salt
recovers the "paleo" pore system: the resolved porosity is the free pore space,
the paleoporosity adds the salt-filled space, the salt saturation classifies how
plugged the rock is, and the (paleo)permeability declines roughly exponentially
with salt saturation.  Pore-throat and grain sizes are summarized by their
percentiles.

Implements:

  - Resolved porosity, paleoporosity, and salt saturation
  - Salt-saturation classification (full / high / low / none)
  - Exponential permeability decline with salt saturation
  - Size-distribution percentiles (D10/D50/D90)

Note: this article's method is image-analysis and lattice-Boltzmann simulation
rather than closed-form equations, so the relations below are the standard
porosity/saturation definitions and the reported exponential k-vs-salt trend the
paper describes.  Porosities/saturations fractional.
"""

import numpy as np


# ---------------------------------------------- porosity / salt --------------

def resolved_porosity(free_pore_volume, bulk_volume):
    """Resolved porosity = free (open) pore space / bulk volume."""
    return free_pore_volume / bulk_volume


def paleoporosity(free_pore_volume, salt_volume, bulk_volume):
    """Paleoporosity = (free pore + salt-filled) space / bulk volume."""
    return (free_pore_volume + salt_volume) / bulk_volume


def salt_saturation(salt_volume, total_pore_volume):
    """Salt saturation  Ssalt = salt volume / total (paleo) pore volume."""
    return salt_volume / total_pore_volume


def salt_class(ssalt):
    """Classify salt saturation: full (>0.97), high (0.85-0.97), low (>0), none."""
    if ssalt > 0.97:
        return "full"
    if ssalt >= 0.85:
        return "high"
    if ssalt > 0.0:
        return "low"
    return "none"


def permeability_vs_salt(k_clean, ssalt, decay=5.0):
    """Exponential permeability decline with salt saturation  k = k_clean*exp(-c*Ssalt)."""
    return k_clean * np.exp(-decay * np.asarray(ssalt, float))


def size_percentiles(sizes):
    """Return (D10, D50, D90) of a pore-throat or grain-size distribution."""
    s = np.asarray(sizes, float)
    return tuple(np.percentile(s, p) for p in (10, 50, 90))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Salt-Bearing Sediments (digital rock)")
    print("=" * 60)

    # Paleoporosity includes the salt-filled space, so it exceeds resolved porosity
    free, salt, bulk = 30.0, 20.0, 200.0
    phi_r = resolved_porosity(free, bulk)
    phi_p = paleoporosity(free, salt, bulk)
    print(f"  resolved / paleo phi   = {phi_r:.3f} / {phi_p:.3f}")
    assert phi_p > phi_r

    # Salt saturation and its class
    ss = salt_saturation(salt, free + salt)
    print(f"  salt saturation        = {ss:.3f} ({salt_class(ss)})")
    assert np.isclose(ss, 0.4) and salt_class(ss) == "low"
    assert salt_class(0.99) == "full" and salt_class(0.0) == "none"

    # Permeability declines exponentially as salt fills the pores
    k = permeability_vs_salt(100.0, np.array([0.0, 0.4, 0.9]))
    assert k[0] > k[1] > k[2] and np.isclose(k[0], 100.0)

    # Size percentiles are ordered
    d10, d50, d90 = size_percentiles([1, 2, 3, 5, 8, 13, 21.0])
    print(f"  D10/D50/D90            = {d10:.1f} / {d50:.1f} / {d90:.1f}")
    assert d10 < d50 < d90
    print("  PASS")
    return {"paleoporosity": float(phi_p), "salt_saturation": float(ss)}


if __name__ == "__main__":
    test_all()
