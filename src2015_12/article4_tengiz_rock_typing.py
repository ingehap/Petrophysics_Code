"""
Article 4: Petrophysical Challenges in Giant Carbonate Tengiz Field, Republic of
           Kazakhstan
Skalinski, Se, Playton, Theologou, Narr, Sullivan, Mallan (2015)
Reference: Petrophysics Vol. 56, No. 6 (December 2015), pp. 615-647
DOI: none assigned (this issue predates SPWLA DOI assignment)

Tengiz is a giant isolated carbonate platform with multimodal pore systems,
bitumen, and fractures.  Pore type is the primary control on petrophysical rock
type; permeability is tied to porosity by rock type, and water saturation is
modeled with pore-type-based saturation-height functions (SHF) combined with a
rock-type-based bulk-volume-water (BVW) approach.  This module implements the
flow-zone-indicator rock-typing framework and the saturation-height / BVW
modeling the paper applies.

Implements:

  - Reservoir quality index  RQI = 0.0314*sqrt(k/phi)
  - Normalized porosity index  phi_z = phi/(1 - phi)
  - Flow zone indicator  FZI = RQI/phi_z  and permeability from FZI
  - Rock-type permeability-porosity transform  log10(k) = a*phi + b
  - Saturation-height function  Sw = a*H^(-b)  and bulk volume water  BVW = phi*Sw

Note: this is a carbonate rock-typing case study; the relations below are the
standard tools it relies on (Amaefule FZI/RQI, saturation-height functions,
bulk volume water).  The typeset glyphs were dropped in extraction, so they are
standard-form reconstructions.  Permeability in mD, porosity/saturation as
fractions, height above free-water level in m or ft consistently.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- rock typing --------------

def rqi(k, phi):
    """Reservoir quality index  RQI = 0.0314*sqrt(k/phi)  [um], k in mD."""
    return petrolib.flow_transport.rqi(k, phi)


def normalized_porosity(phi):
    """Normalized porosity index  phi_z = phi/(1 - phi)  (pore-to-grain volume)."""
    return petrolib.flow_transport.phi_z(phi)


def fzi(k, phi):
    """Flow zone indicator  FZI = RQI/phi_z,

    grouping samples with similar pore geometry into a hydraulic (flow) unit.
    """
    return petrolib.flow_transport.fzi(k, phi)


def permeability_from_fzi(phi, fzi_value):
    """Permeability from FZI and porosity (inverting RQI/FZI)

        k = phi * (FZI*phi_z/0.0314)^2,   phi_z = phi/(1-phi).
    """
    return petrolib.flow_transport.k_from_fzi(phi, fzi_value)


def rocktype_permeability(phi, a, b):
    """Rock-type permeability-porosity transform  log10(k) = a*phi + b."""
    return 10.0 ** (a * np.asarray(phi, float) + b)


# ---------------------------------------------- saturation modeling --------------

def saturation_height_function(height, a, b, swirr=0.0):
    """Pore-type saturation-height function  Sw = Swirr + a*H^(-b),

    decreasing with height H above the free-water level toward Swirr.
    """
    return swirr + a * np.asarray(height, float) ** (-b)


def bulk_volume_water(phi, sw):
    """Bulk volume water  BVW = phi*Sw  (approximately constant per rock type at
    irreducible saturation)."""
    return petrolib.saturation_resistivity.bulk_volume_water(phi, sw)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Tengiz Carbonate Rock Typing")
    print("=" * 60)

    # RQI / FZI rock typing: samples on one hydraulic unit share FZI
    k, phi = 10.0, 0.15
    print(f"  RQI / FZI              = {rqi(k, phi):.4f} / {fzi(k, phi):.4f}")
    assert rqi(k, phi) > 0 and fzi(k, phi) > 0

    # Permeability from FZI round-trips back to the input permeability
    f = fzi(k, phi)
    k_back = permeability_from_fzi(phi, f)
    assert np.isclose(k_back, k)

    # Higher FZI (better pore geometry) gives higher permeability at fixed porosity
    assert permeability_from_fzi(phi, 2.0) > permeability_from_fzi(phi, 1.0)

    # Rock-type transform increases permeability with porosity
    assert rocktype_permeability(0.25, a=20.0, b=-3.0) > rocktype_permeability(0.10, 20.0, -3.0)

    # Saturation-height function decreases with height toward Swirr
    sw_lo = saturation_height_function(10.0, a=0.5, b=0.4, swirr=0.1)
    sw_hi = saturation_height_function(200.0, a=0.5, b=0.4, swirr=0.1)
    print(f"  SHF Sw @10/200         = {sw_lo:.3f} / {sw_hi:.3f}")
    assert sw_lo > sw_hi >= 0.1

    # Bulk volume water
    assert np.isclose(bulk_volume_water(0.15, 0.3), 0.045)
    print("  PASS")
    return {"RQI": float(rqi(k, phi)), "FZI": float(f), "Sw_hi": float(sw_hi)}


if __name__ == "__main__":
    test_all()
