"""
Article 5: Novel Composite Cement for Improved Well Integrity Evaluation
Elshahawi, Huang, Pollock, Veedu (2019)
DOI: 10.30632/PJV60N1Y2019a4

Acoustic cement-evaluation logs infer the material behind casing from its
acoustic impedance (Z = rho*v): gas, liquid (mud) and set cement occupy distinct
impedance ranges, and the reflection coefficient at the casing-annulus interface
drives the pulse-echo and bond-log response.  A novel composite cement is
engineered with an impedance contrast that improves the evaluation.

Implements:

  - Acoustic impedance  Z = rho*v
  - Reflection coefficient  R = (Z2 - Z1)/(Z2 + Z1)
  - Annulus material classification by impedance (gas / liquid / cement)
  - Cement bond index from attenuation

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the acoustic-impedance / bond-evaluation relations the paper applies.
Impedance in Mrayl.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- impedance ---------------

def acoustic_impedance(rho, v):
    """Acoustic impedance  Z = rho*v  (Mrayl, with rho in g/cc and v in m/s).

    rho(g/cc)*1000 -> kg/m^3, times v -> rayl, /1e6 -> Mrayl, i.e. rho*v/1000.
    """
    return petrolib.integrity_drilling.acoustic_impedance(rho, v, rho_unit="g/cc")


def reflection_coefficient(Z1, Z2):
    """Reflection coefficient  R = (Z2 - Z1)/(Z2 + Z1)."""
    return petrolib.acoustic_geomech.reflection_coefficient(Z1, Z2)


def classify_annulus(Z, gas_max=0.5, liquid_max=2.6):
    """Classify the annulus material by acoustic impedance (Mrayl).

      Z < gas_max     -> gas
      Z < liquid_max  -> liquid (mud/water)
      otherwise       -> set cement
    """
    return petrolib.integrity_drilling.classify_annulus(
        Z, gas_max=gas_max, liquid_max=liquid_max, cement_min=liquid_max)


def bond_index(attenuation, atten_free_pipe, atten_well_bonded):
    """Cement bond index from attenuation between free-pipe and well-bonded ends.

        BI = (atten - atten_free_pipe)/(atten_well_bonded - atten_free_pipe)
    """
    return petrolib.integrity_drilling.bond_index(
        attenuation, atten_free_pipe, atten_well_bonded, input_kind="attenuation")


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Composite Cement & Well Integrity")
    print("=" * 60)

    # Impedance ordering: gas << liquid < cement
    Z_gas = acoustic_impedance(0.2, 1000.0)
    Z_liq = acoustic_impedance(1.1, 1500.0)
    Z_cem = acoustic_impedance(1.9, 2800.0)
    print(f"  Z gas/liquid/cement    = {Z_gas:.2f} / {Z_liq:.2f} / {Z_cem:.2f} Mrayl")
    assert Z_gas < Z_liq < Z_cem

    # Annulus classification by impedance
    assert classify_annulus(Z_gas) == "gas"
    assert classify_annulus(Z_liq) == "liquid"
    assert classify_annulus(Z_cem) == "cement"

    # Reflection coefficient: casing(steel)->cement is smaller than casing->gas
    Z_steel = acoustic_impedance(7.85, 5900.0)
    R_cem = abs(reflection_coefficient(Z_steel, Z_cem))
    R_gas = abs(reflection_coefficient(Z_steel, Z_gas))
    print(f"  |R| casing->cement/gas = {R_cem:.3f} / {R_gas:.3f}")
    assert R_gas > R_cem                          # gas behind pipe -> strong echo

    # Bond index rises from free pipe (0) to well bonded (1)
    assert abs(bond_index(1.0, 1.0, 10.0)) < 1e-9
    assert abs(bond_index(10.0, 1.0, 10.0) - 1.0) < 1e-9
    assert abs(bond_index(5.5, 1.0, 10.0) - 0.5) < 1e-9
    print("  PASS")
    return {"Z_cement": float(Z_cem), "R_gas": float(R_gas)}


if __name__ == "__main__":
    test_all()
