"""
Article 1: Enigmatic Reservoir Properties Deciphered Using Petroleum System
Modeling and Reservoir Fluid Geodynamics
Pierpont, Birkeland, Cely, Yang, Chen, Achourov, Betancourt, Canas, Forsythe,
Pomerantz, Yang, Datir, Mullins (2023)
DOI: 10.30632/PJV64N1-2023a1

The paper resolves the apparent contradiction between biodegradation indices
and asphaltene content in two adjacent shallow offshore reservoirs (West /
Central Blocks).  The "trick" is a late condensate charge that mixes into
a previously biodegraded black-oil column and destabilises asphaltenes at
the moving fluid contact.

This module implements the underlying physics that lets you reproduce the
qualitative story:

  - Flory-Huggins-Zuo (FHZ) asphaltene gradient with depth (gravity +
    solubility terms).
  - Two-stage charge model: (i) biodegradation lightens the resident oil
    over geologic time, (ii) a late gas-condensate charge mixes upward
    and lowers oil asphaltene content (resident dilution + solubility
    shock).
  - Wax-appearance-temperature (WAT) check vs reservoir temperature
    (a simple paraffin saturation correlation).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ------------------------------------------------ FHZ asphaltene gradient ---

def fhz_asphaltene_ratio(h_above_m, phi_ref=0.10, T_K=290.0,
                         Va_m3_mol=2.30e-3, rho_a=1140.0, rho_o=780.0,
                         delta_a=20.5e6, delta_o=18.5e6):
    """phi_a(h) / phi_a_ref under the gravity term of the FHZ EOS (SI units).

        ln(phi_a / phi_a_ref) = -V_a * g * (rho_a - rho_o) * h_above / (R * T)

    h_above_m: height above the reference depth, positive upward (m).
    Densities in kg/m^3, V_a in m^3/mol, delta in Pa^0.5.  The solubility
    term ((delta_a - delta_o)^2 * V_a / RT) * ((1-phi_a)^2 - (1-phi_a_ref)^2)
    is iterative; for small phi_a the two square terms nearly cancel, so
    we drop it -- consistent with the linearised form usually quoted in
    the SPWLA RFG literature.
    """
    return petrolib.geochem_fluids.asphaltene.fhz_ratio(
        -np.asarray(h_above_m, float), Va_m3_mol, rho_a - rho_o, T_K)


# ------------------------------------------------ biodegradation kinetics --

def biodegradation_remaining(t_Ma, half_life_Ma=10.0):
    """Fraction of n-alkane mass remaining after t Ma of microbial degradation.

    Simple exponential decay with a depth/temperature-fixed half-life.
    """
    return 0.5 ** (t_Ma / half_life_Ma)


# ------------------------------------------------ two-stage charge model --

def mix_with_condensate(oil_props, condensate_props, f_cond):
    """Volumetric mixing of resident biodegraded oil with a condensate charge.

    Both oil and condensate are dicts holding asphaltene, n-alkane, wax,
    and density.  f_cond is the volume fraction of condensate after mixing.
    """
    out = {}
    for k in ("asphaltene", "n_alkane", "wax"):
        out[k] = (1.0 - f_cond) * oil_props[k] + f_cond * condensate_props[k]
    out["density"] = (1.0 - f_cond) * oil_props["density"] \
                     + f_cond * condensate_props["density"]
    return out


def asphaltene_destabilisation_loss(mixed_oil, solubility_threshold=0.10):
    """When solubility drops below threshold, the excess flocculates upstructure.

    Returns updated oil props (with reduced asphaltene) and the deposited
    asphaltene mass fraction.
    """
    delta = mixed_oil["asphaltene"] - solubility_threshold
    if delta <= 0.0:
        return mixed_oil, 0.0
    out = dict(mixed_oil)
    out["asphaltene"] = solubility_threshold
    return out, float(delta)


# ------------------------------------------------ WAT correlation ---------

def wax_appearance_T_C(wax_fraction, density_kg_m3):
    """Simple WAT correlation: paraffin-rich, dense oils precipitate hotter.

        WAT[C] = 5 + 800 * wax_fraction + 0.04 * (density - 700)

    Tuned so a wax-rich (5 % wax, 850 kg/m^3) crude has WAT ~ 51 C.
    """
    return 5.0 + 800.0 * wax_fraction + 0.04 * (density_kg_m3 - 700.0)


# ------------------------------------------------ tests --------------------

def test_all():
    print("=" * 60)
    print("Article 1: RFG + Petroleum-System Modeling Case Study")
    print("=" * 60)

    # Resident black oil before late charge
    resident_initial = dict(asphaltene=0.12, n_alkane=0.45,
                            wax=0.06, density=860.0)
    # Microbes degrade the n-alkanes for 20 Ma
    t_bio = 20.0
    f_remain = biodegradation_remaining(t_bio)
    resident = dict(resident_initial)
    resident["n_alkane"] *= f_remain
    print(f"  After {t_bio:.0f} Ma biodegradation: "
          f"n-alkane fraction {resident['n_alkane']:.3f} "
          f"(was {resident_initial['n_alkane']:.3f})")

    # Late gas-condensate charge - high light ends, no asphaltene, low wax
    condensate = dict(asphaltene=0.005, n_alkane=0.78, wax=0.005, density=720.0)

    # Block-by-block volumetric mixing fraction
    for block, f_c in [("West (downstructure)", 0.10),
                       ("Central (upstructure)", 0.35)]:
        mixed = mix_with_condensate(resident, condensate, f_c)
        mixed, deposit = asphaltene_destabilisation_loss(mixed,
                                                         solubility_threshold=0.085)
        wat = wax_appearance_T_C(mixed["wax"], mixed["density"])
        print(f"  {block:24s}: "
              f"f_cond={f_c:.2f}  "
              f"asph={mixed['asphaltene']:.3f}  "
              f"deposited_asph={deposit:.3f}  "
              f"WAT={wat:.1f} C")

    # FHZ gradient check: asphaltene should fall going up
    h = np.array([0.0, 20.0, 40.0, 60.0])  # m ABOVE reference depth
    ratio = fhz_asphaltene_ratio(h, phi_ref=0.10)
    print("  FHZ asphaltene ratio vs height above reference:")
    for hi, ri in zip(h, ratio):
        print(f"     h_above={hi:+5.1f} m   phi_a / phi_a_ref = {ri:.3f}")

    # Sanity checks
    assert f_remain < 1.0, "Biodegradation should reduce n-alkanes"
    mid_west = mix_with_condensate(resident, condensate, 0.10)
    mid_central = mix_with_condensate(resident, condensate, 0.35)
    assert mid_central["asphaltene"] < mid_west["asphaltene"], \
        "More-mixed Central should have lower asphaltene (the paradox)"
    assert ratio[-1] < 1.0, "FHZ ratio should decrease upward"
    print("  PASS")
    return {"f_alkane_remaining": float(f_remain),
            "asph_west": mid_west["asphaltene"],
            "asph_central": mid_central["asphaltene"]}


if __name__ == "__main__":
    test_all()
