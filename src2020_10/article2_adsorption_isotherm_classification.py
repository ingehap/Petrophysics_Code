"""
Article 2: Classification of Adsorption Isotherm Curves for Shale Based on Pore
           Structure
Tian, Chen, Yan, Deng, He (2020)
DOI: 10.30632/PJV61N5-2020a2

The paper proposes a three-parameter classification of N2 adsorption-isotherm
curves (pore shape, pore size, sorting), giving 27 generated curve types, and
compares it with the conventional five-type IUPAC/BDDT scheme.  The paper itself
publishes no numbered equations; this module implements the standard
quantitative relations it references by name (BET surface area, the IUPAC type
classifier, pore-size and sorting classes) plus the new 3-parameter classifier.

Implements:

  - BET linearization -> monolayer volume Vm and constant C, surface area
  - IUPAC isotherm type (I-V) from curve shape and hysteresis
  - Pore-size class (micro < 2 nm, meso 2-50 nm, macro > 50 nm)
  - Sorting class from the pore-size-distribution spread
  - Three-parameter (shape x size x sorting) -> 27 curve types

Note: this issue's PDF has a text layer but the paper publishes no numbered
equations; the BET relation is the standard Brunauer-Emmett-Teller form.
The paper's anchors are reproduced: 5 IUPAC base types, 27 generated types,
N2 surface-area lower limit 0.01 m^2/g, minimum pore volume 0.0001 cm^3/g.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

NA = 6.022e23            # Avogadro
N2_CROSS_NM2 = 0.162     # N2 molecular cross-sectional area (nm^2)
V_MOLAR_STP = 22414.0    # cm^3/mol at STP

PORE_TYPES = 27          # 3 shapes x 3 size classes x 3 sortings


# ---------------------------------------------- BET ---------------------

def bet_isotherm(p_rel, vm, c):
    """Forward BET adsorbed volume at relative pressure x = P/P0.

        v = Vm*C*x / ((1-x)*(1 + (C-1)*x))
    """
    return petrolib.geochem_fluids.adsorption.bet_isotherm(p_rel, vm, c)


def bet_surface_area(p_rel, v_ads, cross_nm2=N2_CROSS_NM2):
    """Monolayer volume Vm, BET constant C, and specific surface area.

    Linearizes  x/(v(1-x)) = 1/(Vm*C) + (C-1)/(Vm*C) * x  and regresses slope &
    intercept over the BET range.  Returns (Vm, C, SSA m^2/g).  v_ads in
    cm^3(STP)/g.
    """
    return petrolib.geochem_fluids.adsorption.bet_fit(p_rel, v_ads, cross_nm2=cross_nm2)


# ---------------------------------------------- classification ----------

def iupac_type(has_hysteresis, low_p_uptake, plateau):
    """IUPAC/BDDT isotherm type (I-V) from qualitative curve features.

      Type I   : strong low-pressure uptake + plateau (micropore filling)
      Type II  : no hysteresis, no plateau (macropore / nonporous)
      Type IV  : hysteresis loop present (mesopore capillary condensation)
      Type III : weak low-pressure uptake, no hysteresis, no plateau
      Type V   : hysteresis + weak low-pressure uptake
    """
    if has_hysteresis:
        return "IV" if low_p_uptake else "V"
    if low_p_uptake and plateau:
        return "I"
    if plateau:
        return "II"
    return "III"


def pore_size_class(d_nm):
    """IUPAC pore-size class: micro (<2 nm), meso (2-50 nm), macro (>50 nm)."""
    if d_nm < 2.0:
        return "micro"
    if d_nm <= 50.0:
        return "meso"
    return "macro"


def sorting_class(sizes_nm, weights):
    """Sorting from the spread (CV) of the pore-size distribution.

      CV < 0.5 -> good ; 0.5 <= CV < 1.0 -> medium ; CV >= 1.0 -> poor
    """
    sizes = np.asarray(sizes_nm, float)
    w = np.asarray(weights, float)
    w = w / w.sum()
    mean = np.sum(w * sizes)
    var = np.sum(w * (sizes - mean) ** 2)
    cv = np.sqrt(var) / mean
    if cv < 0.5:
        return "good"
    if cv < 1.0:
        return "medium"
    return "poor"


def classify_isotherm(shape, size_class, sorting):
    """Three-parameter curve type index in 1..27 from (shape, size, sorting).

    shape in {slit, cylinder, ink-bottle}; size in {micro, meso, macro};
    sorting in {good, medium, poor}.
    """
    shapes = ["slit", "cylinder", "ink-bottle"]
    sizes = ["micro", "meso", "macro"]
    sorts = ["good", "medium", "poor"]
    i = shapes.index(shape)
    j = sizes.index(size_class)
    k = sorts.index(sorting)
    return i * 9 + j * 3 + k + 1


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Adsorption Isotherm Classification (Shale)")
    print("=" * 60)

    # BET round-trip: plant Vm and C, forward an isotherm, recover them
    vm_true, c_true = 5.0, 80.0
    x = np.linspace(0.05, 0.30, 8)             # BET linear range
    v = bet_isotherm(x, vm_true, c_true)
    vm, c, ssa = bet_surface_area(x, v)
    print(f"  Vm = {vm:.3f} (true {vm_true})  C = {c:.1f} (true {c_true})")
    print(f"  specific surface area  = {ssa:.2f} m^2/g")
    assert abs(vm - vm_true) < 1e-3 and abs(c - c_true) < 1e-1
    assert ssa > 0.01                           # above the N2 detection limit

    # IUPAC types
    assert iupac_type(False, True, True) == "I"      # micropore
    assert iupac_type(True, True, False) == "IV"     # mesopore hysteresis
    assert iupac_type(False, False, False) == "III"
    print("  IUPAC types I/IV/III ok")

    # Pore-size classes at the boundaries
    assert pore_size_class(1.5) == "micro"
    assert pore_size_class(20.0) == "meso"
    assert pore_size_class(120.0) == "macro"

    # Sorting: a narrow distribution is well sorted, a wide one is poorly sorted
    narrow = sorting_class([9, 10, 11], [1, 3, 1])
    wide = sorting_class([1, 10, 100], [1, 1, 1])
    print(f"  sorting narrow / wide  = {narrow} / {wide}")
    assert narrow == "good" and wide == "poor"

    # Three-parameter classifier spans 1..27
    t = classify_isotherm("cylinder", "meso", "medium")
    print(f"  curve type (cyl/meso/med) = {t} of {PORE_TYPES}")
    assert 1 <= t <= PORE_TYPES
    assert classify_isotherm("slit", "micro", "good") == 1
    assert classify_isotherm("ink-bottle", "macro", "poor") == 27
    print("  PASS")
    return {"Vm": vm, "C": c, "ssa": ssa, "type_cyl_meso_med": t}


if __name__ == "__main__":
    test_all()
