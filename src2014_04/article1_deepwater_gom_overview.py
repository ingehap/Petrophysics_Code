"""
Article 1: Deepwater Exploration and Production in the Gulf of Mexico -
           Challenges and Opportunities
Hani Elshahawi (2014)
Reference: Petrophysics Vol. 55, No. 2 (April 2014), pp. 81-87
DOI: none assigned (this issue predates SPWLA DOI assignment)

Special Issue on Deepwater.  This lead article is a narrative review of the
deepwater Gulf of Mexico plays (Pleistocene, Pliocene, Miocene and Lower-Tertiary
Wilcox).  This module captures the quantitative relations that frame the review's
"challenges": the seawater hydrostatic head and the reduced overburden gradient
that narrow the deepwater drilling margin, a water-depth play classification, and
the porosity-permeability decoupling the article emphasizes.

Implements:

  - Water-depth classification (shelf / deepwater / ultra-deepwater)
  - Seawater hydrostatic pressure  P = rho_sw*g*water_depth
  - Overburden pressure (seawater column + sediment) and its gradient
  - Porosity-permeability spread (orders of magnitude at fixed porosity)
  - Reservoir-quality classification by GoM play / geologic epoch
  - Dual-gradient mud-line pressure reduction (the riser-margin solution)
  - Deepwater demand / supply split and compound (CAGR) spending growth
  - GoM discovered-resource and Lower-Tertiary geometry anchors
  - Landmark deepwater structure water-depth records (m <-> ft)

Note: this is a narrative review with no display equations; the relations below
are the standard pressure/overburden framing of the deepwater challenges it
discusses.  Reported anchors: Miocene porosity 20-35%, Wilcox porosity 15-25%
with permeability spanning ~3 orders of magnitude, water depths to ~10,000 ft.
Pressures in psi, depths in ft, densities in g/cm^3.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

PSI_PER_FT_FRESH = 0.433     # hydrostatic gradient of fresh water, psi/ft
SEAWATER_SG = 1.025          # seawater specific gravity

# Reservoir-quality anchors by GoM play / geologic epoch, as reported in the
# review.  Depths are sub-mudline reservoir depths (ft), porosity in fraction,
# permeability in md.  `net_thickness_ft` is producing/net sand; for the Lower-
# Tertiary Wilcox the article instead reports the gross turbidite *sand-pile*
# thickness (2,500 ft in E Walker Ridge to 6,000 ft in Alaminos Canyon), kept
# in a separate `sand_pile_thickness_ft` field so it is not confused with net
# pay.  None marks a quantity the article does not pin down.
GOM_PLAYS = {
    "pleistocene": dict(reservoir_depth_ft=(3000, 10000), net_thickness_ft=(0, 300),
                        porosity=None, permeability_md=None,
                        water_depth_ft=None, sand_pile_thickness_ft=None),
    "pliocene":    dict(reservoir_depth_ft=(5000, 13000), net_thickness_ft=(0, 300),
                        porosity=None, permeability_md=None,
                        water_depth_ft=None, sand_pile_thickness_ft=None),
    "miocene":     dict(reservoir_depth_ft=None, net_thickness_ft=None,
                        porosity=(0.20, 0.35), permeability_md=None,
                        water_depth_ft=None, sand_pile_thickness_ft=None),
    "wilcox":      dict(reservoir_depth_ft=(10000, 30000), net_thickness_ft=None,
                        porosity=(0.15, 0.25), permeability_md=(0.1, 10.0),
                        water_depth_ft=(5000, 10000),
                        sand_pile_thickness_ft=(2500, 6000)),
}

# Discovered-resource and basin-geometry anchors for the Gulf of Mexico, as
# reported in the review's introduction.
GOM_RESOURCE = dict(
    discovered_oil_Bbbl=25.0,        # ~25 billion bbl oil discovered to date
    discovered_gas_Tcf=200.0,        # ~200 Tcf gas discovered to date
    basin_area_mi2=600000.0,         # ~600,000 mi^2 (about the size of Alaska)
    undiscovered_global_deepwater_BBOE=500.0,  # risk-weighted, >500 billion BOE
    wilcox_offshore_distance_mi=175.0,         # Lower-Tertiary trend ~175 mi offshore
    wilcox_fairway_mi=(80.0, 400.0),           # ~80 x 400 mi fairway
)

# Global liquids-demand framing (by ~2020): incremental demand and the deepwater
# share expected to supply it (MMbbl/d).
GLOBAL_INCREMENTAL_DEMAND_MMBD = 27.0
DEEPWATER_SUPPLY_MMBD = 10.0

# Landmark deepwater structures and their water depths (ft), tracing the
# industry's march into ever-deeper water (Fig. 3 and the historical narrative).
STRUCTURE_RECORDS = [
    ("Cognac",      1978,  1025),
    ("Bullwinkle",  1991,  1353),
    ("Auger TLP",   1994,  2854),
    ("Mensa",       1997,  5300),
    ("Na Kika",     2003,  7745),
]

FT_PER_M = 3.28084


# ---------------------------------------------- water depth --------------

def water_depth_class(water_depth_ft):
    """Classify a well by water depth

        shelf            : < 1,000 ft
        deepwater        : 1,000 - 5,000 ft
        ultra-deepwater  : > 5,000 ft.
    """
    d = water_depth_ft
    if d < 1000:
        return "shelf"
    if d < 5000:
        return "deepwater"
    return "ultra-deepwater"


# ---------------------------------------------- pressures --------------

def seawater_hydrostatic_pressure(water_depth_ft):
    """Hydrostatic pressure at the mud line from the seawater column

        P = 0.433*SG_sw*water_depth   [psi].
    """
    return petrolib.integrity_drilling.hydrostatic_pressure_psi(water_depth_ft, sg=SEAWATER_SG)


def overburden_pressure(water_depth_ft, sediment_depth_ft, sediment_sg=2.3):
    """Total overburden pressure at a sub-mudline depth

        P_ob = 0.433*(SG_sw*water_depth + SG_sed*sediment_depth)   [psi],

    showing how the light seawater column lowers the overburden gradient
    (referenced to total depth) relative to an onshore well - the deepwater
    narrow-margin challenge.
    """
    return petrolib.integrity_drilling.overburden_pressure_psi(
        water_depth_ft, sediment_depth_ft, sw_sg=SEAWATER_SG, sediment_sg=sediment_sg)


def overburden_gradient(water_depth_ft, sediment_depth_ft, sediment_sg=2.3):
    """Effective overburden gradient referenced to total depth below sea level

        grad = P_ob/(water_depth + sediment_depth)   [psi/ft].
    """
    total = np.asarray(water_depth_ft, float) + np.asarray(sediment_depth_ft, float)
    return overburden_pressure(water_depth_ft, sediment_depth_ft, sediment_sg) / total


# ---------------------------------------------- reservoir quality --------------

def permeability_orders_of_magnitude(k_min, k_max):
    """Spread of permeability at a fixed porosity band

        n = log10(k_max/k_min),

    quantifying the porosity-permeability decoupling the review highlights for
    the Wilcox (k spans ~3 orders within 15-25% porosity).
    """
    return np.log10(k_max / k_min)


def reservoir_quality(play):
    """Reported reservoir-quality ranges for a named GoM play / epoch

        play in {pleistocene, pliocene, miocene, wilcox},

    returning the dict of (reservoir_depth_ft, net_thickness_ft, porosity,
    permeability_md) ranges the review tabulates.  The deep, low-permeability,
    moderate-porosity Wilcox is the end member that frames the article's
    "challenges" (deep, hot, tight, narrow-margin).
    """
    return GOM_PLAYS[play.lower()]


# ---------------------------------------------- dual-gradient drilling --------------

def dual_gradient_mudline_pressure(water_depth_ft, mud_sg):
    """Mud-line pressure with dual-gradient drilling

        P = 0.433*(SG_sw*water_depth)   [psi],

    i.e. the long riser column of heavy mud is replaced by a seawater-density
    return so that only the seawater head acts above the mud line.  This lifts
    the effective pressure profile off the fracture gradient and widens the
    narrow deepwater margin (the article's dual-gradient / managed-pressure
    solution).  Compare with the single-gradient riser pressure that would
    instead carry the full mud column to surface.
    """
    return petrolib.integrity_drilling.hydrostatic_pressure_psi(water_depth_ft, sg=SEAWATER_SG)


def single_gradient_mudline_pressure(water_depth_ft, mud_sg):
    """Mud-line pressure with a conventional single (riser) mud gradient

        P = 0.433*SG_mud*water_depth   [psi],

    the heavy-mud column carried all the way to the rig - always heavier than
    the dual-gradient case, the comparison that motivates dual-gradient drilling.
    """
    return PSI_PER_FT_FRESH * mud_sg * np.asarray(water_depth_ft, float)


# ---------------------------------------------- demand & economics --------------

def deepwater_demand_share(deepwater_supply=DEEPWATER_SUPPLY_MMBD,
                           incremental_demand=GLOBAL_INCREMENTAL_DEMAND_MMBD):
    """Fraction of the global incremental liquids demand met by deepwater

        share = deepwater_supply/incremental_demand,

    ~10 of ~27 MMbbl/d (~37%) by 2020 in the review's framing - the demand pull
    that motivates the deepwater push.
    """
    return deepwater_supply / incremental_demand


def capex_growth(initial, cagr, years):
    """Compound (CAGR) growth of deepwater E&P spending

        P = P0*(1 + cagr)**years,

    the one genuinely compounding relation the review states (deepwater spend
    growing ~8-10%/yr toward the >$250 billion cumulative E&P investment).
    """
    return initial * (1.0 + cagr) ** np.asarray(years, float)


# ---------------------------------------------- unit helpers --------------

def ft_to_m(feet):
    """Convert feet to metres (the structure records are reported in both)."""
    return np.asarray(feet, float) / FT_PER_M


def deepest_structure():
    """The deepest-water landmark structure on record (name, year, water_depth_ft).

    Returns Na Kika (7,745 ft), the ultra-deepwater end of the historical march
    the review traces from Cognac (1,025 ft, first >1,000 ft) onward.
    """
    return max(STRUCTURE_RECORDS, key=lambda s: s[2])


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Deepwater GoM - Challenges & Opportunities")
    print("=" * 60)

    # Water-depth classification
    assert water_depth_class(800) == "shelf"
    assert water_depth_class(2854) == "deepwater"       # Auger TLP
    assert water_depth_class(7745) == "ultra-deepwater"  # Na Kika

    # Seawater hydrostatic head grows with water depth
    p5000 = seawater_hydrostatic_pressure(5000)
    print(f"  seawater head at 5,000 ft = {p5000:.0f} psi")
    assert np.isclose(p5000, 0.433 * 1.025 * 5000)

    # The water column lowers the overburden gradient below the ~1 psi/ft
    # onshore value
    grad = overburden_gradient(5000, 10000)
    print(f"  overburden gradient (5,000 ft water + 10,000 ft sed) = {grad:.3f} psi/ft")
    assert 0.433 < grad < 1.0
    # deeper sediment (more rock) raises the gradient back toward 1 psi/ft
    assert overburden_gradient(5000, 20000) > grad

    # Wilcox permeability spans ~3 orders of magnitude
    n = permeability_orders_of_magnitude(0.1, 100.0)
    print(f"  permeability spread = {n:.1f} orders of magnitude")
    assert np.isclose(n, 3.0)

    # Reservoir quality: the Wilcox is the deep, tight, moderate-porosity play
    wilcox = reservoir_quality("Wilcox")
    print(f"  Wilcox: depth {wilcox['reservoir_depth_ft']} ft, "
          f"porosity {wilcox['porosity']}, k {wilcox['permeability_md']} md")
    assert wilcox["reservoir_depth_ft"][1] == 30000
    assert wilcox["porosity"] == (0.15, 0.25)
    # its permeability band itself spans ~2 orders within one porosity range
    assert permeability_orders_of_magnitude(*wilcox["permeability_md"]) >= 2.0
    assert reservoir_quality("miocene")["porosity"] == (0.20, 0.35)
    # the Wilcox gross sand pile (2,500-6,000 ft) is tracked separately from net
    # pay (which the article does not report for the Wilcox)
    assert wilcox["sand_pile_thickness_ft"] == (2500, 6000)
    assert wilcox["net_thickness_ft"] is None
    assert wilcox["water_depth_ft"] == (5000, 10000)

    # Dual-gradient drilling lightens the mud-line pressure vs a single mud column
    p_dual = dual_gradient_mudline_pressure(5000, mud_sg=1.6)
    p_single = single_gradient_mudline_pressure(5000, mud_sg=1.6)
    print(f"  mud-line pressure  dual={p_dual:.0f} psi  single={p_single:.0f} psi")
    assert p_dual < p_single
    assert np.isclose(p_dual, seawater_hydrostatic_pressure(5000))

    # Deepwater is expected to supply ~10 of the ~27 MMbbl/d incremental demand
    share = deepwater_demand_share()
    print(f"  deepwater share of incremental demand = {share*100:.0f}%")
    assert np.isclose(share, 10.0 / 27.0)

    # Deepwater spend compounds at ~8-10%/yr; 10 yr at 9% nearly 2.4x
    grown = capex_growth(100.0, 0.09, 10)
    print(f"  $100 spend at 9% CAGR over 10 yr = ${grown:.0f}")
    assert grown > 200.0 and np.isclose(grown, 100.0 * 1.09 ** 10)

    # Structure records: Na Kika is the deepest; feet/metres round-trip
    name, year, depth_ft = deepest_structure()
    print(f"  deepest structure: {name} ({year}) at {depth_ft} ft = {ft_to_m(depth_ft):.0f} m")
    assert name == "Na Kika" and depth_ft == 7745
    assert np.isclose(ft_to_m(depth_ft) * FT_PER_M, depth_ft)
    assert water_depth_class(depth_ft) == "ultra-deepwater"

    # Discovered-resource anchors
    assert GOM_RESOURCE["discovered_oil_Bbbl"] == 25.0
    assert GOM_RESOURCE["basin_area_mi2"] == 600000.0
    print("  PASS")
    return {"seawater_head_5000ft": float(p5000), "ob_gradient": float(grad),
            "k_orders": float(n), "dual_gradient_relief_psi": float(p_single - p_dual),
            "deepwater_demand_share": float(share), "capex_10yr_9pct": float(grown),
            "deepest_structure": name}


if __name__ == "__main__":
    test_all()
