"""
Article 2: Consideration of the Origin and Characteristics of Turbidite
           Sediments
John R. Dribus (2014)
Reference: Petrophysics Vol. 55, No. 2 (April 2014), pp. 88-95
DOI: none assigned (this issue predates SPWLA DOI assignment)

Special Issue on Deepwater.  This is a geology review of turbidite deposition.
The quantitative content is the Bouma (1962) fining-upward facies sequence and
the contrast in net-to-gross and sand-to-sand pore contact between amalgamated
and layered turbidite architectures (the Ross Formation data point), which
controls connectivity and recoverable fraction.

Implements:

  - Bouma sequence Ta-Te as an ordered fining-upward grain-size model
  - Net-to-gross and sand-to-sand pore contact interpolated between layered and
    amalgamated end members
  - A connectivity/recovery proxy from the sand-to-sand pore contact
  - Submarine-fan facies model (inner / middle channel-levee / outer fan) with
    reservoir-quality rank
  - Mass-transport-deposit (MTD) progression and its rock-quality preservation
    ranking (slide -> slump -> debris flow -> turbidity current)
  - Deepwater giant-discovery statistics (counts, gas Tcf, oil-equivalent BBOE)
  - Amalgamated-vs-layered recovery / connectivity contrast

Note: this is a narrative geology review with no display equations; the Bouma
sequence is an ordinal facies model and the only hard numbers are the Ross
Formation end members (amalgamated: N/G 90%, pore contact 67%; layered: N/G 45%,
pore contact 3%).  Grain sizes in mm, fractions as 0-1.
"""

import numpy as np

# Bouma divisions Ta (base, coarsest) -> Te (top, finest), with representative
# grain sizes in mm (fining upward from a waning turbidity current).
BOUMA = [
    ("Ta", "graded / massive sand",        0.50),
    ("Tb", "parallel-laminated sand",      0.25),
    ("Tc", "climbing-ripple lamination",   0.10),
    ("Td", "parallel-laminated silt",      0.03),
    ("Te", "massive / ductile clay",       0.004),
]

# Ross Formation end members (Fig. 13): (net_to_gross, sand-to-sand pore contact)
LAYERED = (0.45, 0.03)
AMALGAMATED = (0.90, 0.67)

# Deepwater giant-discovery statistics from the introduction (Fig. 1).  A
# "giant" is >500 million BOE found in water depths greater than 500 m.  Two
# windows are reported and they are additive.
GIANT_THRESHOLD_BOE = 500e6
DEEPWATER_THRESHOLD_M = 500.0
DISCOVERY_WINDOWS = {
    "2000-2009": dict(discoveries=18, gas_Tcf=105.0, oil_equiv_BBOE=33.0),
    "2010-2012": dict(discoveries=13, gas_Tcf=72.0,  oil_equiv_BBOE=18.0),
}

# Submarine-fan facies belts (proximal -> distal), each with a qualitative
# reservoir-quality rank (1 worst .. 4 best) and the review's characteristics.
# The outer fan is the best overall target; the proximal levee permeability
# anchor (>100 md) is from Fig. 9.
PROXIMAL_LEVEE_PERMEABILITY_MD = 100.0

FAN_FACIES = [
    ("inner_fan",  "proximal canyon: confined amalgamated channels, high "
                   "drawdown, poor lateral continuity",                 2),
    ("middle_fan", "channel-levee 'gull-wing' systems; proximal levees "
                   ">100 md, large levee storage",                      3),
    ("outer_fan",  "distal sheets and lobes: best overall target "
                   "(amalgamated or layered end members)",              4),
    ("mtd",        "mass-transport deposits: disorganized, variable "
                   "and generally poor reservoir",                      1),
]

# Mass-transport-deposit progression downslope, ordered by *decreasing*
# preservation of the original layering / rock properties (Shanmugam et al.,
# 1994; Fig. 16).  The turbidity current re-sorts sediment and can *restore*
# quality, so it ranks highest.  rank: higher = better preserved/sorted.
MTD_PROGRESSION = [
    ("slide",            "detached/faulted block, original layering intact", 4),
    ("slump",            "plastic deformation, fines begin redispersing",    3),
    ("debris_flow",      "chaotic muddy matrix with clasts/olistoliths",     1),
    ("turbidity_current", "re-suspended and re-sorted, quality restored",    5),
]


# ---------------------------------------------- Bouma sequence --------------

def bouma_sequence():
    """Return the Bouma (1962) divisions as (code, description, grain_size_mm),
    ordered from the basal Ta to the top Te.  The grain size fines upward."""
    return list(BOUMA)


def is_fining_upward(grain_sizes):
    """True if a vertical grain-size profile fines upward (Ta->Te ordering),
    i.e. grain size decreases from base to top."""
    g = np.asarray(grain_sizes, float)
    return bool(np.all(np.diff(g) <= 0))


# ---------------------------------------------- amalgamation --------------

def amalgamation_properties(amalgamation_ratio):
    """Net-to-gross and sand-to-sand pore contact for a given amalgamation ratio

        property = layered + ratio*(amalgamated - layered),   ratio in [0, 1],

    a linear blend between the layered (ratio 0) and amalgamated (ratio 1) Ross
    Formation end members.  Returns (net_to_gross, pore_contact).
    """
    r = amalgamation_ratio
    ng = LAYERED[0] + r * (AMALGAMATED[0] - LAYERED[0])
    pc = LAYERED[1] + r * (AMALGAMATED[1] - LAYERED[1])
    return ng, pc


def recovery_proxy(pore_contact):
    """Connectivity/recovery proxy from the sand-to-sand pore contact.

    Recoverable fraction scales with the connected sand network, here taken as
    the pore-contact fraction itself (0 = isolated layers, 1 = fully connected).
    """
    return np.clip(pore_contact, 0.0, 1.0)


def recovery_contrast():
    """Amalgamated-vs-layered recovery / connectivity ratio (the article's thesis)

        ratio = pore_contact_amalgamated / pore_contact_layered = 0.67/0.03,

    i.e. the amalgamated unit drains a roughly 20-fold better connected sand
    network than the layered unit, the central reason architecture (not just
    net-to-gross) controls recoverable volume in turbidite reservoirs.
    """
    return AMALGAMATED[1] / LAYERED[1]


# ---------------------------------------------- discovery statistics --------------

def discovery_totals():
    """Cumulative deepwater giant-discovery statistics across both windows

        returns dict(discoveries, gas_Tcf, oil_equiv_BBOE),

    summing the 2000-2009 and 2010-2012 windows to 31 discoveries, 177 Tcf gas
    and 51 billion BOE - the growth that frames the article's motivation.
    """
    totals = dict(discoveries=0, gas_Tcf=0.0, oil_equiv_BBOE=0.0)
    for w in DISCOVERY_WINDOWS.values():
        for k in totals:
            totals[k] += w[k]
    return totals


# ---------------------------------------------- fan facies & MTDs --------------

def fan_facies():
    """Return the submarine-fan facies belts as
    (code, description, reservoir_quality_rank), proximal to distal.  The outer
    fan carries the highest rank (best overall reservoir target)."""
    return list(FAN_FACIES)


def best_reservoir_facies():
    """The fan facies belt with the highest reservoir-quality rank.

    Returns its code.  Per the review (and contra a naive Walther's-law
    basinward-fining expectation) this is the distal *outer fan*, not the
    proximal channel.
    """
    return max(FAN_FACIES, key=lambda f: f[2])[0]


def mtd_progression():
    """Return the mass-transport-deposit progression as
    (code, description, preservation_rank), ordered downslope (slide -> slump
    -> debris flow -> turbidity current).  Preservation degrades from slide to
    debris flow, then the turbidity current re-sorts and *restores* quality."""
    return list(MTD_PROGRESSION)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Turbidite Sediments (Bouma Sequence)")
    print("=" * 60)

    # Bouma sequence is five divisions fining upward
    seq = bouma_sequence()
    sizes = [g for _, _, g in seq]
    print(f"  Bouma divisions: {[c for c, _, _ in seq]}")
    assert len(seq) == 5 and seq[0][0] == "Ta" and seq[-1][0] == "Te"
    assert is_fining_upward(sizes)
    assert not is_fining_upward(sizes[::-1])  # coarsening-up is not a Bouma seq

    # Amalgamation: end members reproduce the Ross Formation numbers
    ng_l, pc_l = amalgamation_properties(0.0)
    ng_a, pc_a = amalgamation_properties(1.0)
    print(f"  layered:    N/G={ng_l:.2f}  pore contact={pc_l:.2f}")
    print(f"  amalgamated: N/G={ng_a:.2f}  pore contact={pc_a:.2f}")
    assert (ng_l, pc_l) == LAYERED and (ng_a, pc_a) == AMALGAMATED

    # More amalgamation -> higher N/G, higher pore contact, higher recovery
    ng_m, pc_m = amalgamation_properties(0.5)
    assert ng_l < ng_m < ng_a and pc_l < pc_m < pc_a
    assert recovery_proxy(pc_a) > recovery_proxy(pc_l)
    print(f"  recovery proxy: layered={recovery_proxy(pc_l):.2f}  amalgamated={recovery_proxy(pc_a):.2f}")

    # Fan facies: the distal outer fan is the best overall reservoir target
    facies = fan_facies()
    print(f"  fan facies: {[c for c, _, _ in facies]}")
    assert len(facies) == 4
    assert best_reservoir_facies() == "outer_fan"
    assert PROXIMAL_LEVEE_PERMEABILITY_MD == 100.0

    # MTDs: a slide preserves rock properties better than a debris flow, and the
    # turbidity current re-sorts sediment to the highest quality of all
    mtd = dict((c, rank) for c, _, rank in mtd_progression())
    print(f"  MTD ranks: {mtd}")
    assert mtd["slide"] > mtd["slump"] > mtd["debris_flow"]
    assert mtd["turbidity_current"] == max(mtd.values())

    # Recovery contrast: the amalgamated unit is ~20x better connected
    contrast = recovery_contrast()
    print(f"  amalgamated/layered pore-contact ratio = {contrast:.0f}x")
    assert np.isclose(contrast, 0.67 / 0.03)

    # Discovery statistics: the two windows sum to 31 discoveries, 177 Tcf, 51 BBOE
    tot = discovery_totals()
    print(f"  deepwater giants: {tot['discoveries']} discoveries, "
          f"{tot['gas_Tcf']:.0f} Tcf, {tot['oil_equiv_BBOE']:.0f} BBOE")
    assert tot["discoveries"] == 31
    assert np.isclose(tot["gas_Tcf"], 177.0) and np.isclose(tot["oil_equiv_BBOE"], 51.0)
    assert GIANT_THRESHOLD_BOE == 500e6 and DEEPWATER_THRESHOLD_M == 500.0
    print("  PASS")
    return {"NG_amalgamated": ng_a, "pore_contact_amalgamated": pc_a,
            "recovery_amalgamated": float(recovery_proxy(pc_a)),
            "best_facies": best_reservoir_facies(),
            "recovery_contrast": float(contrast),
            "total_discoveries": tot["discoveries"]}


if __name__ == "__main__":
    test_all()
