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
    print("  PASS")
    return {"NG_amalgamated": ng_a, "pore_contact_amalgamated": pc_a,
            "recovery_amalgamated": float(recovery_proxy(pc_a))}


if __name__ == "__main__":
    test_all()
