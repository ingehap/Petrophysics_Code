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

Note: this is a narrative review with no display equations; the relations below
are the standard pressure/overburden framing of the deepwater challenges it
discusses.  Reported anchors: Miocene porosity 20-35%, Wilcox porosity 15-25%
with permeability spanning ~3 orders of magnitude, water depths to ~10,000 ft.
Pressures in psi, depths in ft, densities in g/cm^3.
"""

import numpy as np

PSI_PER_FT_FRESH = 0.433     # hydrostatic gradient of fresh water, psi/ft
SEAWATER_SG = 1.025          # seawater specific gravity

# Reservoir-quality anchors by GoM play / geologic epoch, as reported in the
# review.  Depths are sub-mudline reservoir depths (ft), porosity in fraction,
# permeability in md.  None marks a quantity the article does not pin down.
GOM_PLAYS = {
    "pleistocene": dict(reservoir_depth_ft=(3000, 10000), net_thickness_ft=(0, 300),
                        porosity=None, permeability_md=None),
    "pliocene":    dict(reservoir_depth_ft=(5000, 13000), net_thickness_ft=(0, 300),
                        porosity=None, permeability_md=None),
    "miocene":     dict(reservoir_depth_ft=None, net_thickness_ft=None,
                        porosity=(0.20, 0.35), permeability_md=None),
    "wilcox":      dict(reservoir_depth_ft=(10000, 30000), net_thickness_ft=(0, 6000),
                        porosity=(0.15, 0.25), permeability_md=(0.1, 10.0)),
}


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
    return PSI_PER_FT_FRESH * SEAWATER_SG * np.asarray(water_depth_ft, float)


def overburden_pressure(water_depth_ft, sediment_depth_ft, sediment_sg=2.3):
    """Total overburden pressure at a sub-mudline depth

        P_ob = 0.433*(SG_sw*water_depth + SG_sed*sediment_depth)   [psi],

    showing how the light seawater column lowers the overburden gradient
    (referenced to total depth) relative to an onshore well - the deepwater
    narrow-margin challenge.
    """
    return PSI_PER_FT_FRESH * (SEAWATER_SG * np.asarray(water_depth_ft, float)
                               + sediment_sg * np.asarray(sediment_depth_ft, float))


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
    return PSI_PER_FT_FRESH * SEAWATER_SG * np.asarray(water_depth_ft, float)


def single_gradient_mudline_pressure(water_depth_ft, mud_sg):
    """Mud-line pressure with a conventional single (riser) mud gradient

        P = 0.433*SG_mud*water_depth   [psi],

    the heavy-mud column carried all the way to the rig - always heavier than
    the dual-gradient case, the comparison that motivates dual-gradient drilling.
    """
    return PSI_PER_FT_FRESH * mud_sg * np.asarray(water_depth_ft, float)


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

    # Dual-gradient drilling lightens the mud-line pressure vs a single mud column
    p_dual = dual_gradient_mudline_pressure(5000, mud_sg=1.6)
    p_single = single_gradient_mudline_pressure(5000, mud_sg=1.6)
    print(f"  mud-line pressure  dual={p_dual:.0f} psi  single={p_single:.0f} psi")
    assert p_dual < p_single
    assert np.isclose(p_dual, seawater_hydrostatic_pressure(5000))
    print("  PASS")
    return {"seawater_head_5000ft": float(p5000), "ob_gradient": float(grad),
            "k_orders": float(n), "dual_gradient_relief_psi": float(p_single - p_dual)}


if __name__ == "__main__":
    test_all()
