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

Note: this is a narrative review with no display equations; the relations below
are the standard pressure/overburden framing of the deepwater challenges it
discusses.  Reported anchors: Miocene porosity 20-35%, Wilcox porosity 15-25%
with permeability spanning ~3 orders of magnitude, water depths to ~10,000 ft.
Pressures in psi, depths in ft, densities in g/cm^3.
"""

import numpy as np

PSI_PER_FT_FRESH = 0.433     # hydrostatic gradient of fresh water, psi/ft
SEAWATER_SG = 1.025          # seawater specific gravity


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
    print("  PASS")
    return {"seawater_head_5000ft": float(p5000), "ob_gradient": float(grad),
            "k_orders": float(n)}


if __name__ == "__main__":
    test_all()
