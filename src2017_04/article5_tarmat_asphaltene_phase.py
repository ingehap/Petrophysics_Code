"""
Article 5: Scanning Electron Micrographs of Tar-Mat Intervals Formed by
           Asphaltene Phase Transition
Pfeiffer, Di Primio, Achourov, Mullins (2017)
Reference: Petrophysics Vol. 58, No. 2 (April 2017), pp. 141-152
DOI: none assigned (this issue predates SPWLA DOI assignment)

Tar mats form when asphaltenes phase-separate at the base of an oil column,
driven by a gravity-controlled asphaltene concentration gradient (Flory-Huggins-
Zuo equation of state) and by worsening solvency as the gas/oil ratio rises.
This module is a quantitative companion to the (qualitative) paper: it implements
the FHZ gravity term for the asphaltene gradient, the linear optical-density /
asphaltene relation downhole-fluid-analysis uses, and the asphaltene-weight
thresholds the paper reports for the onset and the solid tar mat.

Implements:

  - FHZ gravity asphaltene gradient (relative concentration vs depth)
  - Optical density linear in asphaltene content  OD = k*phi_asphaltene
  - Asphaltene-content classification (suspension / onset / solid tar mat)
  - Qualitative solvency vs gas/oil ratio (solvency falls as GOR rises)

Note: the paper itself presents no typeset equations (it cites the FHZ EoS and
Yen-Mullins model), so the relations below are the standard FHZ gravity term and
DFA optical relation it relies on, not formulas transcribed from it.  The 35 wt%
(onset) and 60 wt% (solid tar) thresholds are transcribed from the paper.  SI.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

R_GAS = 8.314                # J/(mol K)
G_ACCEL = 9.81

ONSET_WT = 0.35              # asphaltene weight fraction at phase-separation onset
SOLID_TAR_WT = 0.60         # asphaltene weight fraction of a solid tar mat
SUSPENSION_LIMIT_WT = 0.25  # stable-suspension limit for heavy oil


# ---------------------------------------------- gradient --------------

def fhz_relative_concentration(depth, depth_ref, molar_volume, rho_fluid, rho_asph,
                               temperature_k=353.0):
    """FHZ gravity asphaltene gradient (relative to a reference depth)

        phi(h)/phi(h_ref) = exp[ Va*g*(rho_asph - rho_fluid)*(h - h_ref)/(R*T) ].

    With depth increasing downward and rho_asph > rho_fluid, asphaltenes
    concentrate toward the base of the column.
    """
    return petrolib.geochem_fluids.asphaltene.fhz_ratio(
        np.asarray(depth, float) - depth_ref, molar_volume, rho_asph - rho_fluid, temperature_k)


def optical_density(asphaltene_fraction, k=1.0):
    """DFA optical density, linear in flowing-oil asphaltene content  OD = k*phi_a."""
    return k * np.asarray(asphaltene_fraction, float)


def asphaltene_class(weight_fraction):
    """Classify an asphaltene weight fraction by the paper's tar-mat thresholds."""
    if weight_fraction >= SOLID_TAR_WT:
        return "solid tar mat"
    if weight_fraction >= ONSET_WT:
        return "phase-separation onset"
    if weight_fraction >= SUSPENSION_LIMIT_WT:
        return "heavy suspension"
    return "stable suspension"


def solvency(gor, gor_ref=500.0):
    """Relative asphaltene solvency of the oil; falls as the gas/oil ratio rises."""
    return gor_ref / (gor_ref + np.asarray(gor, float))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Tar-Mat Asphaltene Phase Transition")
    print("=" * 60)

    # Asphaltenes concentrate downward: deeper -> higher relative concentration
    depths = np.array([2000.0, 2050.0, 2100.0])     # m, increasing downward
    rel = fhz_relative_concentration(depths, 2000.0, molar_volume=1.0,
                                     rho_fluid=800.0, rho_asph=1200.0)
    print(f"  relative conc (base/top) = {rel[-1] / rel[0]:.3f}")
    assert rel[0] == 1.0 and rel[-1] > rel[0] and np.all(np.diff(rel) > 0)

    # Optical density is linear in asphaltene content
    assert np.isclose(optical_density(0.3, k=2.0), 0.6)

    # Classification across the reported thresholds
    labels = [asphaltene_class(w) for w in (0.10, 0.30, 0.40, 0.65)]
    print(f"  classes 10/30/40/65 wt% = {labels}")
    assert labels == ["stable suspension", "heavy suspension",
                      "phase-separation onset", "solid tar mat"]

    # Solvency falls as the gas/oil ratio rises (worsening solvent)
    assert solvency(2000.0) < solvency(200.0)
    print("  PASS")
    return {"conc_ratio": float(rel[-1] / rel[0])}


if __name__ == "__main__":
    test_all()
