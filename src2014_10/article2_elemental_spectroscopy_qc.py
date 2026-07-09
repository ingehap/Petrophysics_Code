"""
Article 2: Application and Quality Control of Core Data for the Development and
           Validation of Elemental Spectroscopy Log Interpretation
Susan Herron, Michael Herron, Iain Pirie, Pablo Saldungaray, Paul Craddock,
Alyssa Charsky, Marina Polyakov, Frank Shray, Ting Li (2014)
Reference: Petrophysics Vol. 55, No. 5 (October 2014), pp. 392-414
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best of the 2014 SPWLA Annual Logging Symposium.  Core elemental and mineral
data are used to build and validate elemental-spectroscopy log interpretation.
Capture element yields are converted to dry-weight concentrations by oxide
closure; a QCMIN check reconstructs the elements from a core mineralogy; and
core elements are placed on a common basis through organic-matter and spectral-
interference corrections.

Implements:

  - Oxide closure: renormalize yield-derived oxides to unity (proportionality
    factor F)
  - Element->oxide association factors (Si->SiO2 = 2.139, Ca->CaCO3 = 2.5)
  - Pyrite from sulfur  pyrite = 100*(S/53)
  - QCMIN element reconstruction from a mineralogy and its ad/aad/score metrics
  - Organic matter from TOC  OM = 1.2*TOC  (Eq. A-1) and organic-dilution
    removal (Eq. A-2)
  - Iron spectral-interference correction  Fe + 0.14*Al  (Eq. A-3)
  - Grain density on a common basis from TOC (Eq. A-6)

Note: this issue's PDF has a text layer; Eqs. A-1, A-2, A-3 and the closure /
QCMIN logic are transcribed from the body, while the grain-density relation
(Eq. A-6) and table cell values were dropped in extraction and reconstructed in
standard mass-balance form (Grau, 1989; Hertzog, 1989).  Concentrations as
weight fractions unless noted; densities in g/cm^3.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- oxide closure --------------

def oxide_from_element(element_fraction, association_factor):
    """Convert an element weight fraction to its mineral-oxide fraction via a
    fixed association factor, e.g. Si->SiO2 (2.139) or Ca->CaCO3 (2.5)."""
    return element_fraction * association_factor


def oxide_closure(oxides):
    """Oxide closure: because fluid elements (Cl, H) are excluded, the raw oxide
    sum is < 1; the proportionality factor F renormalizes it to unity

        F = 1/sum(oxides),   closed = F*oxides.

    Returns (closed_oxides, F).
    """
    ox = np.asarray(oxides, float)
    f = 1.0 / ox.sum()
    return f * ox, float(f)


def pyrite_from_sulfur(sulfur_wt_pct):
    """Pyrite weight percent from sulfur (siliciclastics; FeS2 is 53 wt% S)

        pyrite [wt%] = 100*(S/53).
    """
    return 100.0 * np.asarray(sulfur_wt_pct, float) / 53.0


# ---------------------------------------------- QCMIN --------------

def qcmin_reconstruct_elements(mineral_abundances, mineral_element_matrix):
    """Reconstruct element concentrations from a mineralogy

        C_element_j = sum_i abundance_i*conc_element_j_in_mineral_i,

    with ``mineral_element_matrix`` of shape (n_minerals, n_elements).
    """
    a = np.asarray(mineral_abundances, float)
    m = np.asarray(mineral_element_matrix, float)
    return a @ m


def qcmin_metrics(reference, measured):
    """QCMIN quality metrics per element and overall

        ad  = mean(reference - measured)        (bias)
        aad = mean|reference - measured|        (accuracy)
        score = sum(aad) over all elements.

    Score scale: 0-3 excellent, 3-5 good, 5-6 marginal, >6 serious.
    """
    ref = np.asarray(reference, float)
    mea = np.asarray(measured, float)
    ad = float(np.mean(ref - mea))
    aad = np.abs(ref - mea)
    return {"ad": ad, "aad": float(np.mean(aad)), "score": float(np.sum(aad))}


# ---------------------------------------------- organic & interference --------------

def organic_matter_from_toc(toc):
    """Organic matter from TOC, carbon being 0.83 of organic matter (Eq. A-1)

        OM = TOC/0.83 = 1.2*TOC.
    """
    return petrolib.porosity_lithology.kerogen_mass_fraction(toc, k=1.2)


def remove_organic_dilution(element_fraction, toc):
    """Place a core element on a mineral (organic-free) basis (Eq. A-2)

        core_element' = core_element/(1 - 1.2*TOC).
    """
    return element_fraction / (1.0 - organic_matter_from_toc(toc))


def correct_iron_interference(fe, al, delta_fe_al=0.14):
    """Iron spectral-interference correction (Eq. A-3)

        adj_Fe = Fe + 0.14*Al,

    because the log iron already includes a fraction of aluminium when Al is not
    used as an independent standard.
    """
    return fe + delta_fe_al * al


def grain_density(toc, rho_matrix, rho_organic=1.2):
    """Grain (matrix) density from TOC on a common basis (Eq. A-6)

        1/rho_grain = (1 - 1.2*TOC)/rho_matrix + (1.2*TOC)/rho_organic,

    a mass-weighted harmonic mix of mineral matrix and organic matter.
    """
    om = organic_matter_from_toc(toc)
    return 1.0 / ((1.0 - om) / rho_matrix + om / rho_organic)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Elemental Spectroscopy Core Data QC")
    print("=" * 60)

    # Oxide association and closure
    assert np.isclose(oxide_from_element(0.45, 2.139), 0.96255)
    closed, f = oxide_closure([0.50, 0.20, 0.10])
    print(f"  closure factor F = {f:.4f}")
    assert np.isclose(closed.sum(), 1.0) and f > 1.0

    # Pyrite from sulfur
    assert np.isclose(pyrite_from_sulfur(2.65), 5.0)

    # QCMIN: reconstruct Si from quartz (47 wt% Si) and metrics
    # minerals: [quartz, calcite]; elements: [Si, Ca]
    mat = np.array([[0.47, 0.0], [0.0, 0.40]])
    recon = qcmin_reconstruct_elements([0.6, 0.4], mat)
    print(f"  reconstructed [Si, Ca] = {np.round(recon, 4)}")
    assert np.isclose(recon[0], 0.282) and np.isclose(recon[1], 0.16)
    m = qcmin_metrics([28.0, 16.0, 1.0], [27.0, 16.5, 1.2])
    print(f"  QCMIN score = {m['score']:.2f}")
    assert m["score"] > 0 and m["score"] < 3  # excellent

    # Organic matter and dilution removal
    assert np.isclose(organic_matter_from_toc(0.05), 0.06)
    base = remove_organic_dilution(0.40, 0.05)
    assert base > 0.40  # mineral-basis concentration is higher

    # Iron interference adds a fraction of aluminium
    assert np.isclose(correct_iron_interference(2.0, 5.0), 2.7)

    # Grain density: organic matter lowers the grain density below the matrix
    rg = grain_density(toc=0.08, rho_matrix=2.71)
    print(f"  grain density = {rg:.3f} g/cm3")
    assert 1.2 < rg < 2.71
    assert np.isclose(grain_density(0.0, 2.71), 2.71)
    print("  PASS")
    return {"F": f, "QCMIN": m["score"], "rho_grain": float(rg)}


if __name__ == "__main__":
    test_all()
