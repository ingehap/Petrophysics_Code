"""
Article 6: Investigating Delaware Basin Bone Spring and Wolfcamp Observations
           Through Core-Based Quantification: Case Study in the Integrated
           Workflow, Including Closed Retort Comparisons
Perry, Zumberge, Cheng (2022)
DOI: 10.30632/PJV63N1-2022a6

A core-based workflow for the Delaware Basin Bone Spring / Wolfcamp.  The
methodological contribution is a *closed retort* crushed-rock extraction
that seals the sample during thermal fluid recovery, raising collection
efficiency from ~80% (open) to ~95% and directly quantifying both bulk-
volume oil and water.  Results are cross-checked against three independent
porosities and partitioned into free vs bound fluids for log calibration.

The article is a descriptive case study with no symbolic equations; the
relationships below are the standard petrophysical forms the text invokes
(Boyle's-law density porosity, fluid-summation porosity, NMR-T2 free/bound
partition, Schmoker TOC-from-density), implemented as faithful proxies for
the paper's uncalibrated crossplots.

Implements:

  - Boyle's-law (density) porosity     phi = 1 - rho_b / rho_g
  - Fluid-summation porosity           phi = (BV_oil + BV_water)
  - Crushing fluid loss                intact_NMR - crushed_NMR
  - Water / oil saturation             Sw = BV_water/phi, So = BV_oil/phi
  - Free / bound water (NMR T2 cutoff) BVI / FFI partition
  - Schmoker TOC from bulk density
  - Open -> closed retort efficiency correction
"""

import numpy as np

T2_FREE_CUTOFF_MS = 10.0     # NMR T2 free-water cutoff
RETORT_HOT_C = 300.0         # thermal-extraction hold temperature
FREE_WATER_C = 105.0         # >= 105 C accumulates as free water
OPEN_EFFICIENCY = 0.80       # open-retort collection efficiency
CLOSED_EFFICIENCY = 0.95     # closed-retort collection efficiency
BV_OIL_PRONE = 0.01          # v/v threshold above which a sample is oil-prone


# ---------------------------------------------- porosity ---------------

def boyles_law_porosity(rho_bulk, rho_grain):
    """Total porosity from bulk and grain density:  phi = 1 - rho_b/rho_g."""
    return 1.0 - np.asarray(rho_bulk, float) / np.asarray(rho_grain, float)


def fluid_summation_porosity(bv_oil, bv_water):
    """Total porosity from summed fluid volumes per bulk volume (v/v)."""
    return np.asarray(bv_oil, float) + np.asarray(bv_water, float)


# ---------------------------------------------- crushing fluid loss -----

def crushing_fluid_loss(nmr_intact_cc, nmr_crushed_cc):
    """Fluid lost on crushing = intact NMR signal - crushed NMR signal."""
    return np.asarray(nmr_intact_cc, float) - np.asarray(nmr_crushed_cc, float)


# ---------------------------------------------- saturations ------------

def water_saturation(bv_water, phi):
    """Sw = BV_water / phi."""
    phi = np.asarray(phi, float)
    return np.asarray(bv_water, float) / np.where(phi > 1e-9, phi, np.nan)


def oil_saturation(bv_oil, phi):
    """So = BV_oil / phi."""
    phi = np.asarray(phi, float)
    return np.asarray(bv_oil, float) / np.where(phi > 1e-9, phi, np.nan)


def is_oil_prone(bv_oil, threshold=BV_OIL_PRONE):
    """Oil-prone flag: bulk-volume oil above ~0.01 v/v."""
    return bool(np.asarray(bv_oil, float) > threshold)


# ---------------------------------------------- free / bound water ------

def free_bound_water(t2_ms, amplitudes, cutoff_ms=T2_FREE_CUTOFF_MS):
    """Partition an NMR T2 distribution into free (FFI) and bound (BVI) water.

    Returns (free_water, bound_water) as summed amplitudes either side of the
    T2 cutoff.
    """
    t2 = np.asarray(t2_ms, float)
    amp = np.asarray(amplitudes, float)
    free = float(amp[t2 > cutoff_ms].sum())
    bound = float(amp[t2 <= cutoff_ms].sum())
    return free, bound


# ---------------------------------------------- TOC from density --------

def schmoker_toc(rho_bulk, a=154.5, b=57.26):
    """Schmoker (1979) style TOC (wt%) from bulk density:  TOC = a/rho_b - b.

    Coefficients are exposed as tunable parameters; the paper fits a local
    polynomial ratio-of-density relation (Schmoker's law) without publishing
    coefficients, so these defaults are placeholders to be recalibrated.
    """
    return a / np.asarray(rho_bulk, float) - b


# ---------------------------------------------- retort efficiency -------

def efficiency_corrected_volume(measured_volume, efficiency):
    """Recover true fluid volume from a partial-collection measurement."""
    return np.asarray(measured_volume, float) / efficiency


def collection_efficiency(collected, total):
    """efficiency = collected / total."""
    return float(collected) / float(total)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Closed-Retort Core-Based Quantification")
    print("=" * 60)

    # Representative Wolfcamp B sample (within the paper's ranges)
    rho_bulk, rho_grain = 2.55, 2.71
    bv_oil, bv_water = 0.014, 0.045
    nmr_intact, nmr_crushed = 8.0, 6.6

    phi_boyle = boyles_law_porosity(rho_bulk, rho_grain)
    phi_fluid = fluid_summation_porosity(bv_oil, bv_water)
    print(f"  phi  Boyle's law       = {phi_boyle:.3f}")
    print(f"  phi  fluid summation   = {phi_fluid:.3f}")
    assert abs(phi_boyle - phi_fluid) < 0.01, "porosities should agree ~1:1"
    assert 0.02 <= phi_boyle <= 0.14

    loss = crushing_fluid_loss(nmr_intact, nmr_crushed)
    loss_pct = 100.0 * loss / nmr_intact
    print(f"  crushing fluid loss    = {loss:.2f} cc ({loss_pct:.1f}%)")
    assert loss >= 0 and 10 <= loss_pct <= 25

    sw = water_saturation(bv_water, phi_fluid)
    so = oil_saturation(bv_oil, phi_fluid)
    print(f"  Sw / So                = {sw:.2f} / {so:.2f}")
    assert 0.20 <= sw <= 0.90
    assert abs(sw + so - 1.0) < 1e-6
    assert is_oil_prone(bv_oil)

    # WOR from bulk-volume ratio
    wor = bv_water / bv_oil
    print(f"  water/oil ratio        = {wor:.1f} : 1")
    assert 2.0 <= wor <= 10.0

    # NMR free/bound water partition at the 10 ms cutoff
    t2 = np.array([1, 3, 8, 12, 30, 100], float)
    amp = np.array([0.004, 0.006, 0.005, 0.010, 0.012, 0.008], float)
    free, bound = free_bound_water(t2, amp)
    print(f"  free / bound water     = {free:.3f} / {bound:.3f}")
    assert free > 0 and bound > 0

    # Schmoker TOC from density (monotonic decrease with density)
    toc_lo = schmoker_toc(2.45)
    toc_hi = schmoker_toc(2.65)
    print(f"  Schmoker TOC @2.45/2.65 = {toc_lo:.2f} / {toc_hi:.2f} wt%")
    assert toc_lo > toc_hi

    # retort efficiency: closed beats open, correction recovers true volume
    eff = collection_efficiency(95.0, 100.0)
    true_vol = efficiency_corrected_volume(bv_water, OPEN_EFFICIENCY)
    print(f"  closed efficiency      = {eff:.2f} (open {OPEN_EFFICIENCY})")
    assert CLOSED_EFFICIENCY > OPEN_EFFICIENCY
    assert true_vol > bv_water
    print("  PASS")
    return {"phi": phi_boyle, "Sw": sw, "So": so, "fluid_loss_pct": loss_pct,
            "wor": wor}


if __name__ == "__main__":
    test_all()
