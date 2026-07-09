"""
Article 2: Borehole Carbon Corrections Enable Accurate TOC Determination from
           Nuclear Spectroscopy
Jeffrey Miles and Rob Badry (2014)
Reference: Petrophysics Vol. 55, No. 3 (June 2014), pp. 219-228
DOI: none assigned (this issue predates SPWLA DOI assignment)

Total organic carbon (TOC) from nuclear-spectroscopy carbon yields is biased by
inorganic (carbonate) carbon and by carbon in the borehole fluid (oil-based mud).
The paper removes both: the total inorganic carbon yield is computed from the
capture-spectroscopy mineralogy, and the borehole-carbon contribution is removed
with either a constant offset or a self-calibrating function of the environment.

Implements:

  - Constant borehole correction  Y_TOC = Y_c - Y_TIC - delta_Borehole  (Eq. 1)
  - Self-calibrating correction  Y_TOC = Y_c - Y_TIC - f_Borehole(env)  (Eq. 2)
  - Borehole-carbon function f_Borehole (linear in diameter + standoff/HI
    crossterms)
  - Total inorganic carbon yield from carbonate mineralogy
  - TOC weight fraction from the corrected carbon yield

Note: this issue's PDF has a text layer; Eqs. 1 and 2 survived verbatim, while
the explicit f_Borehole fit is reconstructed from the described form (dominant
linear borehole-diameter term plus standoff*diameter and HI*diameter
crossterms).  Yields in counts/counts, TOC in weight fraction.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- inorganic carbon --------------

def total_inorganic_carbon_yield(carbonate_fractions, carbon_factors):
    """Total inorganic (carbonate) carbon yield from the capture-spectroscopy
    mineralogy

        Y_TIC = sum_i V_i*c_i,

    summing each carbonate's volume fraction times its carbon association factor
    (e.g. calcite, dolomite, siderite).
    """
    v = np.asarray(carbonate_fractions, float)
    c = np.asarray(carbon_factors, float)
    return float(v @ c)


# ---------------------------------------------- borehole correction --------------

def constant_borehole_correction(y_carbon, y_tic, delta_borehole):
    """Corrected organic-carbon yield with a constant borehole offset (Eq. 1)

        Y_TOC = Y_c - Y_TIC - delta_Borehole,

    where delta_Borehole is chosen so TOC reads zero in an organic-free zone.
    """
    return y_carbon - y_tic - delta_borehole


def borehole_carbon_function(diameter, standoff, hydrogen_index,
                             a=1.0e-3, b=2.0e-4, c=5.0e-4, d=1.0e-3):
    """Self-calibrating borehole-carbon yield function (env-dependent term of
    Eq. 2)

        f_Borehole = a*D + b*(standoff*D) + c*(HI*D) + d,

    dominated by the borehole diameter D (in.), with standoff and hydrogen-index
    crossterms; reconstructed from the described Monte-Carlo fit.
    """
    return a * diameter + b * (standoff * diameter) + c * (hydrogen_index * diameter) + d


def variable_borehole_correction(y_carbon, y_tic, diameter, standoff,
                                 hydrogen_index, **kw):
    """Corrected organic-carbon yield with the self-calibrating function (Eq. 2)

        Y_TOC = Y_c - Y_TIC - f_Borehole(D, standoff, HI).
    """
    f = borehole_carbon_function(diameter, standoff, hydrogen_index, **kw)
    return y_carbon - y_tic - f


# ---------------------------------------------- TOC weight --------------

def toc_weight_fraction(y_toc, calibration):
    """TOC weight fraction from the corrected organic-carbon yield

        TOC = calibration*Y_TOC,

    a linear yield-to-weight conversion from the spectroscopy closure.
    """
    return petrolib.nuclear.toc_from_yield(y_toc, calibration)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Borehole Carbon Corrections for TOC")
    print("=" * 60)

    # Inorganic carbon yield from a limestone/dolomite mix
    y_tic = total_inorganic_carbon_yield([0.30, 0.10], [0.12, 0.13])
    print(f"  Y_TIC = {y_tic:.4f}")
    assert y_tic > 0

    # Constant correction: organic-free zone (Y_c = Y_TIC + delta) reads TOC ~ 0
    delta = 0.006
    y_toc0 = constant_borehole_correction(y_tic + delta, y_tic, delta)
    assert np.isclose(y_toc0, 0.0)

    # An organic zone with excess carbon yields a positive TOC
    y_c = y_tic + delta + 0.020
    y_toc = constant_borehole_correction(y_c, y_tic, delta)
    print(f"  Y_TOC = {y_toc:.4f}")
    assert np.isclose(y_toc, 0.020)

    # Borehole-carbon function grows with hole size (dominant term)
    f_small = borehole_carbon_function(8.0, 0.5, 0.15)
    f_large = borehole_carbon_function(16.0, 0.5, 0.15)
    print(f"  f_Borehole(8in)={f_small:.5f}  f_Borehole(16in)={f_large:.5f}")
    assert f_large > f_small

    # Variable correction removes the modeled borehole carbon
    y_toc_v = variable_borehole_correction(y_c, y_tic, diameter=12.0,
                                           standoff=0.5, hydrogen_index=0.15)
    assert y_toc_v < y_c - y_tic

    # TOC weight scales with the corrected yield
    toc = toc_weight_fraction(y_toc, calibration=2.0)
    assert np.isclose(toc, 0.040) and toc_weight_fraction(0.0, 2.0) == 0.0
    print("  PASS")
    return {"Y_TIC": y_tic, "Y_TOC": float(y_toc), "TOC": float(toc)}


if __name__ == "__main__":
    test_all()
