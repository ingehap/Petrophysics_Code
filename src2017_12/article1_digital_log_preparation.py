"""
Article 1 (Tutorial): Preparing Your Digital Well Logs for Computer-Based
                      Interpretation
Thomas (2017)
Reference: Petrophysics Vol. 58, No. 6 (December 2017), pp. 559-563
DOI: none assigned (this issue predates SPWLA DOI assignment)

This tutorial lists the preprocessing every digital log needs before automated
interpretation.  The quantitative cores are: the density-porosity transform
(bulk density to porosity given assumed grain and fluid densities, and how gas
inflates it), the iterative flushed-zone saturation solve, "squaring" a log into
constant-value beds before crossplotting, depth alignment on bed-boundary
inflection points, and the minimum bed thickness a deep tool needs to read true
resistivity.

Implements:

  - Density porosity  phi = (rho_ma - rho_b)/(rho_ma - rho_fl)
  - Flushed-zone saturation  Sxo = (a*Rmf/(phi^m*Rxo))^(1/n)
  - Bed "squaring" into constant-value zones
  - Bed-boundary alignment on the inflection (max-gradient) point
  - Minimum-bed-thickness check for a deep resistivity tool

Note: this is a qualitative best-practices tutorial (no typeset equations), so
the relations below are the standard log-preprocessing forms it describes.
Densities in g/cm^3, resistivities in ohm.m, fractions dimensionless.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- porosity --------------

def density_porosity(rho_b, rho_ma=2.65, rho_fl=1.0):
    """Density porosity  phi = (rho_ma - rho_b)/(rho_ma - rho_fl).

    rho_ma defaults to sandstone (2.65; use 2.71 for carbonate); a gas-bearing
    zone lowers rho_b and so inflates the computed porosity.
    """
    return petrolib.porosity_lithology.density_porosity(rho_b, rho_ma, rho_fl)


def flushed_zone_saturation(rmf, rxo, phi, a=1.0, m=2.0, n=2.0):
    """Flushed-zone (Archie) saturation  Sxo = (a*Rmf/(phi^m*Rxo))^(1/n).

    Solved iteratively in practice because porosity itself may be refined; here
    the closed Archie form is returned, clipped to [0, 1].
    """
    sxo = (a * rmf / (np.asarray(phi, float) ** m * rxo)) ** (1.0 / n)
    return np.clip(sxo, 0.0, 1.0)


# ---------------------------------------------- bed handling --------------

def square_log(values, boundaries):
    """Square a log into constant-value beds: replace each bed by its mean.

    boundaries is a sorted list of sample indices delimiting the beds (first 0,
    last len(values)).  Avoids smearing thin-bed boundaries before crossplotting.
    """
    v = np.asarray(values, float)
    out = v.copy()
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        out[s:e] = v[s:e].mean()
    return out


def inflection_index(values):
    """Index of the bed boundary = the steepest-gradient (inflection) sample."""
    return int(np.argmax(np.abs(np.diff(np.asarray(values, float)))))


def bed_resolved(thickness_ft, min_thickness_ft=12.0):
    """True if a deep induction bed is thick enough (>= ~12 ft) to read true Rt."""
    return np.asarray(thickness_ft, float) >= min_thickness_ft


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1 (Tutorial): Digital Log Preparation")
    print("=" * 60)

    # Gas lowers bulk density, inflating the apparent density porosity
    phi_liq = density_porosity(2.40)
    phi_gas = density_porosity(2.25)
    print(f"  phi liquid / gassy     = {phi_liq:.3f} / {phi_gas:.3f}")
    assert phi_gas > phi_liq > 0

    # Flushed-zone saturation falls as Rxo rises
    assert flushed_zone_saturation(0.1, 5.0, 0.2) < flushed_zone_saturation(0.1, 2.0, 0.2)

    # Squaring replaces each bed with its mean (constant within a bed)
    sq = square_log([1.0, 1.2, 0.8, 5.0, 5.1, 4.9], [0, 3, 6])
    print(f"  squared log            = {np.array2string(sq, precision=2)}")
    assert np.allclose(sq[:3], 1.0) and np.allclose(sq[3:], 5.0)

    # Inflection sits at the bed boundary; thickness rule for deep Rt
    assert inflection_index([1, 1, 1, 9, 9, 9]) == 2
    assert bed_resolved(15.0) and not bed_resolved(6.0)
    print("  PASS")
    return {"phi_gas": float(phi_gas)}


if __name__ == "__main__":
    test_all()
