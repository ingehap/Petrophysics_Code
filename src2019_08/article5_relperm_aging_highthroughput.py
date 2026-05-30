"""
Article 5: In-Situ Investigation of Aging Protocol Effect on Relative
           Permeability Measurements Using High-Throughput Experimentation
           Methods
Mascle, Youssef, Deschamps, Vizika (2019)
DOI: 10.30632/PJV60N4-2019a5

Core aging (exposure to crude oil at reservoir conditions) shifts wettability
from water-wet toward mixed/oil-wet, which changes the relative-permeability
curves: endpoints, Corey exponents and the crossover saturation all move.
High-throughput in-situ measurements quantify how the aging protocol affects the
measured kr.  This module implements the Corey relative-permeability model and
an aging transform of its wettability parameters.

Implements:

  - Corey water / oil relative permeability
  - Crossover saturation (krw = kro)
  - Aging transform of the wettability parameters (water-wet -> mixed-wet)

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the issue's table of contents and these are
faithful standard-form reconstructions of the Corey relative-permeability model
the aging study measures.
"""

import numpy as np


# ---------------------------------------------- Corey kr ----------------

def effective_sw(sw, swr, sor):
    """Normalized water saturation  Se = (Sw - Swr)/(1 - Swr - Sor)."""
    return np.clip((np.asarray(sw, float) - swr) / (1.0 - swr - sor), 0.0, 1.0)


def corey_krw(sw, params):
    """Corey water relative permeability  krw = krw_max * Se^nw."""
    se = effective_sw(sw, params["swr"], params["sor"])
    return params["krw_max"] * se ** params["nw"]


def corey_kro(sw, params):
    """Corey oil relative permeability  kro = kro_max * (1 - Se)^no."""
    se = effective_sw(sw, params["swr"], params["sor"])
    return params["kro_max"] * (1.0 - se) ** params["no"]


def crossover_saturation(params, grid=None):
    """Water saturation where krw = kro (the kr crossover)."""
    if grid is None:
        grid = np.linspace(params["swr"], 1.0 - params["sor"], 2001)
    diff = np.abs(corey_krw(grid, params) - corey_kro(grid, params))
    return float(grid[np.argmin(diff)])


# ---------------------------------------------- aging -------------------

WATER_WET = dict(swr=0.20, sor=0.30, nw=4.0, no=2.0, krw_max=0.15, kro_max=1.0)


def age(params, strength=1.0):
    """Aging transform: shift water-wet parameters toward mixed/oil-wet.

    Aging lowers Swr (water no longer clings), raises Sor changes, lowers the
    water exponent nw and raises the water endpoint krw_max (water flows more
    freely), pushing the crossover to lower Sw.  `strength` in [0, 1].
    """
    p = dict(params)
    p["swr"] = params["swr"] - 0.05 * strength
    p["nw"] = params["nw"] - 1.5 * strength
    p["krw_max"] = params["krw_max"] + 0.25 * strength
    p["no"] = params["no"] + 1.0 * strength
    return p


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Aging Effect on Relative Permeability")
    print("=" * 60)

    # Corey endpoints: krw=0 at Swr, kro=0 at 1-Sor
    assert abs(corey_krw(WATER_WET["swr"], WATER_WET)) < 1e-9
    assert abs(corey_kro(1.0 - WATER_WET["sor"], WATER_WET)) < 1e-9
    # monotonic
    sw = np.linspace(WATER_WET["swr"], 1 - WATER_WET["sor"], 30)
    assert np.all(np.diff(corey_krw(sw, WATER_WET)) >= -1e-12)
    assert np.all(np.diff(corey_kro(sw, WATER_WET)) <= 1e-12)

    # Water-wet crossover sits at high water saturation
    xo_ww = crossover_saturation(WATER_WET)
    print(f"  crossover Sw water-wet = {xo_ww:.3f}")
    assert xo_ww > 0.5

    # Aging shifts the crossover to lower water saturation (toward oil-wet) and
    # raises the water-endpoint relative permeability
    aged = age(WATER_WET, strength=1.0)
    xo_aged = crossover_saturation(aged)
    print(f"  crossover Sw aged      = {xo_aged:.3f}")
    print(f"  krw_max ww / aged      = {WATER_WET['krw_max']} / {aged['krw_max']}")
    assert xo_aged < xo_ww
    assert aged["krw_max"] > WATER_WET["krw_max"]
    # at a fixed Sw the aged (more oil-wet) rock passes more water
    assert corey_krw(0.55, aged) > corey_krw(0.55, WATER_WET)
    print("  PASS")
    return {"crossover_ww": xo_ww, "crossover_aged": xo_aged}


if __name__ == "__main__":
    test_all()
