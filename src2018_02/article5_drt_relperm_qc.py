"""
Article 5: Using Digital Rock Technology to Quality Control and Reduce
           Uncertainty in Relative Permeability Measurements
Schembre-McCabe, Kamath (2018)
DOI: 10.30632/petro_059_1_a4

Pore-network flow simulation on segmented micro-CT images produces physically
bounding water-wet and oil-wet relative-permeability scenarios; a measured
relative-permeability curve that falls outside those bounds is flagged as an
experimental outlier, narrowing the remaining-oil-saturation uncertainty.  This
module implements the Corey relative-permeability bounding and outlier logic and
the fractional-flow comparison the workflow uses.

Implements:

  - Corey water/oil relative permeability with wettability-dependent endpoints
  - Water-wet and oil-wet bounding kr envelopes
  - Outlier flag for a measured kr outside the simulated bounds
  - Water fractional flow with a water-oil viscosity ratio

Note: the paper's method is a simulation/QC workflow rather than a closed-form
model, so the relations below are the standard Corey relative-permeability forms
used to realize the bounding/outlier logic it describes.  Saturations and
relative permeabilities fractional.
"""

import numpy as np


# ---------------------------------------------- Corey kr --------------

def normalized_sw(sw, swir, sor):
    """Normalized water saturation  Swn = (Sw - Swir)/(1 - Swir - Sor)."""
    return np.clip((np.asarray(sw, float) - swir) / (1.0 - swir - sor), 0.0, 1.0)


def corey_kr(sw, swir, sor, krw_max, kro_max, nw, no):
    """Corey water/oil relative permeabilities, returns (krw, kro)."""
    swn = normalized_sw(sw, swir, sor)
    return krw_max * swn ** nw, kro_max * (1.0 - swn) ** no


def relperm_bounds(sw):
    """Water-wet and oil-wet bounding water relative permeabilities.

    Oil-wet rock lets water flow more freely (higher krw) at a given saturation
    than water-wet rock, so the two scenarios bracket a measured curve.
    """
    krw_ww, _ = corey_kr(sw, swir=0.20, sor=0.25, krw_max=0.25, kro_max=1.0, nw=4.0, no=2.0)
    krw_ow, _ = corey_kr(sw, swir=0.15, sor=0.15, krw_max=0.55, kro_max=0.8, nw=2.0, no=3.0)
    lower = np.minimum(krw_ww, krw_ow)
    upper = np.maximum(krw_ww, krw_ow)
    return lower, upper


def is_outlier(krw_measured, lower, upper, tol=1e-9):
    """Flag a measured krw that falls outside the simulated [lower, upper] bounds."""
    krw = np.asarray(krw_measured, float)
    return (krw < lower - tol) | (krw > upper + tol)


def fractional_flow(krw, kro, viscosity_ratio=0.13):
    """Water fractional flow  fw = (krw/mu_w)/(krw/mu_w + kro/mu_o).

    viscosity_ratio = mu_w/mu_o (0.13 in the paper).
    """
    lam_w = np.asarray(krw, float)
    lam_o = np.asarray(kro, float) * viscosity_ratio
    return lam_w / (lam_w + lam_o)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: DRT Relative-Permeability QC")
    print("=" * 60)

    sw = np.linspace(0.2, 0.75, 12)
    lower, upper = relperm_bounds(sw)
    assert np.all(upper >= lower) and np.all(np.diff(upper) >= -1e-9)

    # A curve inside the envelope passes; one above the upper bound is flagged
    inside = 0.5 * (lower + upper)
    assert not np.any(is_outlier(inside, lower, upper))
    flagged = is_outlier(upper + 0.1, lower, upper)
    print(f"  outliers above bound   = {int(flagged.sum())}/{len(sw)}")
    assert np.all(flagged)

    # Fractional flow runs 0 -> 1 and is monotonic in water saturation
    krw_ww, kro_ww = corey_kr(sw, 0.20, 0.25, 0.25, 1.0, 4.0, 2.0)
    fw = fractional_flow(krw_ww, kro_ww)
    print(f"  fw range               = {fw[0]:.3f} -> {fw[-1]:.3f}")
    assert fw[0] < fw[-1] and np.all(np.diff(fw) >= -1e-9)
    print("  PASS")
    return {"fw_end": float(fw[-1])}


if __name__ == "__main__":
    test_all()
