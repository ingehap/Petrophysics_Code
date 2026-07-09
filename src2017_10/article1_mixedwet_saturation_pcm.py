"""
Article 1: Improved Assessment of Hydrocarbon Saturation in Mixed-Wet Rocks With
           Complex Pore Structure
Garcia, Heidari, Rostami (2017)
Reference: Petrophysics Vol. 58, No. 5 (October 2017), pp. 454-469
DOI: none assigned (this issue predates SPWLA DOI assignment)

An analytical electrical-conductivity model for partially saturated, mixed-wet
rock built by Pore Combination Modeling (sequential effective-medium theory).
It generalizes Archie with porosity- and saturation-percolation thresholds and a
connectivity term, reduces to Archie when the thresholds vanish, and folds in
wettability through the oil-wet fractional index by mixing water-wet and oil-wet
blocks with a CRIM law.

Implements:

  - Archie conductivity  sigma_R = sigma_w*a*phi^m*Sw^n
  - Montaron connectivity model  sigma_R = sigma_w*(Cw + phi*Sw)^mu
  - Percolation-threshold conductivity  sigma_R = sigma_w*a*(phi*Sw - phi_c*Sw_c)^mu
  - CRIM mixing of water-wet / oil-wet blocks by the oil-wet fraction Xo
  - Water saturation inverted from conductivity

Note: this issue's PDF has a text layer but every typeset equation glyph was
dropped in extraction, so the relations are faithful standard-form
reconstructions from the prose and nomenclature.  Conductivities in S/m.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- conductivity models --------------

def archie_conductivity(sigma_w, phi, sw, a=1.0, m=2.0, n=2.0):
    """Archie bulk conductivity  sigma_R = sigma_w*a*phi^m*Sw^n  (Eq. 1)."""
    return petrolib.saturation_resistivity.archie_conductivity(sw, sigma_w * a, phi=phi, m=m, n=n)


def montaron_conductivity(sigma_w, cw, phi, sw, mu=2.0):
    """Montaron connectivity conductivity  sigma_R = sigma_w*(Cw + phi*Sw)^mu  (Eq. 2).

    Cw = water-connectivity correction, mu = conductivity exponent (~2).
    """
    return sigma_w * (cw + phi * np.asarray(sw, float)) ** mu


def percolation_conductivity(sigma_w, phi, sw, phi_c=0.0, sw_c=0.0, a=1.0, mu=2.0):
    """Percolation-threshold conductivity  sigma_R = sigma_w*a*(phi*Sw - phi_c*Sw_c)^mu.

    Reduces to Archie (with m = n = mu) when both percolation thresholds vanish.
    """
    base = np.clip(phi * np.asarray(sw, float) - phi_c * sw_c, 0.0, None)
    return sigma_w * a * base ** mu


def crim_wettability_mixing(sigma_ww, sigma_ow, x_oil):
    """CRIM mixing of water-wet and oil-wet block conductivities by oil-wet fraction

        sigma = ((1 - Xo)*sqrt(sigma_ww) + Xo*sqrt(sigma_ow))^2.
    """
    return ((1.0 - x_oil) * np.sqrt(sigma_ww) + x_oil * np.sqrt(sigma_ow)) ** 2


def water_saturation(sigma_r, sigma_w, phi, a=1.0, m=2.0, n=2.0):
    """Invert Archie for water saturation  Sw = (sigma_R/(sigma_w*a*phi^m))^(1/n)."""
    sw = (np.asarray(sigma_r, float) / (sigma_w * a * phi ** m)) ** (1.0 / n)
    return np.clip(sw, 0.0, 1.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Mixed-Wet Saturation (PCM)")
    print("=" * 60)

    sw_w = 0.6
    # Conductivity rises with water saturation
    assert archie_conductivity(5.0, 0.2, 0.8) > archie_conductivity(5.0, 0.2, 0.4)

    # Percolation form reduces to Archie when the thresholds vanish (m=n=mu)
    arch = archie_conductivity(5.0, 0.2, sw_w, a=1.0, m=2.0, n=2.0)
    perc = percolation_conductivity(5.0, 0.2, sw_w, phi_c=0.0, sw_c=0.0, a=1.0, mu=2.0)
    print(f"  Archie / percolation   = {arch:.4f} / {perc:.4f} S/m")
    assert np.isclose(arch, perc)

    # A nonzero percolation threshold lowers conductivity
    assert percolation_conductivity(5.0, 0.2, sw_w, phi_c=0.05, sw_c=0.5) < perc

    # CRIM wettability mixing lies between the two block conductivities
    mix = crim_wettability_mixing(1.0, 0.2, x_oil=0.4)
    assert 0.2 < mix < 1.0

    # Saturation inversion round-trips
    sw = water_saturation(arch, 5.0, 0.2)
    print(f"  recovered Sw           = {sw:.3f}  (true {sw_w})")
    assert np.isclose(sw, sw_w)
    print("  PASS")
    return {"sigma_archie": float(arch), "Sw": float(sw)}


if __name__ == "__main__":
    test_all()
