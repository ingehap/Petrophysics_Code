"""
Article 1: High- and Low-Field NMR Relaxometry and Diffusometry of the Bakken
           Petroleum System
Kausik, Fellah, Feng, Simpson (2017)
Reference: Petrophysics Vol. 58, No. 4 (August 2017), pp. 341-351
DOI: none assigned (this issue predates SPWLA DOI assignment)

High- and low-field NMR T1-T2 and D-T2 maps separate the fluid and organic
components of the Bakken (kerogen, bitumen, clay-bound water, free oil, free
water).  Relaxation rates follow the Bloembergen-Purcell-Pound spectral density,
and the T1/T2 ratio (together with the diffusion coefficient) classifies each
component by published cutoffs; the apparent porosity is corrected for the
hydrogen index.

Implements:

  - BPP spectral density  J(omega) = tau_c/(1 + (omega*tau_c)^2)
  - T1/T2 ratio
  - Component classification by T1/T2, T2, and diffusion
  - Hydrogen-index porosity correction

Note: this issue's PDF has a text layer but the typeset relaxation-rate
equations were dropped in extraction, so the BPP form is a standard
reconstruction; the classification cutoffs (kerogen T1/T2 ~ 1000s, bitumen
~10-30) are transcribed from the paper.  Times in s, diffusion in m^2/s.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

D_WATER = 2.5e-9             # bulk water self-diffusion (m^2/s)


# ---------------------------------------------- relaxation --------------

def bpp_spectral_density(omega, tau_c):
    """BPP spectral density  J(omega) = tau_c/(1 + (omega*tau_c)^2)  (Eqs. 1-2).

    Peaks at omega = 0 (slow motion) and rolls off at high frequency.
    """
    return petrolib.nmr.bpp_spectral_density(omega, tau_c)


def t1_t2_ratio(t1, t2):
    """T1/T2 ratio (a mobility / surface-interaction discriminator)."""
    return petrolib.nmr.t1_t2_ratio(t1, t2)


def classify_component(t1t2, t2_s=None, diffusion=None):
    """Classify a Bakken NMR component from T1/T2, T2, and diffusion.

      - T1/T2 > 1000      -> kerogen (rigid solid organic)
      - 8 <= T1/T2 <= 35  -> bitumen
      - T2 < 0.003 s      -> clay-bound water
      - otherwise (T1/T2 ~ 1-2) split free oil / water by diffusion.
    """
    if t1t2 > 1000.0:
        return "kerogen"
    if 8.0 <= t1t2 <= 35.0:
        return "bitumen"
    if t2_s is not None and t2_s < 0.003:
        return "clay-bound water"
    if diffusion is not None:
        return "free water" if diffusion >= 0.5 * D_WATER else "free oil"
    return "free fluid"


def hi_corrected_porosity(apparent_porosity, hydrogen_index):
    """Correct apparent NMR porosity for the fluid hydrogen index  phi = phi_app/HI."""
    return petrolib.nuclear.phi_hi_correction(apparent_porosity, hydrogen_index)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Bakken NMR Relaxometry")
    print("=" * 60)

    # Spectral density is maximal at zero frequency and decays with omega
    assert bpp_spectral_density(0.0, 1e-9) > bpp_spectral_density(1e10, 1e-9)

    # T1/T2 ratio
    assert np.isclose(t1_t2_ratio(2000.0, 1.0), 2000.0)

    # Component classification across the Bakken system
    labels = (classify_component(3000.0),
              classify_component(20.0),
              classify_component(1.5, t2_s=0.001),
              classify_component(1.5, t2_s=0.05, diffusion=D_WATER),
              classify_component(1.5, t2_s=0.05, diffusion=1e-10))
    print(f"  components             = {labels}")
    assert labels == ("kerogen", "bitumen", "clay-bound water", "free water", "free oil")

    # Hydrogen-index correction raises porosity for HI < 1 (gas/light oil)
    assert hi_corrected_porosity(0.08, 0.8) > 0.08
    print("  PASS")
    return {"components": list(labels)}


if __name__ == "__main__":
    test_all()
