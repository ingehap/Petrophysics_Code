"""
Article 2: Presalt Carbonate Evaluation for Santos Basin, Offshore Brazil
Boyd, Souza, Carneiro, Machado, Trevizan, Santos, Neto, Bagueira, Polinski,
Bertolini (2015)
Reference: Petrophysics Vol. 56, No. 6 (December 2015), pp. 577-591
DOI: none assigned (this issue predates SPWLA DOI assignment)

A formation-evaluation workflow for the oil-wet presalt carbonates of Lula
Field: (1) lithology and porosity, (2) pore typing and permeability, (3) fluid
saturation.  Porosity is partitioned into micro/meso/macro (vug) fractions;
permeability follows the SDR NMR transform; Archie's equation is used with a
cementation exponent m from NMR porosity partitioning and a saturation exponent
n recovered from the dielectric "textural exponent" (the water tortuosity factor
m*n).  Water saturation above the transition zone can be cross-checked from the
microporosity-to-total-porosity ratio.

Implements:

  - SDR NMR permeability  k = C*phi^A*T2lm^B
  - Archie water saturation with variable m, n
  - Saturation exponent from the dielectric textural exponent  n = (m*n)/m
  - Macroporosity (vug) indicator from the sonic-vs-total porosity deficit
  - Microporosity water-saturation estimate (capillary-bound fluid)

Note: this is a carbonate case study; the relations below are the standard
petrophysics it relies on (SDR, Archie, dielectric textural exponent).  The
typeset glyphs were dropped in extraction, so they are standard-form
reconstructions.  Resistivity in ohm-m, permeability in mD, times in ms,
porosity/saturation as fractions.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- permeability --------------

def sdr_permeability(phi, t2lm, c=4.0, a_exp=4.0, b_exp=2.0):
    """SDR NMR permeability  k = C*phi^A*T2lm^B,

    with A typically 2-4 and B the T2 exponent (~2); T2lm is the log-mean T2.
    """
    return petrolib.nmr.sdr(phi, t2lm, a=c, m=a_exp, n=b_exp)


# ---------------------------------------------- saturation --------------

def archie_sw(rt, rw, phi, m=2.0, n=2.0, a=1.0):
    """Archie water saturation  Sw = (a*Rw/(phi^m * Rt))^(1/n)."""
    return petrolib.saturation_resistivity.archie_sw(rt, rw, phi=phi, a=a, m=m, n=n)


def saturation_exponent_from_dielectric(textural_exponent, m):
    """Saturation exponent from the dielectric water tortuosity factor

        textural_exponent = m*n  ->  n = (m*n)/m,

    combining the dielectric-derived product with the NMR-partitioning m.
    """
    return textural_exponent / m


def microporosity_sw(phi_micro, phi_total):
    """Water saturation from the microporosity (capillary-bound) fraction

        Sw ~ phi_micro/phi_total,

    a reliable estimate above the oil-water transition zone where the
    microporosity holds the capillary-bound water.
    """
    return phi_micro / phi_total


def macroporosity_indicator(phi_sonic, phi_total):
    """Macro/vug porosity indicator from the porosity deficit

        phi_macro ~ phi_total - phi_sonic,

    since sonic (compressional) porosity under-reads the more spherical, less
    compressible macropores/vugs relative to NMR/nuclear total porosity.
    """
    return phi_total - phi_sonic


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Santos Presalt Carbonate Evaluation")
    print("=" * 60)

    # SDR permeability rises with porosity and T2 log-mean
    k = sdr_permeability(0.18, 120.0)
    print(f"  SDR permeability       = {k:.2f} mD")
    assert k > 0 and sdr_permeability(0.25, 120.0) > k

    # Dielectric textural exponent: recover n from the product m*n and m
    m, n_true = 2.1, 1.9
    mn = m * n_true
    n = saturation_exponent_from_dielectric(mn, m)
    print(f"  m / textural m*n / n   = {m:.2f} / {mn:.2f} / {n:.2f}")
    assert np.isclose(n, n_true)

    # Archie saturation with the recovered exponents
    sw = archie_sw(50.0, 0.05, 0.18, m=m, n=n)
    print(f"  Archie Sw              = {sw:.3f}")
    assert 0 < sw < 1

    # Microporosity Sw estimate matches the bound-fluid ratio
    assert np.isclose(microporosity_sw(0.04, 0.18), 0.04 / 0.18)

    # Macroporosity indicator is positive when sonic under-reads total porosity
    assert macroporosity_indicator(0.12, 0.18) > 0
    print("  PASS")
    return {"k_sdr": float(k), "n": float(n), "Sw": float(sw)}


if __name__ == "__main__":
    test_all()
