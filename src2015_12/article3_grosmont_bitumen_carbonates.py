"""
Article 3: Petrophysical Characterization of Bitumen-Saturated Karsted
           Carbonates: Case Study of the Multibillion Barrel Upper Devonian
           Grosmont Formation, Northern Alberta, Canada
MacNeil (2015)
Reference: Petrophysics Vol. 56, No. 6 (December 2015), pp. 592-614
DOI: none assigned (this issue predates SPWLA DOI assignment)

The Grosmont is the world's largest carbonate-hosted bitumen deposit, with
reservoir quality tied to a karst overprint (vugs, fractures, sinkholes).
Porosity is computed from the lithology-density log against a dolomite grain
density (~2.85 g/cm^3) and a neutron-density crossplot; water saturation uses an
Archie equation with variable m and n exponents, and the (immobile) bitumen
saturation is the hydrocarbon complement.  Core porosity and saturations come
from Dean-Stark (water + bitumen volumes), against which the logs are calibrated.

Implements:

  - Density porosity against a dolomite grain density (Eq. - density tool)
  - Archie water saturation with variable m, n
  - Bitumen (hydrocarbon) saturation  Sb = 1 - Sw
  - Dean-Stark porosity and saturations from water and bitumen volumes
  - Mud-filtrate / formation-water resistivity ratio (laterolog suitability)

Note: this is a carbonate case study; the relations below are the standard
petrophysics it relies on (density porosity, Archie, Dean-Stark).  The typeset
glyphs were dropped in extraction, so they are standard-form reconstructions.
Densities in g/cm^3, resistivity in ohm-m, porosity/saturation as fractions.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

RHO_DOLOMITE = 2.85           # g/cm^3, Grosmont dolomite grain density
RHO_BITUMEN = 1.01            # g/cm^3, typical bitumen density


# ---------------------------------------------- porosity --------------

def density_porosity(rho_b, rho_ma=RHO_DOLOMITE, rho_fl=RHO_BITUMEN):
    """Density porosity  phi = (rho_ma - rho_b)/(rho_ma - rho_fl),

    using the dolomite grain density and the (bitumen) pore-fluid density.
    """
    return petrolib.porosity_lithology.density_porosity(rho_b, rho_ma, rho_fl)


# ---------------------------------------------- saturation --------------

def archie_sw(rt, rw, phi, m=2.0, n=2.0, a=1.0):
    """Archie water saturation with variable m, n  Sw = (a*Rw/(phi^m*Rt))^(1/n)."""
    return petrolib.saturation_resistivity.archie_sw(rt, rw, phi=phi, a=a, m=m, n=n)


def bitumen_saturation(sw):
    """Bitumen (immobile hydrocarbon) saturation  Sb = 1 - Sw."""
    return 1.0 - np.asarray(sw, float)


# ---------------------------------------------- Dean-Stark --------------

def dean_stark(v_water, v_bitumen, v_bulk):
    """Core porosity and saturations from a Dean-Stark extraction.

    Measures the retorted water volume and the bitumen volume (by weight loss /
    solvent extraction).  Returns (porosity, Sw, Sb) with
        phi = (V_water + V_bitumen)/V_bulk,
        Sw  = V_water/(V_water + V_bitumen),  Sb = 1 - Sw.
    """
    v_pore = v_water + v_bitumen
    phi = v_pore / v_bulk
    sw = v_water / v_pore
    return phi, sw, 1.0 - sw


def rmf_rw_ratio(rmf, rw):
    """Mud-filtrate to formation-water resistivity ratio  Rmf/Rw.

    A high Rmf/Rw favors the array laterolog over induction in these
    fresh-mud, bitumen-limited-invasion carbonates.
    """
    return rmf / rw


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Grosmont Bitumen-Saturated Carbonates")
    print("=" * 60)

    # Density porosity against the dolomite grain density
    phi = density_porosity(2.55)
    print(f"  density porosity       = {phi:.3f}")
    assert 0 < phi < 0.5 and density_porosity(2.85) == 0.0

    # Archie Sw and the bitumen complement
    sw = archie_sw(200.0, 0.1, phi, m=1.9, n=2.2)
    sb = bitumen_saturation(sw)
    print(f"  Sw / Sb                = {sw:.3f} / {sb:.3f}")
    assert 0 < sw < 1 and np.isclose(sw + sb, 1.0)

    # Dean-Stark: porosity and saturations from measured volumes
    phi_ds, sw_ds, sb_ds = dean_stark(v_water=0.6, v_bitumen=2.4, v_bulk=15.0)
    print(f"  Dean-Stark phi/Sw/Sb   = {phi_ds:.3f} / {sw_ds:.3f} / {sb_ds:.3f}")
    assert np.isclose(phi_ds, 3.0 / 15.0) and np.isclose(sw_ds, 0.2)
    assert np.isclose(sw_ds + sb_ds, 1.0)

    # Rmf/Rw ratio is high for fresh mud over saline formation water
    assert rmf_rw_ratio(0.5, 0.1) == 5.0
    print("  PASS")
    return {"phi": float(phi), "Sw": float(sw), "Sb": float(sb), "phi_ds": float(phi_ds)}


if __name__ == "__main__":
    test_all()
