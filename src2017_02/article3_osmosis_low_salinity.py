"""
Article 3: Wettability Effects on Osmosis as an Oil-Mobilization Mechanism During
           Low-Salinity Waterflooding
Fredriksen, Rognmo, Sandengen, Ferno (2017)
Reference: Petrophysics Vol. 58, No. 1 (February 2017), pp. 28-35
DOI: none assigned (this issue predates SPWLA DOI assignment)

Osmosis - water transport across an oil film acting as a semipermeable membrane,
driven by a salinity (chemical-potential) contrast - is identified as a pore-
scale oil-mobilization mechanism during low-salinity waterflooding.  This module
implements the standard physics the (experimental) paper invokes: the van't Hoff
osmotic pressure, the Stokes-Einstein diffusivity, Fick's diffusive flux, and the
capillary-pressure convention.

Implements:

  - van't Hoff osmotic pressure  Pi = i*c*R*T
  - Stokes-Einstein diffusivity  D = kB*T/(6*pi*mu*r)
  - Fick's diffusive flux  J = -D*dc/dx
  - Capillary pressure convention  Pc = P_oil - P_water

Note: this is an experimental/mechanism paper that names but does not typeset its
equations, so the relations below are the standard forms it relies on, not
formulas transcribed from it.  SI units (Pi in Pa, D in m^2/s).
"""

import numpy as np

R_GAS = 8.314                # J/(mol K)
KB = 1.380649e-23            # J/K


# ---------------------------------------------- osmosis / diffusion --------------

def vant_hoff_osmotic_pressure(dissociation, concentration_mol_m3, temperature_k):
    """van't Hoff osmotic pressure  Pi = i*c*R*T  (i = van't Hoff factor)."""
    return dissociation * concentration_mol_m3 * R_GAS * temperature_k


def stokes_einstein_diffusivity(temperature_k, viscosity, radius):
    """Stokes-Einstein diffusivity  D = kB*T/(6*pi*mu*r).

    Water diffusion is inversely proportional to the oil-phase viscosity.
    """
    return KB * temperature_k / (6.0 * np.pi * viscosity * radius)


def fick_flux(diffusivity, dconcentration, dx):
    """Fick's first-law diffusive flux  J = -D*dc/dx."""
    return -diffusivity * dconcentration / dx


def capillary_pressure(p_oil, p_water):
    """Capillary pressure convention  Pc = P_oil - P_water."""
    return p_oil - p_water


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Osmosis in Low-Salinity Waterflooding")
    print("=" * 60)

    # Osmotic pressure scales with the salinity (concentration) contrast
    pi_high = vant_hoff_osmotic_pressure(2.0, 1000.0, 323.0)   # ~1 mol/L NaCl
    pi_low = vant_hoff_osmotic_pressure(2.0, 100.0, 323.0)
    print(f"  osmotic pressure hi/lo = {pi_high/1e6:.2f} / {pi_low/1e6:.2f} MPa")
    assert pi_high > pi_low > 0

    # Stokes-Einstein: a more viscous oil slows water diffusion
    assert stokes_einstein_diffusivity(323.0, 7.9e-3, 1.4e-10) < \
           stokes_einstein_diffusivity(323.0, 4.3e-3, 1.4e-10)

    # Fick flux points down the concentration gradient
    assert fick_flux(2e-9, 1000.0, 0.01) < 0

    # Capillary-pressure convention
    assert np.isclose(capillary_pressure(1.2e5, 1.0e5), 2.0e4)
    print("  PASS")
    return {"osmotic_MPa": float(pi_high / 1e6)}


if __name__ == "__main__":
    test_all()
