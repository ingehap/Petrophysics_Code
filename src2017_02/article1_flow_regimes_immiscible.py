"""
Article 1: Flow Regimes During Immiscible Displacement
Armstrong, McClure, Berrill, Rucker, Schluter, Berg (2017)
Reference: Petrophysics Vol. 58, No. 1 (February 2017), pp. 10-18
DOI: none assigned (this issue predates SPWLA DOI assignment)

Fast micro-CT imaging and lattice-Boltzmann simulation of fractional flow show
that relative permeability depends on the capillary number, and that the
phase topology (connectivity, measured by the Euler characteristic) and the
fraction of flow carried by disconnected ganglia change with it.  This module
implements the Corey relative-permeability model the paper fits, the capillary
number, the Euler characteristic, and the ganglion flux fraction.

Implements:

  - Corey relative permeabilities  krw, kro (wetting / non-wetting)
  - Capillary number  Ca = mu*v/sigma
  - Euler characteristic  chi = objects - loops + cavities (connectivity)
  - Ganglion flux fraction (disconnected-phase flux / total)

Note: this issue's PDF has a text layer but the typeset Corey equations were
dropped in extraction, so they are faithful standard-form reconstructions from
the complete nomenclature.  Fractions dimensionless; SI for Ca.
"""

import numpy as np


# ---------------------------------------------- Corey kr --------------

def corey_krw(sw, swc, sor, kwro, nw):
    """Corey wetting-phase relative permeability

        krw = kwro*((Sw - Swc)/(1 - Swc - Sor))^nw.
    """
    swn = np.clip((np.asarray(sw, float) - swc) / (1.0 - swc - sor), 0.0, 1.0)
    return kwro * swn ** nw


def corey_kro(sw, swc, sor, kocw, no):
    """Corey non-wetting-phase relative permeability

        kro = kocw*((1 - Sw - Sor)/(1 - Swc - Sor))^no.
    """
    son = np.clip((1.0 - np.asarray(sw, float) - sor) / (1.0 - swc - sor), 0.0, 1.0)
    return kocw * son ** no


def capillary_number(viscosity, velocity, ift):
    """Capillary number  Ca = mu*v/sigma (viscous / capillary force ratio)."""
    return viscosity * velocity / ift


# ---------------------------------------------- topology --------------

def euler_characteristic(objects, loops, cavities):
    """3D Euler characteristic  chi = objects - loops + cavities.

    More negative chi (more loops) means a more connected phase.
    """
    return objects - loops + cavities


def ganglion_flux_fraction(disconnected_flux, total_flux):
    """Fraction of non-wetting-phase flux carried by disconnected ganglia."""
    return np.clip(disconnected_flux / total_flux, 0.0, 1.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Flow Regimes During Immiscible Displacement")
    print("=" * 60)

    swc, sor, kwro, kocw = 0.1, 0.2, 0.4, 1.0
    sw = np.linspace(swc, 1 - sor, 12)
    krw = corey_krw(sw, swc, sor, kwro, 2.0)
    kro = corey_kro(sw, swc, sor, kocw, 3.0)
    # krw rises, kro falls; endpoints honour the residual saturations
    assert np.all(np.diff(krw) >= 0) and np.all(np.diff(kro) <= 0)
    assert np.isclose(krw[0], 0.0) and np.isclose(kro[-1], 0.0)
    assert np.isclose(krw[-1], kwro) and np.isclose(kro[0], kocw)

    # Capillary number
    ca = capillary_number(1e-3, 1e-4, 0.03)
    print(f"  capillary number       = {ca:.2e}")
    assert ca > 0

    # Euler characteristic: a ball (chi=1) is less connected than a torus (chi=0)
    assert euler_characteristic(1, 0, 0) == 1 and euler_characteristic(1, 1, 0) == 0

    # Ganglion flux fraction stays in [0, 1]
    g = ganglion_flux_fraction(3.0, 10.0)
    print(f"  ganglion flux fraction = {g:.2f}")
    assert np.isclose(g, 0.3)
    print("  PASS")
    return {"Ca": float(ca), "ganglion_fraction": float(g)}


if __name__ == "__main__":
    test_all()
