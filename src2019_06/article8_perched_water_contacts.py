"""
Article 8: Perched Water Contacts: Understanding Fundamental Controls
Hulea (2019)
DOI: 10.30632/PJV60N3-2019a7

A perched water contact is a local water accumulation held above the regional
free-water level by a capillary barrier (a low-permeability, high-entry-pressure
layer).  The fundamental controls are the saturation-height (capillary-pressure)
function and the entry pressure of the barrier: the rock can hold water above
the free-water level wherever the buoyancy pressure stays below the local
capillary entry pressure.

Implements:

  - Buoyancy capillary pressure vs height  Pc = (rho_w - rho_hc)*g*h
  - Brooks-Corey saturation-height function
  - Free-water level (Pc = 0) and the perching capillary barrier
  - Perched-water column held by a barrier's entry pressure

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the journal metadata and these are faithful
standard-form reconstructions of the capillary / saturation-height controls the
paper analyzes.  SI units; h in m, Pc in Pa.
"""

import numpy as np

G_ACCEL = 9.81


# ---------------------------------------------- capillary --------------

def buoyancy_pc(height_above_fwl, rho_w=1000.0, rho_hc=700.0):
    """Capillary (buoyancy) pressure at a height above the free-water level.

        Pc = (rho_w - rho_hc)*g*h
    """
    return (rho_w - rho_hc) * G_ACCEL * np.asarray(height_above_fwl, float)


def brooks_corey_sw(pc, pe, lam, swr=0.1):
    """Brooks-Corey saturation-height function.

        Sw = Swr + (1 - Swr)*(Pe/Pc)^lambda   for Pc >= Pe, else Sw = 1
    """
    pc = np.asarray(pc, float)
    sw = np.where(pc <= pe, 1.0, swr + (1.0 - swr) * (pe / pc) ** lam)
    return np.clip(sw, swr, 1.0)


def entry_height(pe, rho_w=1000.0, rho_hc=700.0):
    """Height above FWL at which buoyancy pressure equals the entry pressure Pe."""
    return pe / ((rho_w - rho_hc) * G_ACCEL)


def is_perched(barrier_pe, buoyancy_at_barrier):
    """A barrier perches water if its entry pressure exceeds the local buoyancy."""
    return barrier_pe > buoyancy_at_barrier


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Perched Water Contacts")
    print("=" * 60)

    # Buoyancy pressure grows linearly with height above the free-water level
    assert abs(buoyancy_pc(0.0)) < 1e-12          # Pc = 0 at the FWL
    assert buoyancy_pc(50.0) > buoyancy_pc(10.0)

    # Saturation-height: Sw = 1 below the entry height, then drops with height
    pe = 5e4                                       # entry pressure (Pa)
    h = np.array([5.0, 20.0, 50.0, 100.0])
    pc = buoyancy_pc(h)
    sw = brooks_corey_sw(pc, pe, lam=1.5)
    print(f"  Sw vs height           = {np.array2string(sw, precision=2)}")
    assert sw[0] == 1.0                            # below entry height -> 100% water
    assert np.all(np.diff(sw[1:]) < 0)             # drains upward

    # Entry height: the transition where buoyancy first exceeds the entry pressure
    he = entry_height(pe)
    print(f"  entry height           = {he:.1f} m")
    assert abs(buoyancy_pc(he) - pe) < 1e-6

    # A high-entry-pressure barrier perches water above the regional FWL; a
    # low-entry-pressure layer does not
    buoyancy = buoyancy_pc(30.0)                   # buoyancy at the barrier depth
    print(f"  buoyancy @30 m         = {buoyancy:.0f} Pa")
    assert is_perched(barrier_pe=2e5, buoyancy_at_barrier=buoyancy)
    assert not is_perched(barrier_pe=1e4, buoyancy_at_barrier=buoyancy)
    print("  PASS")
    return {"entry_height_m": float(he), "sw_profile": sw.tolist()}


if __name__ == "__main__":
    test_all()
