"""
Article 2: More Accurate Quantification of Free and Adsorbed Gas in Shale
           Reservoirs
Ansari, Merletti, Gramin, Armitage (2019)
DOI: 10.30632/PJV60N5-2019a2

Gas-in-place in shale is the sum of free gas (in the pore space) and adsorbed
gas (on the kerogen/clay surface).  Free gas comes from porosity, water
saturation and the gas formation volume factor; adsorbed gas follows a Langmuir
isotherm with a Gibbs adsorbed-phase-density correction; and the pore volume
occupied by the adsorbed monolayer must be removed from the free-gas porosity to
avoid double counting.

Implements:

  - Free gas content  G_free = phi*(1 - Sw)/Bg                    (Eq. a)
  - Langmuir adsorbed gas  Gc = rho_b*VL*P/(PL + P)               (Eq. b)
  - Gibbs adsorbed-phase-density correction                      (Eq. c)
  - Adsorbed-monolayer porosity correction (avoid double counting)
  - Total gas-in-place = free + adsorbed

Note: this issue's PDF body was font-garbled in extraction (numeric benchmarks
largely lost) and the equations are lettered; these are the standard Langmuir /
free-gas forms anchored to the recovered variable definitions.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- free gas ----------------

def free_gas(phi, sw, Bg):
    """Free gas content  G_free = phi*(1 - Sw)/Bg  (Eq. a).

    Bg = gas formation volume factor (reservoir->surface volume ratio).
    """
    return petrolib.geochem_fluids.adsorption.free_gas(phi, sw, Bg)


# ---------------------------------------------- adsorbed gas ------------

def langmuir(rho_b, VL, PL, P):
    """Langmuir adsorbed-gas capacity  Gc = rho_b*VL*P/(PL + P)  (Eq. b).

    VL = Langmuir volume (max capacity), PL = Langmuir pressure (half-capacity
    pressure), rho_b = bulk density.
    """
    return petrolib.geochem_fluids.adsorption.langmuir(P, VL, PL, rho_b=rho_b)


def gibbs_correction(gc, rho_free_gas, rho_adsorbed):
    """Gibbs adsorbed-phase-density correction  Gc' = Gc*(1 - rho_free/rho_ads)  (Eq. c).

    The adsorbed phase is denser than free gas, so the excess (Gibbs) adsorption
    measured is less than the absolute adsorption.
    """
    return petrolib.geochem_fluids.adsorption.gibbs_excess(gc, rho_free_gas, rho_adsorbed)


def monolayer_porosity_correction(phi, VL, rho_b, rho_adsorbed):
    """Remove the adsorbed-monolayer pore volume from porosity (avoid double count).

    Adsorbed-phase volume per rock = (rho_b*VL)/rho_adsorbed; subtract from phi.
    """
    return phi - rho_b * VL / rho_adsorbed


def total_gas_in_place(g_free, g_adsorbed):
    """Total gas-in-place = free + adsorbed."""
    return g_free + g_adsorbed


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Free & Adsorbed Gas in Shale")
    print("=" * 60)

    # Free gas rises with porosity and hydrocarbon saturation
    assert free_gas(0.10, 0.3, 0.005) > free_gas(0.05, 0.3, 0.005)
    assert free_gas(0.10, 0.2, 0.005) > free_gas(0.10, 0.5, 0.005)

    # Langmuir isotherm: half the Langmuir volume at P = PL, saturating at high P
    rho_b, VL, PL = 2.45, 0.005, 1500.0       # VL in volume/mass units
    g_half = langmuir(rho_b, VL, PL, PL)
    g_sat = langmuir(rho_b, VL, PL, 1e6)
    print(f"  Gc at PL / saturation  = {g_half:.4f} / {g_sat:.4f}")
    assert abs(g_half - rho_b * VL / 2.0) < 1e-9
    assert abs(g_sat - rho_b * VL) < 1e-3      # plateau at rho_b*VL
    # monotonically increasing with pressure
    P = np.array([500.0, 1500.0, 3000.0, 6000.0])
    assert np.all(np.diff(langmuir(rho_b, VL, PL, P)) > 0)

    # Gibbs correction reduces the apparent adsorbed gas
    gc = langmuir(rho_b, VL, PL, 4000.0)
    assert gibbs_correction(gc, 0.20, 0.34) < gc

    # Monolayer correction reduces the porosity used for free gas
    assert monolayer_porosity_correction(0.08, VL, rho_b, 0.34) < 0.08

    # Total GIP is the sum of both
    gip = total_gas_in_place(free_gas(0.08, 0.3, 0.005), gc)
    print(f"  total gas-in-place     = {gip:.3f}")
    assert gip > 0
    print("  PASS")
    return {"Gc_PL": float(g_half), "Gc_sat": float(g_sat), "GIP": float(gip)}


if __name__ == "__main__":
    test_all()
