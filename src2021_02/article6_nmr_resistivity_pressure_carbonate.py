"""
Article 6: Formation Evaluation With NMR, Resistivity, and Pressure Data -
           A Case Study of a Carbonate Oil Field Offshore West Africa
Li, Drinkwater, Whittlesey, Condon (2021)
DOI: 10.30632/PJV62N1-2021a5

An integrated case study tying together three independent measurements in a
carbonate oil field: NMR (porosity and permeability), resistivity (Archie water
saturation), and formation pressure (fluid density and fluid contacts from
pretest gradients).  The modules implement the quantitative relations the
workflow relies on and cross-check them against one another.

Implements:

  - Archie water saturation  Sw = (a Rw / (phi^m Rt))^(1/n)
  - NMR permeability: Timur-Coates and SDR
  - Bulk volume water / Buckles product
  - Formation-pressure fluid density  rho = (dP/dz) / g
  - Fluid contact from intersecting pressure gradients (e.g., OWC)

Note: this issue's source PDF has no usable text layer, so the formulas are
faithful standard-form reconstructions of the classic Archie / NMR /
pressure-gradient relations the case study integrates.  Depths in metres,
pressures in bar, T2 in ms.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

G_ACCEL = 9.81           # m/s^2


# ---------------------------------------------- resistivity -------------

def archie_sw(Rt, Rw, phi, a=1.0, m=2.0, n=2.0):
    """Archie water saturation  Sw = (a Rw / (phi^m Rt))^(1/n)."""
    return petrolib.saturation_resistivity.archie_sw(Rt, Rw, phi=phi, a=a, m=m, n=n)


def formation_factor(phi, a=1.0, m=2.0):
    """Archie formation factor  F = a / phi^m."""
    return petrolib.saturation_resistivity.formation_factor(phi, a=a, m=m)


# ---------------------------------------------- NMR permeability --------

def timur_coates(phi, ffi, bvi, C=10.0):
    """Timur-Coates permeability  k = (phi/C)^4 (FFI/BVI)^2  (mD)."""
    # This copy reports in mD scaled by 1e6 (the unit adapter stays local).
    return petrolib.nmr.timur_coates(phi, ffi, bvi, C=C) * 1e6


def sdr(phi, t2lm_ms, a=4.0, m=4.0, n=2.0):
    """SDR permeability  k = a phi^m T2LM^n  (mD).  T2LM in ms."""
    return petrolib.nmr.sdr(phi, t2lm_ms, a=a, m=m, n=n)


def bulk_volume_water(phi, sw):
    """Bulk volume water  BVW = phi * Sw (constant in a Buckles-law zone)."""
    return petrolib.saturation_resistivity.bulk_volume_water(phi, sw)


# ---------------------------------------------- pressure ----------------

def gradient_density(dP_dz_bar_per_m):
    """Fluid density (kg/m^3) from a pressure gradient  rho = (dP/dz)/g.

    dP/dz in bar/m (1 bar = 1e5 Pa).
    """
    return dP_dz_bar_per_m * 1e5 / G_ACCEL


def fit_gradient(depth, pressure):
    """Least-squares (dP/dz, P0) for pressure vs depth.  Returns (slope, intercept)."""
    return petrolib.geochem_fluids.gradients.fit_pressure_gradient(depth, pressure)


def fluid_contact(depth_a, press_a, depth_b, press_b):
    """Depth where two fluid pressure gradients intersect (a fluid contact).

    Fits a line to each fluid's pretest points (e.g., oil above, water below)
    and returns the crossover depth = the oil-water (or gas-oil) contact.
    """
    return petrolib.geochem_fluids.gradients.fluid_contact(depth_a, press_a, depth_b, press_b)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: NMR + Resistivity + Pressure (Carbonate)")
    print("=" * 60)

    # Archie in the oil leg: high Rt -> low Sw
    sw_oil = archie_sw(Rt=50.0, Rw=0.05, phi=0.22)
    sw_wet = archie_sw(Rt=2.0, Rw=0.05, phi=0.22)
    print(f"  Sw oil / water leg     = {sw_oil:.3f} / {sw_wet:.3f}")
    assert sw_oil < 0.4 < sw_wet

    # NMR permeability from a porous interval
    k_tc = timur_coates(0.22, ffi=0.16, bvi=0.06)
    k_sdr = sdr(0.22, t2lm_ms=120.0)
    print(f"  k Timur-Coates / SDR   = {k_tc:.1f} / {k_sdr:.1f} mD")
    assert k_tc > 0 and k_sdr > 0

    # Buckles: BVW constant at irreducible saturation
    bvw = bulk_volume_water(0.22, sw_oil)
    print(f"  bulk volume water      = {bvw:.3f}")
    assert 0 < bvw < 0.1

    # Pressure gradients: oil ~0.7 g/cc, water ~1.0 g/cc densities recovered
    rho_oil = gradient_density(0.0687)        # ~0.70 sg
    rho_wat = gradient_density(0.0981)        # ~1.00 sg
    print(f"  density oil / water    = {rho_oil:.0f} / {rho_wat:.0f} kg/m^3")
    assert 600 < rho_oil < 800 and 950 < rho_wat < 1050

    # Fluid contact from two synthetic pretest gradients (planted OWC = 3120 m)
    owc_true = 3120.0
    P_owc = 320.0                              # bar at the contact
    z_oil = np.array([3050.0, 3080.0, 3110.0])
    z_wat = np.array([3130.0, 3160.0, 3190.0])
    P_oil = P_owc + 0.0687 * (z_oil - owc_true)
    P_wat = P_owc + 0.0981 * (z_wat - owc_true)
    owc = fluid_contact(z_oil, P_oil, z_wat, P_wat)
    print(f"  recovered OWC          = {owc:.1f} m  (true {owc_true:.0f})")
    assert abs(owc - owc_true) < 1.0
    print("  PASS")
    return {"sw_oil": sw_oil, "k_tc": k_tc, "owc": owc,
            "rho_oil": rho_oil, "rho_wat": rho_wat}


if __name__ == "__main__":
    test_all()
