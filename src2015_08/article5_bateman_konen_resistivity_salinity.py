"""
Article 5 (Technical Note): The Bateman-Konen Resistivity-Salinity Transform II
Kennedy (2015)
Reference: Petrophysics Vol. 56, No. 4 (August 2015), pp. 379-381
DOI: none assigned (this issue predates SPWLA DOI assignment)

A technical note revisiting the Bateman-Konen analytic transform between
formation-water resistivity (Rw), NaCl-equivalent salinity and temperature - the
closed-form fit to the classic Rw-salinity-temperature chart.  This module
implements the standard resistivity-salinity-temperature conversions the note
concerns: the Arps temperature correction and the salinity<->Rw transform, plus
the apparent-water-resistivity (Rwa) and Archie water-saturation use.

Implements:

  - Arps temperature conversion of resistivity
  - Rw from NaCl-equivalent salinity and temperature (and the inverse)
  - Apparent water resistivity Rwa = Rt*phi^m
  - Archie water saturation

Note: this technical note's body was beyond the PDF text extraction for this
issue (the source text truncates within the preceding article), so this module
is a methodology proxy implementing the standard resistivity-salinity-temperature
transforms it concerns, consistent with how other truncated items are handled in
this repository.  Resistivity in ohm-m, temperature in deg F, salinity in ppm
NaCl-equivalent.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

ARPS_C_FAHRENHEIT = 6.77      # Arps constant for deg F (use 21.5 for deg C)


# ---------------------------------------------- temperature conversion --------------

def arps_temperature_conversion(r1, t1, t2, c=ARPS_C_FAHRENHEIT):
    """Arps resistivity temperature conversion

        R2 = R1*(T1 + c)/(T2 + c),

    converting a resistivity measured at T1 to its value at T2 (c = 6.77 for
    deg F, 21.5 for deg C).
    """
    return r1 * (t1 + c) / (t2 + c)


# ---------------------------------------------- salinity <-> Rw --------------

def rw_from_salinity(salinity_ppm, temperature_f):
    """Formation-water resistivity from NaCl-equivalent salinity and temperature

        Rw = (0.0123 + 3647.5/C^0.955) * 81.77/(T + 6.77),

    the standard chart fit (the Bateman-Konen transform represents this
    salinity-resistivity-temperature relationship analytically).
    """
    return petrolib.geochem_fluids.brine.rw_from_salinity(salinity_ppm, temperature_f, unit="F")


def salinity_from_rw(rw, temperature_f):
    """NaCl-equivalent salinity from Rw and temperature (inverse transform)

        Rw75 = Rw*(T + 6.77)/81.77,
        C    = (3647.5/(Rw75 - 0.0123))^(1/0.955)   [ppm].
    """
    return petrolib.geochem_fluids.brine.salinity_from_rw(rw, temperature_f, unit="F")


# ---------------------------------------------- Archie use --------------

def apparent_water_resistivity(rt, phi, m=2.0, a=1.0):
    """Apparent water resistivity  Rwa = Rt*phi^m/a  (= Rw at Sw = 1)."""
    return petrolib.saturation_resistivity.apparent_water_resistivity(rt, phi, a=a, m=m)


def archie_sw(rt, rw, phi, m=2.0, n=2.0, a=1.0):
    """Archie water saturation  Sw = (a*Rw/(phi^m*Rt))^(1/n)."""
    return petrolib.saturation_resistivity.archie_sw(rt, rw, phi=phi, a=a, m=m, n=n)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5 (TN): Bateman-Konen Resistivity-Salinity")
    print("=" * 60)

    # Resistivity decreases as temperature increases (Arps)
    r200 = arps_temperature_conversion(0.1, 75.0, 200.0)
    print(f"  Rw 75F->200F           = 0.100 -> {r200:.4f} ohm-m")
    assert r200 < 0.1

    # Rw decreases with increasing salinity and increasing temperature
    rw_lo = rw_from_salinity(20000.0, 150.0)
    rw_hi = rw_from_salinity(80000.0, 150.0)
    print(f"  Rw 20k/80k ppm @150F   = {rw_lo:.4f} / {rw_hi:.4f} ohm-m")
    assert rw_lo > rw_hi > 0
    assert rw_from_salinity(50000.0, 200.0) < rw_from_salinity(50000.0, 100.0)

    # Salinity<->Rw transforms are mutual inverses
    sal = 55000.0
    rw = rw_from_salinity(sal, 180.0)
    sal_back = salinity_from_rw(rw, 180.0)
    print(f"  salinity round-trip    = {sal:.0f} -> {sal_back:.0f} ppm")
    assert np.isclose(sal_back, sal, rtol=1e-3)

    # Rwa equals Rw when the formation is fully water-saturated (Sw = 1)
    phi, m = 0.2, 2.0
    rt = rw / phi ** m                         # R0 at Sw = 1
    assert np.isclose(apparent_water_resistivity(rt, phi, m), rw)
    assert np.isclose(archie_sw(rt, rw, phi, m), 1.0)
    print("  PASS")
    return {"Rw_50k_150F": float(rw_from_salinity(50000.0, 150.0)), "salinity_back": float(sal_back)}


if __name__ == "__main__":
    test_all()
