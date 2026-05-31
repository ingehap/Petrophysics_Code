"""
Article 4 (Technical Note): The Bateman-Konen Resistivity-Salinity Transform
Kennedy (2015)
Reference: Petrophysics Vol. 56, No. 3 (June 2015), pp. 282-283
DOI: none assigned (this issue predates SPWLA DOI assignment)

A technical note attributing the salinity-resistivity-temperature transform to
Bateman & Konen (1977) and explaining its form.  At 75 deg F the brine
resistivity is  R75 = 0.0123 + 3647.5/C^0.955  for NaCl concentration C (ppm);
the small additive 0.0123 term reproduces the slight upturn of the chart at low
concentration.  Resistivities are carried to other temperatures by the Arps
(1953) formula.  The note also frames the transform alongside the modified
formation-factor power law  F = b + a/phi^m  (the additive b term lets a power
law curve on the high-porosity side).

Implements:

  - Bateman-Konen brine resistivity at 75 deg F from NaCl concentration (Eq. 1)
  - Inverse transform: NaCl concentration from R75 (Eqs. 2-3)
  - Arps temperature conversion of brine resistivity
  - Rw at any temperature from salinity; modified formation-factor power law

Note: this issue's PDF has a text layer and this technical note's formula is
transcribed directly from the body.  Resistivity in ohm-m, temperature in deg F,
concentration in ppm NaCl.
"""

import numpy as np

ARPS_C_FAHRENHEIT = 6.77      # Arps constant for deg F


# ---------------------------------------------- Bateman-Konen --------------

def bateman_konen_rw75(salinity_ppm):
    """Brine resistivity at 75 deg F (Bateman & Konen, 1977; Eq. 1)

        R75 = 0.0123 + 3647.5/C^0.955,

    with C the NaCl concentration in ppm.  The 0.0123 term reproduces the chart's
    low-concentration upturn.
    """
    return 0.0123 + 3647.5 / np.asarray(salinity_ppm, float) ** 0.955


def bateman_konen_salinity(r75):
    """NaCl concentration from R75 (inverse transform, Eqs. 2-3)

        C = (3647.5/(R75 - 0.0123))^(1/0.955)   [ppm].
    """
    return (3647.5 / (np.asarray(r75, float) - 0.0123)) ** (1.0 / 0.955)


# ---------------------------------------------- temperature --------------

def arps_temperature_conversion(r75, temperature_f):
    """Arps (1953) temperature conversion from 75 deg F

        R(T) = R75*(75 + 6.77)/(T + 6.77) = R75*81.77/(T + 6.77).
    """
    return r75 * 81.77 / (temperature_f + ARPS_C_FAHRENHEIT)


def rw_from_salinity(salinity_ppm, temperature_f):
    """Brine resistivity at temperature from salinity (Bateman-Konen + Arps)

        Rw(T) = (0.0123 + 3647.5/C^0.955) * 81.77/(T + 6.77).
    """
    return arps_temperature_conversion(bateman_konen_rw75(salinity_ppm), temperature_f)


# ---------------------------------------------- provenance power law --------------

def modified_formation_factor(phi, a=1.0, m=2.0, b=0.0):
    """Modified formation-factor power law discussed in the note

        F = b + a/phi^m,

    where the additive b term (b = 0 recovers Archie) lets the function curve on
    the high-porosity side - the same additive-constant idea as the 0.0123 in
    the Bateman-Konen formula.
    """
    return b + a / np.asarray(phi, float) ** m


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4 (TN): Bateman-Konen Resistivity-Salinity")
    print("=" * 60)

    # R75 decreases with increasing salinity and stays above the 0.0123 floor
    r_lo = bateman_konen_rw75(20000.0)
    r_hi = bateman_konen_rw75(150000.0)
    print(f"  R75 20k/150k ppm       = {r_lo:.4f} / {r_hi:.4f} ohm-m")
    assert r_lo > r_hi > 0.0123

    # Forward/inverse transforms are mutual inverses
    c = 55000.0
    r75 = bateman_konen_rw75(c)
    assert np.isclose(bateman_konen_salinity(r75), c, rtol=1e-6)

    # Arps: resistivity falls as temperature rises (and equals R75 at 75 F)
    assert np.isclose(arps_temperature_conversion(r75, 75.0), r75)
    assert arps_temperature_conversion(r75, 200.0) < r75

    # Rw at temperature from salinity matches the chained transform
    rw = rw_from_salinity(c, 180.0)
    print(f"  Rw (55k ppm, 180F)     = {rw:.4f} ohm-m")
    assert np.isclose(rw, r75 * 81.77 / (180.0 + 6.77))

    # Modified power law reduces to Archie when b = 0 and curves up when b > 0
    assert np.isclose(modified_formation_factor(0.2, a=1.0, m=2.0, b=0.0), 1.0 / 0.2 ** 2)
    assert modified_formation_factor(0.2, 1.0, 2.0, b=5.0) > modified_formation_factor(0.2, 1.0, 2.0, 0.0)
    print("  PASS")
    return {"R75_55k": float(r75), "salinity_back": float(bateman_konen_salinity(r75)), "Rw_180F": float(rw)}


if __name__ == "__main__":
    test_all()
