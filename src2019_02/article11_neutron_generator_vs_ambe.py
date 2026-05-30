"""
Article 11: Neutron Generators as Alternatives to Am-Be Sources in Well Logging:
            An Assessment of Fundamentals
Badruzzaman, Schmidt, Antolak (2019)
DOI: 10.30632/PJV60N1Y2019a10

Chemical Am-Be neutron sources (continuous, ~4.5 MeV average energy) are being
replaced by electronic D-T neutron generators (pulsed, 14.1 MeV).  The higher
source energy lengthens the slowing-down length and shifts the porosity
sensitivity, while the pulsed operation and far higher output change the
statistics and depth of investigation.  This module compares the fundamentals.

Implements:

  - Source comparison (energy, output, on/off control)
  - Neutron slowing-down length and its energy dependence
  - Slowing-down-length-based neutron porosity sensitivity
  - Counting-statistics precision from source output

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard neutron-source / slowing-down fundamentals the
paper assesses.
"""

import numpy as np

SOURCES = {
    "Am-Be": {"energy_MeV": 4.5, "output_n_s": 2.2e7, "pulsed": False},
    "D-T":   {"energy_MeV": 14.1, "output_n_s": 1.0e8, "pulsed": True},
}


# ---------------------------------------------- slowing down ------------

def slowing_down_length(porosity, source_energy_MeV, Ls0=20.0, e_ref=4.5):
    """Neutron slowing-down length (cm): falls with porosity (more hydrogen),
    rises with source energy.

        Ls = Ls0*(1 - 0.6*phi)*sqrt(E/E_ref)
    """
    return Ls0 * (1.0 - 0.6 * np.asarray(porosity, float)) \
        * np.sqrt(source_energy_MeV / e_ref)


def apparent_neutron_porosity(Ls, source_energy_MeV, Ls0=20.0, e_ref=4.5):
    """Invert the slowing-down length for apparent porosity at a given source E."""
    return (1.0 - Ls / (Ls0 * np.sqrt(source_energy_MeV / e_ref))) / 0.6


def counting_precision(output_n_s, acquisition_s, efficiency=1e-4):
    """Relative counting precision  1/sqrt(N), N = output*efficiency*time."""
    N = output_n_s * efficiency * acquisition_s
    return 1.0 / np.sqrt(N)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 11: Neutron Generators vs Am-Be Sources")
    print("=" * 60)

    # Source fundamentals: the D-T generator is higher energy, higher output,
    # and can be switched off (pulsed)
    ambe, dt = SOURCES["Am-Be"], SOURCES["D-T"]
    assert dt["energy_MeV"] > ambe["energy_MeV"]
    assert dt["output_n_s"] > ambe["output_n_s"] and dt["pulsed"]

    # Slowing-down length falls with porosity (more hydrogen moderates faster)
    ls_lo = slowing_down_length(0.05, 4.5)
    ls_hi = slowing_down_length(0.30, 4.5)
    print(f"  Ls at 5%/30% porosity  = {ls_lo:.1f} / {ls_hi:.1f} cm")
    assert ls_lo > ls_hi

    # Higher source energy (D-T) lengthens the slowing-down length at fixed phi
    ls_ambe = slowing_down_length(0.15, ambe["energy_MeV"])
    ls_dt = slowing_down_length(0.15, dt["energy_MeV"])
    print(f"  Ls Am-Be / D-T (15%)   = {ls_ambe:.1f} / {ls_dt:.1f} cm")
    assert ls_dt > ls_ambe

    # Porosity inversion round-trips for a given source energy
    phi_back = apparent_neutron_porosity(ls_dt, dt["energy_MeV"])
    assert abs(phi_back - 0.15) < 1e-9
    # ignoring the energy difference (using the Am-Be reference on a D-T Ls)
    # biases the apparent porosity -> requires source-specific calibration
    phi_wrong = apparent_neutron_porosity(ls_dt, ambe["energy_MeV"])
    assert phi_wrong < 0.15

    # The high-output D-T generator gives far better counting precision
    prec_ambe = counting_precision(ambe["output_n_s"], 1.0)
    prec_dt = counting_precision(dt["output_n_s"], 1.0)
    print(f"  precision Am-Be / D-T  = {prec_ambe:.3f} / {prec_dt:.3f}")
    assert prec_dt < prec_ambe
    print("  PASS")
    return {"Ls_DT_15": float(ls_dt), "precision_DT": float(prec_dt)}


if __name__ == "__main__":
    test_all()
