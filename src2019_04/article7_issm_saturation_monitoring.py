"""
Article 7: In-Situ Saturation Monitoring (ISSM) - Recommendations for Improved
           Processing
Reed, Cense (2019)
DOI: 10.30632/PJV60N2-2019a5

In-situ saturation monitoring uses X-ray or gamma-ray attenuation to track fluid
saturations during a coreflood.  The transmitted intensity follows the
Beer-Lambert law; with dry and fully-saturated calibration scans, the measured
attenuation maps linearly to water saturation.  Dual-energy acquisition
separates two fluids.

Implements:

  - Beer-Lambert attenuation  I = I0*exp(-mu*x)
  - Linear attenuation from intensity  mu = ln(I0/I)/x
  - Water saturation from attenuation between dry and saturated calibrations
  - Dual-energy two-fluid saturation solve

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the attenuation / saturation-monitoring relations the paper recommends.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Beer-Lambert ------------

def beer_lambert(I0, mu, x):
    """Transmitted intensity  I = I0*exp(-mu*x)."""
    return petrolib.nuclear.beer_lambert(I0, mu, x)


def attenuation(I0, I, x):
    """Linear attenuation coefficient  mu = ln(I0/I)/x."""
    return petrolib.nuclear.mu_from_intensity(I0, I, x)


# ---------------------------------------------- saturation --------------

def water_saturation(mu, mu_dry, mu_wet):
    """Water saturation from attenuation between dry and saturated calibrations.

        Sw = (mu - mu_dry)/(mu_wet - mu_dry)
    mu_dry = 100% non-wetting (e.g. oil/gas) scan, mu_wet = 100% water scan.
    """
    return np.clip((np.asarray(mu, float) - mu_dry) / (mu_wet - mu_dry), 0.0, 1.0)


def dual_energy_saturation(mu_low, mu_high, end_low, end_high):
    """Solve two-fluid volume fractions from dual-energy attenuations.

    end_low / end_high are 2x2 matrices [[mu_fluidA, mu_fluidB]] of the pure-fluid
    attenuations at the low and high energies; returns (fA, fB) with fA+fB=1.
    Here we use the low-energy equation plus closure for a quick two-fluid solve.
    """
    a1, b1 = end_low
    # mu_low = fA*a1 + fB*b1, fA + fB = 1  ->  fA = (mu_low - b1)/(a1 - b1)
    fA = (mu_low - b1) / (a1 - b1)
    return float(np.clip(fA, 0.0, 1.0)), float(np.clip(1.0 - fA, 0.0, 1.0))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: In-Situ Saturation Monitoring (ISSM)")
    print("=" * 60)

    # Beer-Lambert round-trip
    I0, mu, x = 1.0, 40.0, 0.025
    I = beer_lambert(I0, mu, x)
    assert abs(attenuation(I0, I, x) - mu) < 1e-9
    assert beer_lambert(I0, 60.0, x) < I          # denser -> less transmission

    # Saturation calibration: dry -> 0, wet -> 1, mid -> 0.5
    mu_dry, mu_wet = 30.0, 50.0
    assert abs(water_saturation(mu_dry, mu_dry, mu_wet)) < 1e-9
    assert abs(water_saturation(mu_wet, mu_dry, mu_wet) - 1.0) < 1e-9
    sw_mid = water_saturation(40.0, mu_dry, mu_wet)
    print(f"  Sw at mid attenuation  = {sw_mid:.2f}")
    assert abs(sw_mid - 0.5) < 1e-9

    # Saturation during a waterflood: attenuation rises as water replaces oil
    mu_series = np.array([31.0, 38.0, 45.0, 49.0])
    sw_series = water_saturation(mu_series, mu_dry, mu_wet)
    print(f"  Sw series              = {np.array2string(sw_series, precision=2)}")
    assert np.all(np.diff(sw_series) > 0)

    # Dual-energy two-fluid solve recovers a planted water fraction
    fw_true = 0.6
    mu_water, mu_oil = 50.0, 30.0
    mu_meas = fw_true * mu_water + (1 - fw_true) * mu_oil
    fw, fo = dual_energy_saturation(mu_meas, None, (mu_water, mu_oil), None)
    print(f"  dual-energy fw / fo    = {fw:.2f} / {fo:.2f}")
    assert abs(fw - fw_true) < 1e-9 and abs(fw + fo - 1.0) < 1e-9
    print("  PASS")
    return {"sw_mid": sw_mid, "fw_dual": fw}


if __name__ == "__main__":
    test_all()
