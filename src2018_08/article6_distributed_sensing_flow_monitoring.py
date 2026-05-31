"""
Article 6: Production Monitoring Using Next-Generation Distributed Sensing
           Systems
Naldrett, Cerrahoglu, Mahue (2018)
DOI: 10.30632/PJV59V4-2018a5

Distributed fiber-optic sensing measures temperature (DTS) and acoustics (DAS)
continuously along the wellbore.  A thermal-mixing energy balance at an inflow
converts the local temperature change into a flow-rate split between zones; the
Joule-Thomson effect ties a pressure drop to a temperature change; and the
acoustic gauge length sets the maximum detectable frequency, while the measured
sound speed types the inflowing fluid (gas drops the sound speed sharply).

Implements:

  - Thermal mixing temperature  Tmix = sum(w*c*T)/sum(w*c)
  - Flow split from a two-zone mixing temperature
  - Joule-Thomson temperature change  dT = JTC*dP
  - Max detectable frequency  fmax = c/(2*GL)
  - Fluid typing from the acoustic sound speed

Note: this issue's PDF has a text layer but its typeset display-equation glyphs
were dropped in extraction, so the mixing / Joule-Thomson / gauge-length
relations (Eqs. 3-5) are faithful standard-form reconstructions; the worked
gauge-length numbers (water 1500 m/s, GL 10 m -> 75 Hz) are reproduced.  SI units.
"""

import numpy as np


# ---------------------------------------------- thermal --------------

def mixing_temperature(mass_rates, heat_caps, temps):
    """Energy-balance mixing temperature  Tmix = sum(w*c*T)/sum(w*c)  (Eq. 3)."""
    w = np.asarray(mass_rates, float)
    c = np.asarray(heat_caps, float)
    t = np.asarray(temps, float)
    return float((w * c * t).sum() / (w * c).sum())


def flow_split_from_mixing(t_mix, t1, t2, c1, c2):
    """Mass-flow fraction of zone 1 from a two-zone mixing temperature.

    Inverts the energy balance w1*c1*(T1-Tmix) = w2*c2*(Tmix-T2) for the
    fraction f = w1/(w1+w2).  (Non-unique in true multiphase flow because water
    carries ~2x the heat capacity of oil - hence c1, c2 must be supplied.)
    """
    num = c2 * (t_mix - t2)
    den = c1 * (t1 - t_mix) + c2 * (t_mix - t2)
    return num / den


def joule_thomson_dT(jtc, dp):
    """Joule-Thomson temperature change  dT = JTC*dP  (Eq. 4)."""
    return jtc * np.asarray(dp, float)


# ---------------------------------------------- acoustic --------------

def max_detectable_frequency(sound_speed, gauge_length):
    """Max detectable frequency  fmax = c/(2*GL)  (Eq. 5).

    The gauge length GL sets the shortest resolvable wavelength (2*GL); a slower
    medium (gas) lowers fmax.
    """
    return np.asarray(sound_speed, float) / (2.0 * gauge_length)


def fluid_from_sound_speed(c):
    """Type the inflowing fluid from the measured acoustic sound speed (m/s).

    Water ~ 1500, oil ~ 1300; a small gas fraction collapses the sound speed
    below ~200 m/s, making it a very sensitive gas indicator.
    """
    if c < 200.0:
        return "gas"
    if c < 1400.0:
        return "oil"
    return "water"


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Distributed Sensing Flow Monitoring")
    print("=" * 60)

    # Mixing temperature lies between the inflow temperatures
    tmix = mixing_temperature([2.0, 1.0], [4180.0, 2000.0], [80.0, 110.0])
    print(f"  mixing temperature     = {tmix:.2f} C")
    assert 80.0 < tmix < 110.0

    # Recover the planted mass-flow split (w1=2, w2=1 -> f = 2/3) from the
    # mixing temperature, given the two heat capacities
    f = flow_split_from_mixing(tmix, 80.0, 110.0, 4180.0, 2000.0)
    print(f"  zone-1 flow fraction   = {f:.3f}")
    assert np.isclose(f, 2.0 / (2.0 + 1.0))

    # Joule-Thomson: a pressure drop cools/heats linearly
    assert np.isclose(joule_thomson_dT(0.05, 200.0), 10.0)

    # Gauge length: water at 1500 m/s, GL 10 m -> 75 Hz; gas at 350 -> 17.5 Hz
    print(f"  fmax water/gas         = {max_detectable_frequency(1500.0, 10.0):.1f}"
          f" / {max_detectable_frequency(350.0, 10.0):.1f} Hz")
    assert np.isclose(max_detectable_frequency(1500.0, 10.0), 75.0)
    assert np.isclose(max_detectable_frequency(350.0, 10.0), 17.5)

    # Fluid typing thresholds
    assert (fluid_from_sound_speed(1500.0), fluid_from_sound_speed(1300.0),
            fluid_from_sound_speed(150.0)) == ("water", "oil", "gas")
    print("  PASS")
    return {"t_mix": tmix, "flow_fraction": float(f)}


if __name__ == "__main__":
    test_all()
