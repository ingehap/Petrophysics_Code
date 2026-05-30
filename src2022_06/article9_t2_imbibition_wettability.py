"""
Article 9: NMR-Based Wettability Index for Unconventional Rocks
Dick, Veselinovic, Bonnie, Kelly (2022)
DOI: 10.30632/PJV63N3-2022a9

Body text was not present in the available PDF extract, so this module
is a *methodology proxy* guided by the Guest Editor's summary:
classical Amott-Harvey, USBM and contact-angle wettability indices are
impractical on unconventional rocks.  A time-lapse T2 NMR method monitors
sequential water imbibition then oil imbibition; the relative T2-area
shifts define a wettability index validated by deuterated-water (D2O)
imbibition.

Implements:

  - Synthetic time-lapse T2 spectra for sequential water and oil
    imbibition into a mixed-wet rock.
  - Area-shift wettability index in [-1, +1] where +1 = strongly
    water-wet, -1 = strongly oil-wet:

        WI = (A_water_long - A_oil_long) / (A_water_long + A_oil_long)

    where A_*_long is the integrated long-T2 signal recovered after
    each imbibition stage.
  - D2O verification: the D2O-imbibition signal is the difference
    between the H2O-imbibed and D2O-imbibed spectra (D2O is invisible
    to 1H NMR), which isolates the protonated-phase contribution.
"""

import numpy as np


# ---------------------------------------------- synthetic T2 spectra -----

def synth_t2_spectrum(T2_axis_ms, centres_ms, amps, sigmas=None,
                     noise=0.005, seed=0):
    """Sum of log-Gaussian peaks on the T2 axis."""
    rng = np.random.default_rng(seed)
    if sigmas is None:
        sigmas = [0.30] * len(centres_ms)
    s = np.zeros_like(T2_axis_ms)
    for c, sig, a in zip(centres_ms, sigmas, amps):
        s += a * np.exp(-((np.log10(T2_axis_ms) - np.log10(c)) / sig) ** 2)
    s += noise * rng.standard_normal(len(T2_axis_ms))
    return np.clip(s, 0.0, None)


def simulate_imbibition(T2_axis_ms, initial_amps, final_long_amp,
                        long_centre_ms=300.0, sigma=0.30):
    """Add a long-T2 peak of amplitude final_long_amp on top of the
    initial spectrum to simulate the filled phase after imbibition."""
    base = synth_t2_spectrum(T2_axis_ms, initial_amps["centres"],
                             initial_amps["amps"], noise=0.0)
    add = final_long_amp * np.exp(-((np.log10(T2_axis_ms) - np.log10(long_centre_ms))
                                    / sigma) ** 2)
    return base + add


# ---------------------------------------------- wettability index ------

def long_t2_area(T2_axis_ms, spectrum, cutoff_ms=100.0):
    """Area integral of the T2 distribution beyond a long-T2 cutoff."""
    mask = T2_axis_ms >= cutoff_ms
    return float(spectrum[mask].sum())


def wettability_index(area_water_long, area_oil_long):
    """WI = (A_w - A_o) / (A_w + A_o) in [-1, +1]."""
    return float((area_water_long - area_oil_long) /
                 (area_water_long + area_oil_long + 1e-12))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: T2-Imbibition NMR Wettability Index (proxy)")
    print("=" * 60)

    T2 = np.logspace(-1, 3.5, 64)

    # Initial dry rock - small bound-water signal only
    initial = dict(centres=[2.0, 8.0], amps=[0.20, 0.10])

    # Water-wet rock: water imbibes more strongly than oil
    spec_after_water = simulate_imbibition(T2, initial, final_long_amp=0.80)
    spec_after_oil = simulate_imbibition(T2, initial, final_long_amp=0.20)

    A_w = long_t2_area(T2, spec_after_water)
    A_o = long_t2_area(T2, spec_after_oil)
    WI_ww = wettability_index(A_w, A_o)
    print(f"  Water-wet rock:  A_w = {A_w:5.2f}   A_o = {A_o:5.2f}   "
          f"WI = {WI_ww:+.3f}")

    # Oil-wet rock: oil imbibes more
    spec_after_water_ow = simulate_imbibition(T2, initial, final_long_amp=0.20)
    spec_after_oil_ow = simulate_imbibition(T2, initial, final_long_amp=0.85)
    A_w_ow = long_t2_area(T2, spec_after_water_ow)
    A_o_ow = long_t2_area(T2, spec_after_oil_ow)
    WI_ow = wettability_index(A_w_ow, A_o_ow)
    print(f"  Oil-wet rock:    A_w = {A_w_ow:5.2f}   A_o = {A_o_ow:5.2f}   "
          f"WI = {WI_ow:+.3f}")

    # D2O verification: D2O-imbibed spectrum has the water peak invisible,
    # so subtracting from H2O-imbibed isolates the water contribution.
    spec_water_h2o = spec_after_water
    spec_water_d2o = simulate_imbibition(T2, initial, final_long_amp=0.0)
    diff = spec_water_h2o - spec_water_d2o
    A_diff = long_t2_area(T2, diff)
    print(f"  D2O - H2O long-T2 isolated water signal = {A_diff:.3f}")

    assert WI_ww > 0.5, "Water-wet rock must give clearly positive WI"
    assert WI_ow < -0.5, "Oil-wet rock must give clearly negative WI"
    assert A_diff > 0.0, "Isolated water signal must be positive"
    print("  PASS")
    return {"WI_water_wet": WI_ww, "WI_oil_wet": WI_ow,
            "D2O_isolation_area": A_diff}


if __name__ == "__main__":
    test_all()
