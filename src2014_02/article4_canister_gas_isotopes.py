"""
Article 4: Desorbed Canister Gas Sampling and Gas Isotopic Analysis Procedures
           and Practices: A Case Study of Two Coalbed Methane Wells from the
           Lower Saxony Basin, Germany
Russell W. Spears, Sascha Alles, Alexey Makhonin (2014)
Reference: Petrophysics Vol. 55, No. 1 (February 2014), pp. 38-50
DOI: none assigned (this issue predates SPWLA DOI assignment)

A coalbed-methane case study on canister desorption sampling and gas isotopic
analysis.  Total gas content is the lost gas (back-extrapolated by the USBM
square-root-of-time method) plus the measured desorbed and residual gas; the
samples are quality-controlled for air contamination and gas chromatograph
signal, and the gas origin is classified from carbon-isotope ratios.

Implements:

  - Air-contamination ("airfree") correction using the atmospheric N2:O2 = 3.73
  - USBM square-root-of-time lost-gas estimate and total gas content
  - Isotope and GC quality-control checks (delta-13C limits, CH4 peak area)
  - Biogenic / thermogenic gas-origin classification from delta-13C of methane

Note: this procedural paper renders no display equations; the air-correction
ratio (3.73:1), the delta-13C validity limits (CH4 > -20 permil and CO2 > +20
permil do not occur naturally), the 50 mV-sec CH4 peak-area cutoff and the USBM
square-root-of-time lost-gas method are transcribed from the text.  Gas volumes
in cm^3, isotopes in permil.
"""

import numpy as np

ATM_N2_O2 = 3.73  # atmospheric N2:O2 volume ratio


# ---------------------------------------------- air correction --------------

def airfree_correction(n2_measured, o2_measured):
    """Remove atmospheric air contamination using the N2:O2 = 3.73 ratio

        N2_air = 3.73*O2_measured,
        N2_coal = N2_measured - N2_air,

    so that O2 -> 0 after correction.  Returns (N2_coal, air_fraction_removed).
    """
    n2_air = ATM_N2_O2 * o2_measured
    n2_coal = n2_measured - n2_air
    air_removed = n2_air + o2_measured
    return n2_coal, air_removed


def air_contamination_fraction(o2_measured, total_gas):
    """Fraction of a measured gas sample attributable to atmospheric air

        air_fraction = (1 + 3.73)*O2_measured/total_gas,

    a contamination metric (each measured O2 implies 3.73 of accompanying air
    N2); high values flag samples to reject.
    """
    return (1.0 + ATM_N2_O2) * o2_measured / total_gas


# ---------------------------------------------- lost / total gas --------------

def lost_gas_usbm(sqrt_times, cumulative_gas):
    """USBM square-root-of-time lost-gas estimate (Direct Method)

        V(t) = m + b*sqrt(t),

    fit to the early desorption data; the lost gas is the back-extrapolated
    intercept magnitude |m| at t = 0 (the gas escaping before measurement).
    """
    b, m = np.polyfit(np.asarray(sqrt_times, float),
                      np.asarray(cumulative_gas, float), 1)
    return abs(m)


def total_gas_content(lost_gas, desorbed_gas, residual_gas):
    """Total gas content  = lost gas + measured desorbed gas + residual gas."""
    return lost_gas + desorbed_gas + residual_gas


# ---------------------------------------------- quality control --------------

def isotope_valid(delta13c_ch4, delta13c_co2):
    """Isotopic-validity check: values heavier than the natural limits are
    rejected

        valid if delta13C-CH4 < -20 permil and delta13C-CO2 < +20 permil.
    """
    return bool(delta13c_ch4 < -20.0 and delta13c_co2 < 20.0)


def gc_peak_valid(ch4_peak_area_mvs, threshold=50.0):
    """GC quality check: methane peak area must exceed ~50 mV-sec, below which
    the delta-13C uncertainty rises sharply."""
    return bool(ch4_peak_area_mvs >= threshold)


def tedlar_holdtime_ok(hold_time_hours, max_hours=24.0):
    """Tedlar (PVF) bag hold-time check.

    The paper's key finding is that gases permeate the PVF film (O2 ~10x faster
    than N2, He ~15x faster than CO2), biasing composition and isotopes, so the
    USEPA <24 h hold-time is recommended; glass vials are effectively
    impermeable.  Returns True if the sample was analysed within the limit.
    """
    return bool(hold_time_hours <= max_hours)


# ---------------------------------------------- gas origin --------------

def gas_origin(delta13c_ch4):
    """Classify methane gas origin from its carbon isotope ratio (Whiticar, 1996)

        < -60 permil : biogenic,  -60 to -50 : mixed,  > -50 : thermogenic.
    """
    d = delta13c_ch4
    if d < -60.0:
        return "biogenic"
    if d <= -50.0:
        return "mixed"
    return "thermogenic"


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Canister Gas Sampling & Isotopes")
    print("=" * 60)

    # Air correction removes atmospheric N2 in the 3.73:1 ratio with O2
    n2_coal, air = airfree_correction(n2_measured=10.0, o2_measured=2.0)
    print(f"  coal N2 = {n2_coal:.2f}   air removed = {air:.2f}")
    assert np.isclose(n2_coal, 10.0 - 3.73 * 2.0) and np.isclose(air, 3.73 * 2 + 2)
    # air-contamination fraction of the whole sample
    frac = air_contamination_fraction(o2_measured=2.0, total_gas=100.0)
    print(f"  air contamination = {frac*100:.1f}%")
    assert np.isclose(frac, (1 + 3.73) * 2.0 / 100.0)

    # Tedlar-bag hold-time guidance (<24 h); glass vials effectively impermeable
    assert tedlar_holdtime_ok(12.0) and not tedlar_holdtime_ok(48.0)

    # USBM lost gas: a perfect sqrt(t) line back-extrapolates to its intercept
    t = np.array([1.0, 2.0, 3.0, 4.0])
    sqrt_t = np.sqrt(t)
    cum = -5.0 + 20.0 * sqrt_t          # intercept -5 -> lost gas 5
    lost = lost_gas_usbm(sqrt_t, cum)
    print(f"  lost gas = {lost:.2f} cm3")
    assert np.isclose(lost, 5.0)
    total = total_gas_content(lost, desorbed_gas=120.0, residual_gas=15.0)
    assert np.isclose(total, 140.0)

    # QC: natural isotopes pass, anomalously heavy ones fail; weak peaks fail
    assert isotope_valid(-55.0, 5.0)
    assert not isotope_valid(-10.0, 5.0)        # CH4 too heavy
    assert not isotope_valid(-55.0, 25.0)       # CO2 too heavy
    assert gc_peak_valid(80.0) and not gc_peak_valid(30.0)

    # Gas-origin classification
    print(f"  origins: {gas_origin(-70)}, {gas_origin(-55)}, {gas_origin(-45)}")
    assert gas_origin(-70) == "biogenic"
    assert gas_origin(-55) == "mixed"
    assert gas_origin(-45) == "thermogenic"
    print("  PASS")
    return {"coal_N2": float(n2_coal), "lost_gas": float(lost), "total_gas": float(total)}


if __name__ == "__main__":
    test_all()
