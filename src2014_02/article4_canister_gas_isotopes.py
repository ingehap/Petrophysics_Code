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

  - Air-contamination ("airfree") correction using the atmospheric N2:O2 = 3.73,
    and an air-free renormalised composition
  - Square-root-of-time lost-gas estimate (Direct Method) and total gas content
  - Isotope and GC quality-control checks (delta-13C limits, CH4 peak area)
  - The canister desorption-rate shipping criterion (<= 10 cm^3/day)
  - The canister overpressure / venting threshold (~1 bar = 15 psi)
  - The 4-5 canister-volume helium flush, and the candidate-material thermal
    conductivities (PVC / stainless steel / aluminum, Yaws 1995)
  - The Tedlar (PVF) bag relative-permeation finding and hold-time check
  - Biogenic / thermogenic gas-origin classification from delta-13C of methane,
    and the CO2-CH4 carbon-isotope separation (using the measured CO2 isotope)

Note: this procedural paper renders no display equations.  The air-correction
ratio (3.73:1, after Jin et al., 2010), the delta-13C validity limits (CH4
> -20 permil and CO2 > +20 permil do not occur naturally), the 50 mV-sec CH4
peak-area cutoff, the <=10 cm^3/day shipping criterion, the ~1 bar (15 psi)
canister venting pressure and the PVF-bag relative-permeation rates are
transcribed from the text; the square-root-of-time lost-gas estimate follows the
accepted desorption guidelines the paper cites (Diamond & Levine, 1981; Mavor &
Nelson, 1997; Barker et al., 2003).  Gas volumes in cm^3, isotopes in permil.
"""

import numpy as np

ATM_N2_O2 = 3.73  # atmospheric N2:O2 volume ratio (Jin et al., 2010)

# Relative permeation rates through PVF (Tedlar) bag film, from the paper's
# controlled experiments (the basis for the <24 h hold-time guidance): O2
# permeates ~10x faster than N2, and He ~15x faster than CO2.
PVF_PERMEATION_O2_VS_N2 = 10.0
PVF_PERMEATION_HE_VS_CO2 = 15.0

# Canister sealed/vented when internal pressure exceeds ~1 bar (15 psi)
CANISTER_VENT_PRESSURE_BAR = 1.0

# Thermal conductivity of candidate canister materials (BTU/ft-hr-degF, Yaws
# 1995): a higher-conductivity body equilibrates the core to reservoir
# temperature faster, but the paper's canisters are PVC (cheap, <=80 degC limit).
CANISTER_THERMAL_CONDUCTIVITY = {
    "PVC": 0.19,
    "stainless_steel": 9.8243,
    "aluminum": 147.3645,
}


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


def airfree_composition(composition):
    """Air-free gas composition by volume percent.

    Removes the atmospheric air (all O2 plus the accompanying N2_air = 3.73*O2)
    and renormalises the remaining components to 100%, the "airfree" basis the
    paper reports its compositions on.  ``composition`` maps species (e.g. "O2",
    "N2", "CH4", "CO2") to volume percent.  Returns a new dict with O2 dropped,
    atmospheric N2 stripped and the residue renormalised.
    """
    comp = {k: float(v) for k, v in composition.items()}
    o2 = comp.pop("O2", 0.0)
    comp["N2"] = comp.get("N2", 0.0) - ATM_N2_O2 * o2
    total = sum(v for v in comp.values() if v > 0.0)
    if total <= 0.0:
        raise ValueError("no coal gas left after the air correction")
    return {k: (v / total * 100.0 if v > 0.0 else 0.0) for k, v in comp.items()}


def helium_flush_ok(canister_volumes, lo=4.0, hi=5.0):
    """Helium-flush check: the paper purges the air from each canister with 4 to
    5 total canister volumes of helium before sealing."""
    return bool(lo <= canister_volumes <= hi)


# ---------------------------------------------- lost / total gas --------------

def lost_gas_usbm(sqrt_times, cumulative_gas):
    """Square-root-of-time lost-gas estimate (Direct Method)

        V(t) = m + b*sqrt(t),

    fit to the early desorption data; the lost gas is the back-extrapolated
    intercept magnitude |m| at t = 0 (the gas escaping before measurement).
    Follows the accepted desorption guidelines the paper cites (Diamond &
    Levine, 1981; Barker et al., 2003) rather than a printed equation.
    """
    b, m = np.polyfit(np.asarray(sqrt_times, float),
                      np.asarray(cumulative_gas, float), 1)
    return abs(m)


def total_gas_content(lost_gas, desorbed_gas, residual_gas):
    """Total gas content  = lost gas + measured desorbed gas + residual gas.

    The crushed (residual) gas represents the endpoint of total gas content.
    """
    return lost_gas + desorbed_gas + residual_gas


def desorption_shipping_ready(rate_cm3_per_day, limit=10.0):
    """Canister shipping criterion: a canister is stable enough to ship once its
    desorption rate falls to no more than ~10 cm^3 of gas per day."""
    return bool(rate_cm3_per_day <= limit)


def canister_overpressured(pressure_bar, limit=CANISTER_VENT_PRESSURE_BAR):
    """Canister venting check: the canister is sealed but vented when the
    internal pressure exceeds ~1 bar (15 psi); canisters are pressure-tested to
    2 bar.  Returns True when the canister should be vented."""
    return bool(pressure_bar > limit)


def fastest_equilibrating_material(conductivities=None):
    """Canister material that brings the core to reservoir temperature fastest -
    the one with the highest thermal conductivity (Yaws, 1995).  Defaults to the
    paper's three candidate materials (PVC, stainless steel, aluminum)."""
    c = conductivities or CANISTER_THERMAL_CONDUCTIVITY
    return max(c, key=c.get)


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


def carbon_isotope_separation(delta13c_co2, delta13c_ch4):
    """Carbon-isotope separation between co-existing CO2 and CH4

        eps_C = delta13C-CO2 - delta13C-CH4   (Whiticar, 1996).

    The paper measures the CO2 isotope alongside CH4 precisely so the two can be
    cross-plotted; the separation discriminates the methane-generation pathway.
    """
    return delta13c_co2 - delta13c_ch4


def gas_origin_co2_ch4(delta13c_co2, delta13c_ch4):
    """Gas origin from the CO2-CH4 carbon-isotope separation (Whiticar, 1996)

        eps_C > 60 permil : biogenic (microbial CO2 reduction),
        eps_C < 25 permil : thermogenic,
        otherwise         : mixed/transitional.

    Using both isotopes (rather than CH4 alone) is what let the paper resolve the
    Well 1 gas as a mixed thermogenic+biogenic system once the weak (<50 mV-sec)
    GC points were discarded.
    """
    eps = carbon_isotope_separation(delta13c_co2, delta13c_ch4)
    if eps > 60.0:
        return "biogenic"
    if eps < 25.0:
        return "thermogenic"
    return "mixed"


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

    # Air-free composition: drop O2 and its 3.73x N2, renormalise to 100%
    raw = {"O2": 2.0, "N2": 12.0, "CH4": 80.0, "CO2": 6.0}
    af = airfree_composition(raw)
    print(f"  airfree composition = {{'CH4': {af['CH4']:.1f}, 'CO2': {af['CO2']:.1f}, 'N2': {af['N2']:.1f}}}")
    assert "O2" not in af and np.isclose(sum(af.values()), 100.0)
    # methane is enriched relative to its raw fraction once air is removed
    assert af["CH4"] > 80.0 and af["N2"] < 12.0

    # Helium flush of 4-5 canister volumes; material with highest conductivity
    assert helium_flush_ok(4.5) and not helium_flush_ok(2.0)
    assert fastest_equilibrating_material() == "aluminum"
    assert CANISTER_THERMAL_CONDUCTIVITY["PVC"] == 0.19

    # Tedlar-bag hold-time guidance (<24 h); glass vials effectively impermeable
    assert tedlar_holdtime_ok(12.0) and not tedlar_holdtime_ok(48.0)
    # PVF relative-permeation finding underpinning the hold-time guidance
    assert PVF_PERMEATION_O2_VS_N2 == 10.0 and PVF_PERMEATION_HE_VS_CO2 == 15.0

    # Canister handling thresholds: ship once desorption slows, vent above ~1 bar
    assert desorption_shipping_ready(8.0) and not desorption_shipping_ready(25.0)
    assert canister_overpressured(1.5) and not canister_overpressured(0.5)

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

    # CO2-CH4 carbon-isotope separation classifies the methanogenesis pathway
    eps = carbon_isotope_separation(delta13c_co2=10.0, delta13c_ch4=-65.0)
    print(f"  CO2-CH4 isotope separation = {eps:.1f} permil -> {gas_origin_co2_ch4(10.0, -65.0)}")
    assert np.isclose(eps, 75.0)
    assert gas_origin_co2_ch4(10.0, -65.0) == "biogenic"   # large separation
    assert gas_origin_co2_ch4(-20.0, -40.0) == "thermogenic"  # small separation
    assert gas_origin_co2_ch4(-5.0, -50.0) == "mixed"
    print("  PASS")
    return {"coal_N2": float(n2_coal), "lost_gas": float(lost), "total_gas": float(total)}


if __name__ == "__main__":
    test_all()
