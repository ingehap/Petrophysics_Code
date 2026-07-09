"""
Article 5: Innovative Formation Tester Sampling Procedures for Carbon Dioxide
           and Other Reactive Components
Piazza, Vieira, Sacorague, Jones, Dai, Pearl, Aguiar (2021)
DOI: 10.30632/PJV62N1-2021a4

A short operational paper (two figures) on capturing representative samples when
the reservoir fluid contains CO2 and other reactive species that partition,
react, or change phase during pumpout and storage.  The modules implement the
standard quantitative relations the sampling procedure relies on: power-law
contamination cleanup during pumpout, CO2 phase identification at downhole P/T,
Henry's-law CO2 solubility in brine with salting-out, and a mass-balance
correction that recovers the true reservoir CO2 fraction from a sample that
lost CO2 to the aqueous phase / reaction.

Implements:

  - Power-law OBM cleanup  eta(V) = eta0 * (1 + V/V*)^(-5/12)
  - CO2 phase state at (P, T) relative to the critical point (31 C, 73.8 bar)
  - Henry's-law CO2 solubility with a salting-out (Sechenov) factor
  - Reactive-loss mass-balance correction for sampled CO2 fraction

Note: this issue's source PDF has no usable text layer and the paper publishes
no equations, so these are standard petrophysical / PVT proxies for the
quantities the operational procedure controls.  P in bar, T in deg C.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

CO2_TC = 31.0          # deg C, critical temperature
CO2_PC = 73.8          # bar, critical pressure


# ---------------------------------------------- cleanup -----------------

def cleanup_contamination(pumped_vol, eta0, v_star, exponent=5.0 / 12.0):
    """Oil-based-mud filtrate contamination vs pumped volume (power-law decay).

        eta(V) = eta0 * (1 + V/V*)^(-5/12)
    The -5/12 exponent is the late-time formation-tester cleanup law for a
    spherical-then-radial cleanup.  Returns contamination fraction (0..eta0).
    """
    return petrolib.geochem_fluids.contamination.cleanup_powerlaw(pumped_vol, eta0, v_star, exponent=exponent)


def volume_to_target(eta0, v_star, eta_target, exponent=5.0 / 12.0):
    """Pumped volume required to reach a target contamination fraction."""
    return petrolib.geochem_fluids.contamination.volume_to_target(eta0, v_star, eta_target, exponent=exponent)


# ---------------------------------------------- CO2 phase ---------------

def co2_phase(P_bar, T_C):
    """Classify CO2 as gas / liquid / supercritical at (P, T).

      T > Tc and P > Pc          -> supercritical
      P >= Pc and T <= Tc        -> liquid
      otherwise                  -> gas (vapor)
    """
    if T_C > CO2_TC and P_bar > CO2_PC:
        return "supercritical"
    if P_bar >= CO2_PC and T_C <= CO2_TC:
        return "liquid"
    return "gas"


# ---------------------------------------------- solubility --------------

def co2_solubility(P_co2_bar, T_C, salinity_molal=0.0, kH0=0.034, k_salt=0.11):
    """CO2 solubility in brine (mol/kg) by Henry's law with salting-out.

        x = kH(T) * P_co2 * 10^(-k_salt * m_salt)
    kH0 is the Henry coefficient (mol/kg/bar) at 25 C; temperature lowers
    solubility (van 't Hoff style) and salinity reduces it (Sechenov).
    """
    # mild temperature dependence: solubility falls ~1.5%/degC above 25 C
    kH = kH0 * np.exp(-0.015 * (T_C - 25.0))
    salt = 10.0 ** (-k_salt * salinity_molal)
    return kH * P_co2_bar * salt


# ---------------------------------------------- reactive correction -----

def reactive_correction(co2_sampled, co2_lost_fraction):
    """Recover the true reservoir CO2 mole fraction from a depleted sample.

    If a known fraction f of the CO2 was lost to the aqueous phase / reaction
    before capture, the in-situ fraction is  co2_true = co2_sampled / (1 - f),
    renormalized so the corrected composition still sums to one.
    """
    f = float(co2_lost_fraction)
    co2_true = co2_sampled / (1.0 - f)
    # renormalize the (CO2 + hydrocarbon) binary so it sums to 1
    return co2_true / (co2_true + (1.0 - co2_sampled))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Formation-Tester Sampling for CO2")
    print("=" * 60)

    # Cleanup decays monotonically and reaches the target after a finite volume
    V = np.array([0.0, 10.0, 50.0, 200.0])
    eta = cleanup_contamination(V, eta0=0.5, v_star=20.0)
    print(f"  contamination vs vol   = {np.array2string(eta, precision=3)}")
    assert np.all(np.diff(eta) < 0) and abs(eta[0] - 0.5) < 1e-9
    v_need = volume_to_target(0.5, 20.0, 0.05)
    assert abs(cleanup_contamination(v_need, 0.5, 20.0) - 0.05) < 1e-6

    # CO2 phase at downhole reservoir conditions is supercritical
    print(f"  CO2 @ 300 bar / 90 C   = {co2_phase(300.0, 90.0)}")
    assert co2_phase(300.0, 90.0) == "supercritical"
    assert co2_phase(100.0, 15.0) == "liquid"
    assert co2_phase(10.0, 90.0) == "gas"

    # Solubility rises with CO2 partial pressure and falls with salinity
    s_fresh = co2_solubility(100.0, 80.0, salinity_molal=0.0)
    s_brine = co2_solubility(100.0, 80.0, salinity_molal=3.0)
    print(f"  CO2 solubility fresh / brine = {s_fresh:.3f} / {s_brine:.3f} mol/kg")
    assert s_brine < s_fresh
    assert co2_solubility(200.0, 80.0) > co2_solubility(100.0, 80.0)

    # Reactive correction recovers the higher in-situ CO2 fraction
    co2_true = reactive_correction(co2_sampled=0.10, co2_lost_fraction=0.30)
    print(f"  sampled 10% CO2 -> true = {co2_true*100:.1f}%")
    assert co2_true > 0.10
    print("  PASS")
    return {"v_to_5pct": v_need, "co2_true": co2_true,
            "sol_fresh": s_fresh, "sol_brine": s_brine}


if __name__ == "__main__":
    test_all()
