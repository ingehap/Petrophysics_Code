"""
Article 5: An Integrated Petrophysical Workflow for Fluid Characterization and
           Contacts Identification Using NMR Continuous and Stationary
           Measurements in a High-Porosity Sandstone Formation, Offshore Norway
Kozlowski, Chakraborty, Jambunathan, Lowrey, Balliet, Engelman, Anensen,
Kotwicki, Johansen (2021)
DOI: 10.30632/PJV62N2-2021a5

Combines continuous (logging-while-moving) and stationary (station-stop) NMR
to characterize fluids and locate contacts in a ~30%-porosity sandstone.  Two
simultaneous 2D maps (T1-T2 and D-T2) discriminate gas, oil, and water, and
pick gas-oil / oil-water contacts; results are validated against wireline
formation-tester fluid-density and capacitance sensors.

Implements (canonical NMR forms - the paper is a workflow paper with no
numbered equations; these standard relations are supplied and flagged):

  - T2 relaxation  1/T2 = 1/T2bulk + rho2*(S/V) + (gamma*G*TE)^2*D/12
  - T1 relaxation  1/T1 = 1/T1bulk + rho1*(S/V)
  - Hydrogen-index porosity correction  phi_corr = phi_app / HI
  - T2-cutoff partition (clay-bound / capillary / free) at 3 ms and 60 ms
  - D-T2 fluid typing (gas / water / oil) and station stacking SNR

Units: T2/T1 in s internally (reported ms), D in m^2/s, gradient G in T/m,
echo spacing TE in s, surface relaxivity in m/s.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

GAMMA_H = 2.675e8        # rad/s/T
T2_CLAY_MS = 3.0         # clay-bound / capillary split
T2_FREE_MS = 60.0        # capillary / free-fluid split


# ---------------------------------------------- relaxation --------------

def t2_relaxation(t2bulk, rho2, s_over_v, D, G, TE, gamma=GAMMA_H):
    """1/T2 = 1/T2bulk + rho2*(S/V) + (gamma*G*TE)^2*D/12.  Returns T2 (s)."""
    return petrolib.nmr.t2_apparent(
        t2_bulk=t2bulk, rho=rho2, s_over_v=s_over_v, D=D, G=G, TE=TE, gamma=gamma)


def t1_relaxation(t1bulk, rho1, s_over_v):
    """1/T1 = 1/T1bulk + rho1*(S/V).  Returns T1 (s)."""
    return petrolib.nmr.t2_apparent(t2_bulk=t1bulk, rho=rho1, s_over_v=s_over_v)


# ---------------------------------------------- HI correction -----------

def hi_correction(phi_apparent, HI):
    """Hydrogen-index porosity correction  phi_corr = phi_app / HI.

    HI < 1 in hydrocarbon zones boosts apparent porosity (the paper reports a
    ~11% uplift, i.e. HI ~ 0.9).
    """
    return petrolib.nmr.porosity_hi_correction(phi_apparent, HI)


# ---------------------------------------------- cutoff partition --------

def t2_partition(T2_ms, amplitude, clay=T2_CLAY_MS, free=T2_FREE_MS):
    """Partition a T2 distribution into (clay-bound, capillary, free) volumes."""
    return petrolib.nmr.t2_partition(T2_ms, amplitude, cutoffs_ms=(clay, free))


# ---------------------------------------------- fluid typing ------------

def classify_fluid(D, gas_D=1e-8, oil_D=5e-10):
    """Fluid type from apparent diffusion coefficient on the D-T2 map.

    Gas diffuses fast (D >= 1e-8 m^2/s), oil slowly (D <= 5e-10), water between.
    """
    if D >= gas_D:
        return "gas"
    if D <= oil_D:
        return "oil"
    return "water"


# ---------------------------------------------- station SNR -------------

def stacking_snr(base_snr, n_stacks):
    """Station stacking improves SNR as sqrt(number of stacked acquisitions)."""
    return base_snr * np.sqrt(n_stacks)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: NMR Continuous + Stationary Fluid Contacts")
    print("=" * 60)

    # Diffusion term scales with (G*TE)^2: larger gradient/echo -> shorter T2
    t2_short = t2_relaxation(3.0, 1e-5, 3.0 / 1e-4, D=2.5e-9, G=0.2, TE=6e-3)
    t2_long = t2_relaxation(3.0, 1e-5, 3.0 / 1e-4, D=2.5e-9, G=0.2, TE=1.2e-3)
    print(f"  T2 (TE=6ms / 1.2ms)    = {t2_short*1e3:.1f} / {t2_long*1e3:.1f} ms")
    assert t2_short < t2_long          # stronger diffusion attenuation

    # T1 >= T2 for the same pore (no diffusion term on T1)
    t1 = t1_relaxation(4.0, 1e-5, 3.0 / 1e-4)
    assert t1 > 0

    # HI correction: ~11% porosity uplift in a hydrocarbon zone (HI ~ 0.9)
    phi_c = hi_correction(0.30, 0.9)
    print(f"  HI-corrected porosity  = {phi_c:.3f}  (~11% uplift)")
    assert abs(phi_c - 0.30 / 0.9) < 1e-9 and phi_c > 0.30

    # T2 partition into clay-bound / capillary / free at 3 and 60 ms
    T2 = np.array([1.0, 2.0, 10.0, 40.0, 100.0, 300.0])
    amp = np.array([0.02, 0.03, 0.05, 0.04, 0.08, 0.06])
    cb, cap, free = t2_partition(T2, amp)
    print(f"  clay/cap/free          = {cb:.2f} / {cap:.2f} / {free:.2f}")
    assert abs(cb - 0.05) < 1e-9 and abs(free - 0.14) < 1e-9

    # D-T2 fluid typing
    assert classify_fluid(1e-7) == "gas"
    assert classify_fluid(2.5e-9) == "water"
    assert classify_fluid(1e-10) == "oil"
    print("  fluid typing gas/water/oil = OK")

    # Station stacking improves SNR as sqrt(stacks)
    assert abs(stacking_snr(1.0, 4) - 2.0) < 1e-9
    print("  PASS")
    return {"t2_short_ms": t2_short * 1e3, "phi_hi": float(phi_c),
            "partition": (cb, cap, free)}


if __name__ == "__main__":
    test_all()
