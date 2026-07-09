"""
Article 9: NMR Relaxation of Surface-Functionalized Fe3O4 Nanoparticles
Zhu, Ko, Daigle, Zhang (2018)
DOI: 10.30632/PJV59N3-2018a8  (inferred - see note)

Magnetic Fe3O4 nanoparticles strongly shorten the NMR relaxation of the
surrounding fluid; surface functionalization changes how accessible their
magnetic surface is and hence the relaxivity.  This *methodology proxy*
implements the standard NMR relaxation relations the paper measures: the
surface-relaxation (fast-diffusion) rate from the surface-to-volume ratio, the
concentration-linear relaxivity of a contrast agent, and the relaxivity (slope)
recovered by fitting the relaxation rate vs. nanoparticle concentration.

Implements:

  - Surface (fast-diffusion) relaxation  1/T2 = rho2*(S/V)
  - Relaxivity rate law  1/T = 1/T0 + r*C
  - Relaxivity from a linear fit of 1/T vs concentration
  - r2/r1 relaxivity ratio (contrast-agent character)

Note: this article's body was beyond this issue's machine extraction (the source
text ended at journal p372), so - as with the other methodology proxies in this
repository - the relations below are the standard NMR-relaxation formulas the
paper measures, not formulas transcribed from it.  The DOI suffix (a8) is
inferred from the issue's confirmed pattern.  Times in s, relaxivity in s^-1 per
unit concentration.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- relaxation --------------

def surface_relaxation_t2(rho2, s_over_v):
    """Surface (fast-diffusion) transverse relaxation time  T2 = 1/(rho2*S/V).

    rho2 = surface relaxivity (m/s), S/V = pore surface-to-volume ratio (1/m).
    A higher S/V (smaller pores / more particle surface) shortens T2.
    """
    return petrolib.nmr.t2_apparent(rho=rho2, s_over_v=s_over_v)


def relaxation_rate(t0, relaxivity, concentration):
    """Relaxation rate with a contrast agent  1/T = 1/T0 + r*C."""
    return 1.0 / t0 + relaxivity * np.asarray(concentration, float)


def fit_relaxivity(concentrations, rates):
    """Recover (relaxivity r, intercept 1/T0) from 1/T vs concentration data."""
    r, intercept = np.polyfit(np.asarray(concentrations, float),
                              np.asarray(rates, float), 1)
    return r, intercept


def relaxivity_ratio(r2, r1):
    """r2/r1 relaxivity ratio (large -> a T2 / negative-contrast agent)."""
    return r2 / r1


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: NMR Relaxation of Fe3O4 Nanoparticles (proxy)")
    print("=" * 60)

    # Fast-diffusion: a larger surface-to-volume ratio shortens T2
    t2_small_sv = surface_relaxation_t2(1e-5, 1e5)
    t2_large_sv = surface_relaxation_t2(1e-5, 1e6)
    print(f"  T2 (S/V 1e5 / 1e6)     = {t2_small_sv:.3e} / {t2_large_sv:.3e} s")
    assert t2_large_sv < t2_small_sv

    # Relaxivity: build a 1/T2 vs concentration series and recover the slope
    conc = np.array([0.0, 0.5, 1.0, 2.0, 5.0])         # mmol/L of nanoparticles
    r2_true, t0 = 12.0, 2.0
    rates = relaxation_rate(t0, r2_true, conc)
    r2_fit, inter = fit_relaxivity(conc, rates)
    print(f"  fitted r2 / intercept  = {r2_fit:.2f} / {inter:.3f}")
    assert np.isclose(r2_fit, r2_true) and np.isclose(inter, 1.0 / t0)

    # Adding nanoparticles increases the relaxation rate (shortens T2)
    assert relaxation_rate(t0, r2_true, 5.0) > relaxation_rate(t0, r2_true, 0.0)

    # r2/r1 ratio of a strongly transverse (negative-contrast) agent is large
    assert relaxivity_ratio(r2_true, 1.5) > 1.0
    print("  PASS")
    return {"r2_fit": float(r2_fit), "T2_largeSV": float(t2_large_sv)}


if __name__ == "__main__":
    test_all()
