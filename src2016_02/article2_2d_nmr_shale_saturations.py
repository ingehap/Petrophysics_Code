"""
Article 2: Estimating Saturations in Organic Shales Using 2D NMR
Nicot, Vorapalawut, Rousseau, Madariaga, Hamon, Korb (2016)
Reference: Petrophysics Vol. 57, No. 1 (February 2016), pp. 19-29
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best Papers of the 2015 SCA Symposium.  In organic shales the oil/water contrast
on a 2D T1-T2 map is not diffusion-based (as in conventional rocks) but comes
from the T1/T2 ratio: oil confined in organic (kerogen) pores shows a high
T1/T2, water a lower one.  Saturations are obtained by partitioning the T1-T2
map by a T1/T2 cutoff and integrating the signal.  Multifrequency NMR dispersion
(NMRD) confirms the mechanism: water relaxes by 2D diffusion near paramagnetic
surfaces (logarithmic dispersion) while oil relaxes by quasi-1D diffusion in
kerogen pores (inverse-square-root dispersion).

Implements:

  - T1/T2 fluid typing (oil high, water low)
  - 2D T1-T2 map partition by a T1/T2 cutoff and NMR saturations
  - Water NMRD relaxation: biphasic fast exchange with 2D diffusion (Eq. 1)
  - Oil NMRD relaxation: quasi-1D diffusion, R1 ~ 1/sqrt(omega) (Eq. 2)

Note: this issue's PDF has a text layer; the saturation method and the Korb
(2014) NMRD relations (Eqs. 1-2) are transcribed from the body/appendix, while
the typeset glyphs were dropped and the dispersion models are reconstructed in
their characteristic frequency-dependent forms.  Relaxation rates in 1/s,
angular frequencies in rad/s, times in s.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

ELECTRON_PROTON_RATIO = 659.0     # omega_s / omega_I (electron / proton Larmor)


# ---------------------------------------------- fluid typing --------------

def t1t2_fluid_label(t1t2, cutoff=2.0):
    """Type the fluid from the T1/T2 ratio: oil (organic-pore confined) reads
    high, water reads low."""
    return "oil" if t1t2 >= cutoff else "water"


def partition_2d_map(t1t2_values, amplitudes, cutoff=2.0):
    """Partition a 2D T1-T2 map into oil and water volumes by a T1/T2 cutoff.

    Returns (V_oil, V_water): the summed signal amplitudes (porosity units) with
    T1/T2 >= cutoff assigned to oil and the rest to water.
    """
    r = np.asarray(t1t2_values, float)
    a = np.asarray(amplitudes, float)
    v_oil = float(np.sum(a[r >= cutoff]))
    return v_oil, float(np.sum(a)) - v_oil


def nmr_oil_saturation(v_oil, v_water):
    """Oil saturation from the partitioned NMR signal  So = V_oil/(V_oil + V_water)."""
    return petrolib.nmr.nmr_saturation(v_oil, v_oil + v_water)


# ---------------------------------------------- NMRD dispersion --------------

def _j_2d(omega, tau_m):
    """2D spectral density (logarithmic) for diffusion near a surface."""
    return tau_m * np.log(1.0 + 1.0 / (np.asarray(omega, float) * tau_m) ** 2)


def nmrd_water_relaxation(omega_i, r1_bulk, r1_bound, a, tau_m):
    """Water NMRD relaxation rate (Eq. 1, Korb et al., 2014; 2D fast exchange)

        R1 = 1/T1_bulk + R1_bound + a*[3*J2D(omega_I) + 7*J2D(omega_s)],

    with the electronic Larmor frequency omega_s = 659*omega_I.  The 2D spectral
    density gives the logarithmic frequency dispersion observed for water.
    """
    omega_s = ELECTRON_PROTON_RATIO * omega_i
    return r1_bulk + r1_bound + a * (3.0 * _j_2d(omega_i, tau_m)
                                     + 7.0 * _j_2d(omega_s, tau_m))


def nmrd_oil_relaxation(omega_i, r1_bulk, b):
    """Oil NMRD relaxation rate (Eq. 2; quasi-1D diffusion in kerogen pores)

        R1 = R1_bulk + b/sqrt(omega_I),

    the inverse-square-root frequency dispersion of a liquid confined in the
    sponge-like, quasi-1D kerogen pore network.
    """
    return r1_bulk + b / np.sqrt(np.asarray(omega_i, float))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Estimating Saturations in Organic Shales (2D NMR)")
    print("=" * 60)

    # T1/T2 fluid typing: oil reads high, water low
    assert t1t2_fluid_label(8.0) == "oil" and t1t2_fluid_label(1.3) == "water"

    # Partition a small synthetic T1-T2 map and compute oil saturation
    t1t2 = np.array([1.2, 1.5, 3.0, 6.0, 9.0])
    amps = np.array([0.04, 0.03, 0.02, 0.05, 0.06])    # porosity units
    v_oil, v_water = partition_2d_map(t1t2, amps, cutoff=2.0)
    so = nmr_oil_saturation(v_oil, v_water)
    print(f"  V_oil / V_water        = {v_oil:.3f} / {v_water:.3f}")
    print(f"  oil saturation         = {so:.3f}")
    assert np.isclose(v_oil + v_water, amps.sum()) and 0 < so < 1
    assert np.isclose(v_oil, 0.02 + 0.05 + 0.06)

    # Water NMRD: 2D logarithmic dispersion - rate decreases with frequency
    r_lo = nmrd_water_relaxation(1e5, r1_bulk=0.5, r1_bound=1.0, a=1e-6, tau_m=1e-9)
    r_hi = nmrd_water_relaxation(1e7, r1_bulk=0.5, r1_bound=1.0, a=1e-6, tau_m=1e-9)
    print(f"  water R1 lo/hi freq    = {r_lo:.3f} / {r_hi:.3f} 1/s")
    assert r_lo > r_hi > 0

    # Oil NMRD: inverse-square-root dispersion, approaching the bulk rate
    o_lo = nmrd_oil_relaxation(1e5, r1_bulk=1.0, b=300.0)
    o_hi = nmrd_oil_relaxation(1e7, r1_bulk=1.0, b=300.0)
    print(f"  oil R1 lo/hi freq      = {o_lo:.3f} / {o_hi:.3f} 1/s")
    assert o_lo > o_hi > 1.0
    assert np.isclose(o_hi - 1.0, 300.0 / np.sqrt(1e7))
    print("  PASS")
    return {"So": float(so), "water_R1": float(r_lo), "oil_R1": float(o_lo)}


if __name__ == "__main__":
    test_all()
