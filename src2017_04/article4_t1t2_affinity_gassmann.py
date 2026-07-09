"""
Article 4: Low-Field NMR Spectrometry of Chalk and Argillaceous Sandstones:
           Rock-Fluid Affinity Assessed from T1/T2 Ratio
Katika, Saidian, Prasad, Fabricius (2017)
Reference: Petrophysics Vol. 58, No. 2 (April 2017), pp. 126-140
DOI: none assigned (this issue predates SPWLA DOI assignment)

The NMR T1/T2 ratio measures rock-fluid affinity (a higher ratio means stronger
adsorption, i.e. the wetting fluid), independent of pore geometry.  The
wettability inferred this way then selects the fluid-mixing rule (Voigt for
water-wet, Reuss for oil-wet, Hill for intermediate) feeding a Gassmann fluid
substitution that predicts the saturated bulk modulus from velocities.

Implements:

  - T1/T2 ratio and affinity (wettability) classification
  - Elastic moduli from velocities  M = rho*Vp^2, G = rho*Vs^2, K = M - 4/3*G
  - Voigt / Reuss / Hill fluid-modulus averages and their selection by wettability
  - Gassmann saturated bulk modulus

Note: this issue's PDF has a text layer; the relaxation and elastic relations are
standard forms (typeset glyphs dropped) and faithfully reconstructed.  Moduli in
Pa, velocities in m/s, density in kg/m^3.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- affinity --------------

def t1_t2_ratio(t1, t2):
    """T1/T2 ratio (rock-fluid affinity indicator)."""
    return petrolib.nmr.t1_t2_ratio(t1, t2)


def affinity_class(t1t2, neutral=1.5):
    """Classify rock-fluid affinity: higher T1/T2 -> stronger (wetting) adsorption."""
    if t1t2 > neutral:
        return "wetting"
    if t1t2 < 1.0 / 1.0 * 1.2:        # ~ <=1.2 -> weak interaction
        return "non-wetting"
    return "neutral"


# ---------------------------------------------- elastic --------------

def compressional_modulus(rho, vp):
    """P-wave modulus  M = rho*Vp^2  (Eq. 2)."""
    return rho * np.asarray(vp, float) ** 2


def shear_modulus(rho, vs):
    """Shear modulus  G = rho*Vs^2  (Eq. 3)."""
    return rho * np.asarray(vs, float) ** 2


def bulk_modulus(m, g):
    """Bulk modulus  K = M - 4/3*G  (Eq. 4)."""
    return m - 4.0 / 3.0 * g


def voigt_average(fractions, moduli):
    """Voigt (arithmetic) average modulus  sum(f_i*K_i)."""
    return float(np.sum(np.asarray(fractions, float) * np.asarray(moduli, float)))


def reuss_average(fractions, moduli):
    """Reuss (harmonic) average modulus  1/sum(f_i/K_i)."""
    return float(1.0 / np.sum(np.asarray(fractions, float) / np.asarray(moduli, float)))


def fluid_modulus(fractions, moduli, wettability):
    """Select the fluid-mixing rule by wettability: Voigt (water-wet), Reuss
    (oil-wet), Hill (neutral/mixed)."""
    v = voigt_average(fractions, moduli)
    r = reuss_average(fractions, moduli)
    if wettability == "wetting":
        return v
    if wettability == "non-wetting":
        return r
    return 0.5 * (v + r)


def gassmann(k_frame, k_mineral, k_fluid, phi):
    """Gassmann saturated bulk modulus (Eq. 5)

        Ksat = Kf + (1 - Kf/Km)^2 / (phi/Kfl + (1 - phi)/Km - Kf/Km^2).
    """
    num = (1.0 - k_frame / k_mineral) ** 2
    den = phi / k_fluid + (1.0 - phi) / k_mineral - k_frame / k_mineral ** 2
    return k_frame + num / den


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: T1/T2 Affinity & Gassmann")
    print("=" * 60)

    # Affinity: water in chalk (T1/T2 ~ 2.2) is wetting; oil (~1.2) is not
    assert affinity_class(2.2) == "wetting" and affinity_class(1.1) == "non-wetting"

    # Elastic moduli: K = M - 4/3 G
    M = compressional_modulus(2500.0, 4000.0)
    G = shear_modulus(2500.0, 2300.0)
    K = bulk_modulus(M, G)
    print(f"  M / G / K              = {M/1e9:.2f} / {G/1e9:.2f} / {K/1e9:.2f} GPa")
    assert np.isclose(K, M - 4.0 / 3.0 * G)

    # Voigt >= Hill >= Reuss for a water/gas fluid mixture
    v = voigt_average([0.5, 0.5], [2.2e9, 0.5e9])
    r = reuss_average([0.5, 0.5], [2.2e9, 0.5e9])
    assert v > 0.5 * (v + r) > r
    assert fluid_modulus([0.5, 0.5], [2.2e9, 0.5e9], "wetting") == v

    # Gassmann: saturating the frame stiffens it (Ksat > Kframe)
    ksat = gassmann(k_frame=10e9, k_mineral=37e9, k_fluid=2.2e9, phi=0.2)
    print(f"  Ksat                   = {ksat/1e9:.2f} GPa (frame 10)")
    assert ksat > 10e9
    print("  PASS")
    return {"K_GPa": float(K / 1e9), "Ksat_GPa": float(ksat / 1e9)}


if __name__ == "__main__":
    test_all()
