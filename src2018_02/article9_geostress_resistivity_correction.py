"""
Article 9: Study on the Mechanism of Geostress Difference Effect on Tight
           Sandstone Resistivity and Its Correction Method
Liu, Zhang, Zheng, Xin (2018)
DOI: 10.30632/petro_059_1_a8  (contents-only - see note)

A differential geostress (the difference between the principal stresses) deforms
the pore network of a tight sandstone, changing its measured resistivity and
biasing the Archie water saturation; the paper proposes a correction.  This
*methodology proxy* implements the standard relations: a stress-dependent
resistivity model, the inverse correction back to the unstressed resistivity,
and the resulting Archie-saturation bias if the correction is skipped.

Implements:

  - Stress-dependent resistivity  Rt = Rt0*(1 + c*dStress)
  - Resistivity correction back to the unstressed value
  - Archie water saturation  Sw = (a*Rw/(phi^m*Rt))^(1/n)
  - Saturation bias from neglecting the stress correction

Note: this article's body was not in this issue's machine extraction (contents
page only), so - as with the other methodology proxies in this repository - the
relations below are the standard stress-resistivity correction forms the title
describes, not formulas transcribed from the paper.  The DOI is the authoritative
SPWLA/CrossRef value for this issue.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- resistivity --------------

def stressed_resistivity(rt0, d_stress, c=1.0e-5):
    """Resistivity under differential geostress  Rt = Rt0*(1 + c*dStress).

    A larger stress difference closes conductive pathways, raising resistivity.
    """
    return rt0 * (1.0 + c * np.asarray(d_stress, float))


def correct_resistivity(rt_measured, d_stress, c=1.0e-5):
    """Correct a stressed resistivity back to the unstressed Rt0 = Rt/(1 + c*dStress)."""
    return np.asarray(rt_measured, float) / (1.0 + c * d_stress)


def archie_sw(rw, rt, phi, a=1.0, m=2.0, n=2.0):
    """Archie water saturation  Sw = (a*Rw/(phi^m*Rt))^(1/n), clipped to [0,1]."""
    # HAZARD (LIBRARY_MERGE_PLAN.md section 9): this article's argument order
    # is (rw, rt) — the canonical order is (rt, rw).  Mapped explicitly.
    return petrolib.saturation_resistivity.archie_sw(rt, rw, phi=phi, a=a, m=m, n=n, clip=(0.0, 1.0))


def saturation_bias(rw, rt0, d_stress, phi, c=1.0e-5, **archie):
    """Sw error from using the stressed resistivity instead of the corrected one."""
    rt_stressed = stressed_resistivity(rt0, d_stress, c)
    sw_true = archie_sw(rw, rt0, phi, **archie)
    sw_biased = archie_sw(rw, rt_stressed, phi, **archie)
    return sw_biased - sw_true


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: Geostress Resistivity Correction (proxy)")
    print("=" * 60)

    # Differential stress raises resistivity; the correction recovers Rt0
    rt0, dstress = 20.0, 3.0e4
    rt = stressed_resistivity(rt0, dstress)
    print(f"  Rt0 / stressed Rt      = {rt0:.1f} / {rt:.2f} ohm.m")
    assert rt > rt0 and np.isclose(correct_resistivity(rt, dstress), rt0)

    # Neglecting the correction biases the Archie saturation low (Rt too high)
    bias = saturation_bias(0.05, rt0, dstress, phi=0.10)
    print(f"  Sw bias (uncorrected)  = {bias:+.4f}")
    assert bias < 0
    print("  PASS")
    return {"stressed_Rt": float(rt), "Sw_bias": float(bias)}


if __name__ == "__main__":
    test_all()
