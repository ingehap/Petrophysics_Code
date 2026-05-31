"""
Article 3: Macroscale Young's Moduli of Shale Based on Nanoindentations
Li, Sakhaee-Pour (2016)
Reference: Petrophysics Vol. 57, No. 6 (December 2016), pp. 597-603
DOI: none assigned (this issue predates SPWLA DOI assignment)

Macroscale Young's modulus of shale is predicted from a population of
nanoindentation measurements.  Each indentation gives an indentation modulus
(from the unloading stiffness and contact area) and hardness, converted to
Young's modulus.  The paper's "representative" (loading-frame) model takes a
fraction-weighted mean of the nanoindentation moduli - controlled by the softer
entities - which matches core measurements better than a volume-weighted average
of the stiff mineral moduli.

Implements:

  - Indentation modulus  M = (1/alpha)*(sqrt(pi)/2)*S/sqrt(Ac)
  - Hardness  H = P/Ac
  - Young's modulus from the indentation modulus
  - Representative (nanoindentation-population) vs volume-average (mineral) modulus

Note: this issue's PDF has a text layer; the representative/volume-average
relations (Eqs. 4-5) survived, while the indentation primitives (Eqs. 1-3) lost
their glyphs and are faithful standard-form reconstructions.  Moduli in GPa.
"""

import numpy as np

ALPHA_BERKOVICH = 1.034
E_DIAMOND = 1141.0           # GPa
V_DIAMOND = 0.07


# ---------------------------------------------- nanoindentation --------------

def indentation_modulus(stiffness, contact_area, alpha=ALPHA_BERKOVICH):
    """Indentation modulus  M = (1/alpha)*(sqrt(pi)/2)*S/sqrt(Ac)  (Eq. 1)."""
    return (1.0 / alpha) * (np.sqrt(np.pi) / 2.0) * stiffness / np.sqrt(contact_area)


def hardness(peak_load, contact_area):
    """Indentation hardness  H = P/Ac  (Eq. 2)."""
    return peak_load / contact_area


def youngs_modulus(indent_modulus, nu=0.3, e_indenter=E_DIAMOND, v_indenter=V_DIAMOND):
    """Young's modulus from the indentation modulus (Eq. 3)

        1/M = (1 - nu^2)/E + (1 - nu_i^2)/E_i  ->  solve E.
    """
    return (1.0 - nu ** 2) / (1.0 / indent_modulus - (1.0 - v_indenter ** 2) / e_indenter)


# ---------------------------------------------- upscaling --------------

def representative_modulus(moduli, fractions):
    """Representative macroscale modulus  En = sum_i E_i*f_i  (Eq. 4).

    Fraction-weighted mean of the nanoindentation moduli (soft-entity controlled).
    """
    return float(np.sum(np.asarray(moduli, float) * np.asarray(fractions, float)))


def volume_average_modulus(mineral_moduli, volume_fractions):
    """Volume-average modulus  Em = sum_i E_i*f_i  (Eq. 5), from mineral fractions.

    Same weighted-mean form but using stiff mineral moduli; overestimates E.
    """
    return float(np.sum(np.asarray(mineral_moduli, float) * np.asarray(volume_fractions, float)))


def softest_frame_contribution(moduli, fractions):
    """Fractional contribution of the softest loading frame to En (Fig. 5)

        ratio = (E1n * f1n) / En,

    where (E1n, f1n) are the modulus/fraction of the softest (smallest-modulus)
    distribution.  The paper reports this ratio < 0.33, consistent with the
    soft-entity-controlled representative model.
    """
    moduli = np.asarray(moduli, float)
    fractions = np.asarray(fractions, float)
    i = int(np.argmin(moduli))
    en = representative_modulus(moduli, fractions)
    return float(moduli[i] * fractions[i] / en)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Shale Young's Moduli From Nanoindentation")
    print("=" * 60)

    # Indentation modulus and hardness are positive
    M = indentation_modulus(stiffness=5e4, contact_area=1e-12)
    assert M > 0 and hardness(0.5, 1e-12) > 0

    # Young's modulus from M is close to (1 - nu^2)*M (indenter term small)
    e = youngs_modulus(50.0, nu=0.3)
    print(f"  E from M=50 GPa        = {e:.2f} GPa")
    assert np.isclose(e, (1 - 0.3 ** 2) * 50.0, rtol=0.05)

    # Representative (soft-controlled) modulus is below the mineral volume average
    en = representative_modulus([10.0, 25.0, 45.0], [0.3, 0.4, 0.3])
    em = volume_average_modulus([40.0, 70.0, 95.0], [0.3, 0.4, 0.3])
    print(f"  En (representative) / Em (volume) = {en:.1f} / {em:.1f} GPa")
    assert em > en

    # The softest loading frame contributes < 1/3 of En (Fig. 5)
    ratio = softest_frame_contribution([10.0, 25.0, 45.0], [0.5, 0.3, 0.2])
    print(f"  softest-frame contribution  = {ratio:.3f}")
    assert ratio < 0.33
    print("  PASS")
    return {"E": float(e), "En": en, "Em": em, "soft_ratio": ratio}


if __name__ == "__main__":
    test_all()
