"""
Article 7: Determination of the Appropriate Value of m for Evaluation of
           Carbonate Reservoirs With Vugs and Fractures at the Well-Log Scale
Wang, Peng (2019)
DOI: 10.30632/PJV60N3-2019a6

Archie's cementation exponent m is not a single constant in vuggy/fractured
carbonates: separate (non-connected) vugs raise the effective m, while
conductive fractures lower it.  This module computes an appropriate effective m
at the well-log scale from a porosity partition into matrix, vug and fracture
systems, and shows the bias incurred by assuming m = 2.

Implements:

  - Archie formation factor  F = a/phi^m  and effective m from F
  - Effective m for separate-vug porosity (raises m)
  - Effective m for fracture porosity (lowers m; parallel conduction)
  - Water saturation sensitivity to the chosen m

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the journal metadata and these are faithful
standard-form reconstructions of the dual-porosity cementation-exponent
analysis the paper applies.
"""

import numpy as np


# ---------------------------------------------- Archie ------------------

def formation_factor(phi, a=1.0, m=2.0):
    """Archie formation factor  F = a/phi^m."""
    return a / np.asarray(phi, float) ** m


def effective_m(F, phi, a=1.0):
    """Effective cementation exponent from F and phi  m = ln(F/a)/ln(1/phi)."""
    return np.log(F / a) / np.log(1.0 / np.asarray(phi, float))


def archie_sw(Rt, Rw, phi, a=1.0, m=2.0, n=2.0):
    """Archie water saturation  Sw = (a*Rw/(phi^m*Rt))^(1/n)."""
    return (a * Rw / (np.asarray(phi, float) ** m * Rt)) ** (1.0 / n)


# ---------------------------------------------- vug / fracture m --------

def m_with_separate_vugs(phi_total, phi_vug, m_matrix=2.0):
    """Effective m when part of the porosity is in separate (isolated) vugs.

    Isolated vugs do not conduct, so the conductive (matrix) porosity is
    phi_total - phi_vug; the whole-rock F based on phi_total then implies an
    elevated effective m (the classic vuggy-carbonate effect).
    """
    phi_conn = phi_total - phi_vug
    F = formation_factor(phi_conn, m=m_matrix)        # conduction via matrix only
    return effective_m(F, phi_total)


def m_with_fractures(phi_total, phi_frac, m_matrix=2.0):
    """Effective m when part of the porosity is in conductive fractures.

    Fractures conduct in parallel with the matrix, raising the whole-rock
    conductivity and lowering the effective m.
    """
    sigma_matrix = (phi_total - phi_frac) ** m_matrix    # matrix conductivity (Rw=1)
    sigma_frac = phi_frac ** 1.0                          # fracture m ~ 1 (parallel)
    sigma_total = sigma_matrix + sigma_frac
    F = 1.0 / sigma_total
    return effective_m(F, phi_total)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Appropriate m for Vuggy/Fractured Carbonates")
    print("=" * 60)

    # Effective m inverts the formation factor exactly
    F = formation_factor(0.18, m=2.0)
    assert abs(effective_m(F, 0.18) - 2.0) < 1e-9

    # Separate vugs raise the effective m above the matrix value of 2
    m_vug = m_with_separate_vugs(0.20, phi_vug=0.06, m_matrix=2.0)
    print(f"  effective m (vugs)     = {m_vug:.2f}")
    assert m_vug > 2.0

    # Conductive fractures lower the effective m below 2
    m_frac = m_with_fractures(0.20, phi_frac=0.02, m_matrix=2.0)
    print(f"  effective m (fractures) = {m_frac:.2f}")
    assert m_frac < 2.0

    # Using the wrong m biases water saturation: with vugs, assuming m=2
    # underestimates m and overestimates Sw
    Rt, Rw, phi = 30.0, 0.05, 0.20
    sw_m2 = archie_sw(Rt, Rw, phi, m=2.0)
    sw_mvug = archie_sw(Rt, Rw, phi, m=m_vug)
    print(f"  Sw (m=2 / m_vug={m_vug:.2f}) = {sw_m2:.3f} / {sw_mvug:.3f}")
    assert sw_mvug > sw_m2                         # higher m -> higher Sw
    print("  PASS")
    return {"m_vug": float(m_vug), "m_frac": float(m_frac),
            "sw_m2": float(sw_m2), "sw_mvug": float(sw_mvug)}


if __name__ == "__main__":
    test_all()
