"""
Article 2 (Tutorial): Introduction to Resistivity Principles for Formation
                      Evaluation: A Tutorial Primer
Kennedy, Garcia (2019)
DOI: 10.30632/PJV60N2-2019t2

A primer on the resistivity model that has been the formation-evaluation
standard since G.E. Archie (1941): the formation factor relating rock
resistivity to porosity, the resistivity index relating it to water saturation,
and their combination into Archie's water-saturation equation, plus how the
exponents m and n are determined empirically from core data.

Implements:

  - Formation factor  F = a/phi^m  and R0 = F*Rw
  - Resistivity index  I = Rt/R0 = Sw^(-n)
  - Archie water saturation  Sw = (a*Rw/(phi^m*Rt))^(1/n)
  - Empirical m (from F vs phi) and n (from I vs Sw) by log-log regression

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are the standard Archie relations the primer
teaches.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Archie ------------------

def formation_factor(phi, a=1.0, m=2.0):
    """Archie formation factor  F = a/phi^m."""
    return petrolib.saturation_resistivity.formation_factor(phi, a=a, m=m)


def r_zero(Rw, phi, a=1.0, m=2.0):
    """Resistivity at full water saturation  R0 = F*Rw = a*Rw/phi^m."""
    return formation_factor(phi, a, m) * Rw


def resistivity_index(Rt, R0):
    """Resistivity index  I = Rt/R0 = Sw^(-n)."""
    return petrolib.saturation_resistivity.resistivity_index(Rt, R0)


def archie_sw(Rt, Rw, phi, a=1.0, m=2.0, n=2.0):
    """Archie water saturation  Sw = (a*Rw/(phi^m*Rt))^(1/n)."""
    return petrolib.saturation_resistivity.archie_sw(Rt, Rw, phi=phi, a=a, m=m, n=n)


# ---------------------------------------------- exponent fitting --------

def fit_m(phi, F, a=1.0):
    """Cementation exponent from F vs phi  m = -slope of log F vs log phi."""
    m, _ = petrolib.saturation_resistivity.fit_cementation_exponent(phi, np.asarray(F, float) / a)
    return m


def fit_n(sw, I):
    """Saturation exponent from I vs Sw  n = -slope of log I vs log Sw."""
    return petrolib.saturation_resistivity.fit_saturation_exponent(sw, I)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2 (Tutorial): Resistivity Principles Primer")
    print("=" * 60)

    # Formation factor falls with porosity; R0 scales with Rw
    assert formation_factor(0.10) > formation_factor(0.30)
    assert abs(r_zero(0.05, 0.20) - formation_factor(0.20) * 0.05) < 1e-12

    # Archie Sw round-trip with the resistivity index
    Rw, phi, m, n = 0.05, 0.22, 2.0, 2.0
    R0 = r_zero(Rw, phi, m=m)
    Rt = 20.0
    sw = archie_sw(Rt, Rw, phi, m=m, n=n)
    I = resistivity_index(Rt, R0)
    print(f"  Sw = {sw:.3f}   I = {I:.2f}   Sw^-2 = {sw**-2:.2f}")
    assert abs(I - sw ** -n) < 1e-9

    # Recover m from a synthetic F-phi core data set
    phis = np.array([0.08, 0.12, 0.18, 0.25, 0.32])
    F = formation_factor(phis, m=1.9)
    m_fit = fit_m(phis, F)
    print(f"  fitted m               = {m_fit:.3f}  (true 1.9)")
    assert abs(m_fit - 1.9) < 1e-6

    # Recover n from a synthetic I-Sw core data set
    sws = np.array([0.2, 0.35, 0.5, 0.7, 1.0])
    Ivals = sws ** (-1.8)
    n_fit = fit_n(sws, Ivals)
    print(f"  fitted n               = {n_fit:.3f}  (true 1.8)")
    assert abs(n_fit - 1.8) < 1e-6
    print("  PASS")
    return {"Sw": float(sw), "m_fit": float(m_fit), "n_fit": float(n_fit)}


if __name__ == "__main__":
    test_all()
