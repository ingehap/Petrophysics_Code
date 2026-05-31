"""
Article 2: Capillary Pressure and Resistivity Index Measurements in a Mixed-Wet
           Carbonate Reservoir
Moustafa R. Dernaika, Mohamed S. Efnik, Safouh Koronfol, Svein M. Skjaeveland,
Maisoon M. Al Mansoori, Hafez Hafez, Mohammed Z. Kalam (2014)
Reference: Petrophysics Vol. 55, No. 1 (February 2014), pp. 24-30
DOI: none assigned (this issue predates SPWLA DOI assignment)

A SCAL study measures capillary pressure and the resistivity index on the same
plugs across primary-drainage, spontaneous- and forced-imbibition cycles in a
mixed-wet carbonate, classified into rock-reservoir types (RRTs).  The
saturation exponent n is the key deliverable and increases through the
displacement cycles.

Implements:

  - Resistivity index  RI = Rt/Ro = Sw^(-n)
  - Saturation-exponent fit from a log-log RI vs Sw regression
  - Archie formation resistivity factor  FRF = Ro/Rw = a/phi^m
  - Cementation-exponent fit from a log-log FRF vs phi regression

Note: this experimental paper renders no display equations; the resistivity-
index power law and the Archie formation factor are written in standard form.
Reported saturation exponents: high-perm RRT 1-5 n = 1.99 (PD) -> 2.28 (FI);
tight RRT 6-7 n = 1.56 -> 1.82.  Saturations as fractions, resistivities in Ohm*m.
"""

import numpy as np


# ---------------------------------------------- resistivity index --------------

def resistivity_index(rt, ro):
    """Resistivity index  RI = Rt/Ro, the resistivity at saturation Sw relative
    to the fully brine-saturated resistivity Ro."""
    return np.asarray(rt, float) / ro


def resistivity_index_from_sw(sw, n):
    """Resistivity index from water saturation  RI = Sw^(-n)."""
    return np.asarray(sw, float) ** (-n)


def fit_saturation_exponent(sw, ri):
    """Fit the saturation exponent n from a log-log RI vs Sw regression

        log(RI) = -n*log(Sw)  ->  n = -slope.
    """
    x = np.log10(np.asarray(sw, float))
    y = np.log10(np.asarray(ri, float))
    slope, _ = np.polyfit(x, y, 1)
    return -slope


# ---------------------------------------------- formation factor --------------

def formation_resistivity_factor(ro, rw):
    """Archie formation resistivity factor  FRF = Ro/Rw."""
    return ro / rw


def frf_from_porosity(phi, a=1.0, m=2.0):
    """Formation factor from porosity  FRF = a/phi^m."""
    return a * np.asarray(phi, float) ** (-m)


def fit_cementation_exponent(phi, frf):
    """Fit the cementation exponent m from a log-log FRF vs phi regression

        log(FRF) = log(a) - m*log(phi)  ->  m = -slope.

    Returns (m, a).
    """
    x = np.log10(np.asarray(phi, float))
    y = np.log10(np.asarray(frf, float))
    slope, intercept = np.polyfit(x, y, 1)
    return -slope, 10.0 ** intercept


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Pc & Resistivity Index in Mixed-Wet Carbonate")
    print("=" * 60)

    # Resistivity index rises as water saturation falls
    sw = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    ri = resistivity_index_from_sw(sw, n=2.0)
    print(f"  RI(Sw) = {np.round(ri, 2)}")
    assert ri[0] == 1.0 and np.all(np.diff(ri) > 0)

    # Recover the saturation exponent (PD ~1.99, FI ~2.28 in the paper)
    n_pd = fit_saturation_exponent(sw, resistivity_index_from_sw(sw, 1.99))
    n_fi = fit_saturation_exponent(sw, resistivity_index_from_sw(sw, 2.28))
    print(f"  fitted n: PD={n_pd:.2f}  FI={n_fi:.2f}")
    assert np.isclose(n_pd, 1.99) and np.isclose(n_fi, 2.28) and n_fi > n_pd

    # RI from measured resistivities
    assert np.isclose(resistivity_index(40.0, 10.0), 4.0)

    # Cementation exponent from a synthetic FRF vs phi trend
    phi = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    frf = frf_from_porosity(phi, a=1.0, m=2.0)
    m_fit, a_fit = fit_cementation_exponent(phi, frf)
    print(f"  fitted m={m_fit:.3f}  a={a_fit:.3f}")
    assert np.isclose(m_fit, 2.0) and np.isclose(a_fit, 1.0)
    assert np.isclose(formation_resistivity_factor(20.0, 0.5), 40.0)
    print("  PASS")
    return {"n_PD": float(n_pd), "n_FI": float(n_fi), "m": float(m_fit)}


if __name__ == "__main__":
    test_all()
