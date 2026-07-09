"""
Article 2: A Breakthrough in Accurate Downhole Fluid Sample Contamination
           Prediction in Real Time
Zuo, Gisolf, Dumont, Dubost, Pfeiffer, Wang, Mishra, Chen, Mullins, Biagi,
Gemelli (2015)
Reference: Petrophysics Vol. 56, No. 3 (June 2015), pp. 251-265
DOI: none assigned (this issue predates SPWLA DOI assignment)

Oil-based-mud (OBM) filtrate contamination of a downhole fluid sample is
quantified in real time by combining multiple DFA sensors.  Each fluid property
(optical density, density, GOR via the modified-GOR "f-function") mixes linearly
between the virgin fluid and the filtrate, so the contamination volume fraction
follows from any sensor.  During pumpout, properties clean up as a power law in
pumped volume (exponent set by the probe geometry), so extrapolating the
power-law fit to infinite volume gives the virgin-fluid endpoints.

Implements:

  - Linear mixing rule for OD, density and shrinkage factor (Eqs. 1-2, 5)
  - Modified-GOR f-function  f = b*GOR  (Eq. 4)
  - Contamination volume fraction from any sensor (Eq. 6)
  - Power-law cleanup vs. pumped volume and virgin-endpoint extrapolation

Note: this issue's PDF has a text layer; the mixing rules and f-function
(Eqs. 1-6) are transcribed from the body, while the typeset glyphs were dropped
and reconstructed in standard form.  Contamination as a volume fraction; sensor
properties in their own units.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- mixing rules --------------

def mixing_rule(p_virgin, p_filtrate, eta):
    """Linear binary mixing of a fluid property (Eqs. 1, 2, 5)

        P = (1 - eta)*P_virgin + eta*P_filtrate,

    with eta the OBM-filtrate contamination volume fraction (OD, density and
    shrinkage factor all follow this ideal-mixing rule).
    """
    return petrolib.geochem_fluids.contamination.mix_linear(p_virgin, p_filtrate, eta)


def f_function(shrinkage_factor, gor):
    """Modified single-stage-flash GOR (f-function)  f = b*GOR  (Eq. 4),

    which makes the GOR mixing rule linear and consistent with OD/density.
    """
    return shrinkage_factor * gor


def contamination_fraction(p_measured, p_virgin, p_filtrate):
    """OBM-filtrate contamination volume fraction from a sensor (Eq. 6)

        eta = (P - P_virgin)/(P_filtrate - P_virgin).

    Computable independently from OD, density or the f-function; the estimates
    should agree.
    """
    return petrolib.geochem_fluids.contamination.contamination_fraction(p_measured, p_virgin, p_filtrate)


# ---------------------------------------------- power-law cleanup --------------

def power_law_cleanup(volume, p_virgin, amplitude, gamma):
    """Fluid property vs. pumped volume during cleanup

        P(V) = P_virgin + amplitude*V^(-gamma),

    with gamma the probe-geometry exponent (e.g. 2/3 for a 3D radial probe,
    5/12 for a wireline probe); P -> P_virgin as V -> infinity.
    """
    return p_virgin + amplitude * np.asarray(volume, float) ** (-gamma)


def fit_virgin_endpoint(volumes, properties, gamma):
    """Extrapolate the power-law cleanup to infinite pumped volume.

    Fits  P = P_virgin + amplitude*V^(-gamma)  as a straight line in V^(-gamma)
    (intercept = virgin-fluid endpoint, slope = amplitude).  Returns
    (P_virgin, amplitude).
    """
    lf = petrolib.inversion_numerics.fitting.fit_line(
        np.asarray(volumes, float) ** (-gamma), properties)
    return lf.intercept, lf.slope


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Downhole Fluid Contamination Prediction")
    print("=" * 60)

    # Mixing rule: pure virgin at eta=0, pure filtrate at eta=1
    assert np.isclose(mixing_rule(1.2, 0.3, 0.0), 1.2)
    assert np.isclose(mixing_rule(1.2, 0.3, 1.0), 0.3)

    # Contamination fraction inverts the mixing rule and agrees across sensors
    eta_true = 0.12
    od = mixing_rule(1.5, 0.4, eta_true)          # optical density
    rho = mixing_rule(0.75, 0.82, eta_true)       # density
    eta_od = contamination_fraction(od, 1.5, 0.4)
    eta_rho = contamination_fraction(rho, 0.75, 0.82)
    print(f"  eta from OD / density  = {eta_od:.3f} / {eta_rho:.3f}")
    assert np.isclose(eta_od, eta_true) and np.isclose(eta_rho, eta_true)

    # f-function makes GOR mixing linear
    assert np.isclose(f_function(0.7, 800.0), 560.0)

    # Power-law cleanup: property approaches the virgin endpoint as V grows
    v = np.array([5.0, 10.0, 20.0, 40.0, 80.0])
    p = power_law_cleanup(v, p_virgin=0.5, amplitude=2.0, gamma=2.0 / 3.0)
    print(f"  cleanup P(5)/P(80)     = {p[0]:.3f} / {p[-1]:.3f}")
    assert p[0] > p[-1] > 0.5

    # Extrapolating the power-law fit recovers the virgin endpoint
    p_virgin_fit, amp_fit = fit_virgin_endpoint(v, p, gamma=2.0 / 3.0)
    print(f"  fitted virgin endpoint = {p_virgin_fit:.3f}")
    assert np.isclose(p_virgin_fit, 0.5) and np.isclose(amp_fit, 2.0)
    print("  PASS")
    return {"eta_od": float(eta_od), "P_virgin": float(p_virgin_fit)}


if __name__ == "__main__":
    test_all()
