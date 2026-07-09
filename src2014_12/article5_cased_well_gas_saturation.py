"""
Article 5: Physical Basis for a Cased-Well Quantitative Gas-Saturation
           Analysis Method
F. Inanc, W.A. Gilchrist, R. Ansari, D. Chace (2014)
Reference: Petrophysics Vol. 55, No. 6 (December 2014), pp. 598-617
DOI: none assigned (this issue predates SPWLA DOI assignment)

A pulsed-neutron tool with short-, long- and extra-long-spaced detectors
records inelastic (burst-phase) and capture (thermalized) gamma counts.  The
inelastic count ratio RIN13 (short / extra-long spaced) is sensitive to the
neutron slowing-down power and so to gas, while the capture ratio RATO13 is
salinity/clay sensitive.  Gas saturation is read by interpolating the measured
ratio between the modelled gas- and liquid-filled limits (the Dynamic Gas
Envelope).

Implements:

  - Elastic energy loss  Ef/Ei = 1/2[(1+a) + (1-a)cos(theta)]  (Eq. 3)
  - Collision parameter  a = ((A-1)/(A+1))^2  (Eq. 4)
  - Lethargy  u = ln(E0/E)  (Eq. 5)
  - Average lethargy gain  xi = 1 + a/(1-a)*ln(a)  (Eq. 6)
  - Moderating power  MP = xi*Sigma_s  (Eq. 7)
  - Inelastic and capture count ratios RIN13, RATO13 (Eqs. 1, 2)
  - Gas saturation by Dynamic-Gas-Envelope interpolation

Note: this issue's PDF has a text layer; the neutron-physics relations (Eqs.
3-7) and the count-ratio definitions (Eqs. 1-2) are transcribed from the body
and reconstructed in standard form (Reuss, 2008).  The gas-saturation step is a
Monte-Carlo chart interpolation; here it is a normalized (optionally nonlinear)
interpolation between the gas and liquid envelope limits.  Energies in MeV.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- elastic scattering --------------

def collision_parameter(mass_number):
    """Maximum-energy-loss collision parameter (Eq. 4)

        a = ((A-1)/(A+1))^2,

    with the target mass number A.  a -> 0 for hydrogen (A=1).
    """
    return petrolib.nuclear.collision_parameter(mass_number)


def elastic_energy_ratio(mass_number, theta_cm):
    """Fractional energy after a single elastic collision (Eq. 3)

        Ef/Ei = 1/2[(1+a) + (1-a)*cos(theta_cm)],

    with the centre-of-mass scattering angle theta_cm (radians).
    """
    a = collision_parameter(mass_number)
    return 0.5 * ((1.0 + a) + (1.0 - a) * np.cos(theta_cm))


def lethargy(energy, e0=14.2):
    """Neutron lethargy (Eq. 5)

        u = ln(E0/E),

    referenced to the source energy E0 (default 14.2 MeV PNG neutrons).
    """
    return np.log(e0 / np.asarray(energy, float))


def average_lethargy_gain(mass_number):
    """Average logarithmic energy decrement per collision (Eq. 6)

        xi = 1 + a/(1-a)*ln(a),

    independent of the incident energy.  xi -> 1.0 for hydrogen.
    """
    return petrolib.nuclear.average_lethargy_gain(mass_number)


def moderating_power(mass_number, sigma_s):
    """Slowing-down (moderating) power (Eq. 7)

        MP = xi*Sigma_s,

    with the macroscopic scattering cross-section Sigma_s (1/cm).
    """
    return petrolib.nuclear.moderating_power(mass_number, sigma_s)


# ---------------------------------------------- count ratios --------------

def inelastic_ratio(counts_ss, counts_xls):
    """Inelastic (burst-phase) count ratio RIN13 (Eq. 1)

        RIN13 = sum N_SS / sum N_XLS,

    summed over the early 10-us time windows (short-spaced over extra-long-
    spaced detector), background subtracted.
    """
    return float(np.sum(counts_ss)) / float(np.sum(counts_xls))


def capture_ratio(counts_ss, counts_xls):
    """Capture (thermalized) count ratio RATO13 (Eq. 2)

        RATO13 = sum N_SS(capture) / sum N_XLS(capture),

    summed over the late time windows, background subtracted.
    """
    return float(np.sum(counts_ss)) / float(np.sum(counts_xls))


# ---------------------------------------------- gas saturation --------------

def gas_saturation(r_meas, r_gas, r_wet, nonlinearity=1.0):
    """Gas saturation by Dynamic-Gas-Envelope interpolation

        Sg = [(R_wet - R_meas)/(R_wet - R_gas)]^p,

    the position of the measured ratio between the liquid-filled (R_wet) and
    gas-filled (R_gas) envelope limits, with an optional nonlinearity exponent p
    (the RIN13 response is more nonlinear than RATO13).
    """
    frac = (r_wet - r_meas) / (r_wet - r_gas)
    frac = np.clip(frac, 0.0, 1.0)
    return frac ** nonlinearity


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Cased-Well Gas-Saturation Analysis")
    print("=" * 60)

    # Hydrogen is the strongest moderator: a~0, xi~1, big lethargy gain/collision
    a_h = collision_parameter(1)
    a_o = collision_parameter(16)
    print(f"  collision parameter: H={a_h:.3f}  O={a_o:.3f}")
    assert np.isclose(a_h, 0.0) and a_o > a_h
    assert np.isclose(average_lethargy_gain(1), 1.0)
    assert average_lethargy_gain(1) > average_lethargy_gain(16)

    # Elastic backscatter (theta=pi) loses the most energy; forward (0) the least
    assert (elastic_energy_ratio(12, np.pi)
            < elastic_energy_ratio(12, 0.0))
    # lethargy increases as the neutron slows down
    assert lethargy(1.0) > lethargy(10.0) >= 0

    # Hydrogen has the highest moderating power for equal scattering cross-section
    assert moderating_power(1, 1.0) > moderating_power(16, 1.0)

    # Count ratios are positive; gas lowers RIN13 toward the matrix value
    rin = inelastic_ratio([120, 110, 100], [40, 38, 35])
    rato = capture_ratio([90, 85], [50, 48])
    print(f"  RIN13={rin:.3f}  RATO13={rato:.3f}")
    assert rin > 0 and rato > 0

    # Gas saturation: at the gas limit Sg=1, at the wet limit Sg=0, monotone
    sg_full = gas_saturation(r_meas=2.0, r_gas=2.0, r_wet=3.5)
    sg_none = gas_saturation(r_meas=3.5, r_gas=2.0, r_wet=3.5)
    sg_mid = gas_saturation(r_meas=2.75, r_gas=2.0, r_wet=3.5)
    print(f"  Sg: gas-limit={sg_full:.2f}  wet-limit={sg_none:.2f}  mid={sg_mid:.2f}")
    assert np.isclose(sg_full, 1.0) and np.isclose(sg_none, 0.0)
    assert np.isclose(sg_mid, 0.5)
    print("  PASS")
    return {"RIN13": float(rin), "RATO13": float(rato), "Sg_mid": float(sg_mid)}


if __name__ == "__main__":
    test_all()
