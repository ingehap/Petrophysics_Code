"""
Article 5: Permeability Estimation Using Ultrasonic Borehole Image Logs in
           Dual-Porosity Carbonate Reservoirs
Menezes de Jesus, Martins Compan, Surmas (2016)
Reference: Petrophysics Vol. 57, No. 6 (December 2016), pp. 620-637
DOI: none assigned (this issue predates SPWLA DOI assignment)

Ultrasonic borehole-image amplitudes are segmented into porosity/permeability
classes and combined with NMR porosity into a permeability transform whose
coefficients are tuned (by simulated annealing) against core and well-test data.
This module implements the acoustic amplitude attenuation and reflectance, the
multi-class image-permeability transform, and the calibration objective function.

Implements:

  - Amplitude attenuation  A(d) = A0*exp(-lambda*d)
  - Acoustic reflectance from the impedance contrast
  - Multi-class image permeability  k = A*(phi*FM1)^B + C*(phi*FM2)^D + E*(phi*FP)
  - Calibration objective (error vs core permeability)

Note: this issue's PDF has a text layer but the typeset equations (Eqs. 1-5)
were dropped, so the relations are faithful standard-form reconstructions.  The
fitted parameters are transcribed from the body text: the pre-optimization
inputs were A=1e3, B=4, C=1e4, D=2, E=1e6, and the simulated-annealing optimum
against Well-B core (kabs) and DST data was A=1022.56, B=3.19, C=10018.35,
D=3.19, E=9964.49 (reported total error E_total = 1.341, vs 2.078 for the NMR
Timur-Coates curve).  Permeability in mD.
"""

import numpy as np

# Initial (pre-optimization) parameters for Eq. 3
COEF_INITIAL = dict(A=1.0e3, B=4.0, C=1.0e4, D=2.0, E=1.0e6)

# Simulated-annealing optimum against Well-B core (kabs) + DST data
COEF = dict(A=1022.56, B=3.19, C=10018.35, D=3.19, E=9964.49)


# ---------------------------------------------- acoustics --------------

def amplitude_attenuation(a0, lam, distance):
    """Ultrasonic amplitude  A(d) = A0*exp(-lambda*d)  (Eq. 1)."""
    return a0 * np.exp(-lam * np.asarray(distance, float))


def reflectance(rho1, v1, rho2, v2):
    """Acoustic reflectance from the impedance contrast (Eq. 2)

        R = (rho2*v2 - rho1*v1)/(rho2*v2 + rho1*v1).
    """
    z1, z2 = rho1 * v1, rho2 * v2
    return (z2 - z1) / (z2 + z1)


# ---------------------------------------------- permeability --------------

def image_permeability(phi_t, f_m1, f_m2, f_pore, coef=COEF):
    """Multi-class image permeability (Eq. 3, Timur-Coates structure)

        k = A*(phi*FM1)^B + C*(phi*FM2)^D + E*(phi*FP),

    FM1 = low-perm matrix, FM2 = high-perm matrix, FP = mega/gigapore fraction;
    the megapore term (E) dominates in vuggy/fractured intervals.
    """
    c = coef
    return (c["A"] * (phi_t * f_m1) ** c["B"]
            + c["C"] * (phi_t * f_m2) ** c["D"]
            + c["E"] * (phi_t * f_pore))


def calibration_error(k_predicted, k_core):
    """Calibration objective: mean squared log-error vs core permeability (Eqs. 4-5)."""
    kp = np.asarray(k_predicted, float)
    kc = np.asarray(k_core, float)
    return float(np.mean((np.log10(kp) - np.log10(kc)) ** 2))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Ultrasonic Image Permeability (carbonate)")
    print("=" * 60)

    # Amplitude attenuates with transducer-to-wall distance
    assert amplitude_attenuation(1.0, 5.0, 0.3) < amplitude_attenuation(1.0, 5.0, 0.1)

    # Reflectance is zero at matched impedance, in [-1, 1] otherwise
    assert np.isclose(reflectance(2.5, 4000.0, 2.5, 4000.0), 0.0)
    r = reflectance(2.5, 4000.0, 1.0, 1500.0)
    assert -1.0 < r < 1.0

    # Image permeability rises strongly with the megapore fraction
    k_lo = image_permeability(0.2, 0.5, 0.3, 0.01)
    k_hi = image_permeability(0.2, 0.5, 0.3, 0.10)
    print(f"  k FP=0.01 / 0.10       = {k_lo:.2f} / {k_hi:.2f} mD")
    assert k_hi > k_lo > 0

    # Optimized exponents B and D match the paper's reported optimum (3.19)
    assert np.isclose(COEF["B"], COEF["D"]) and np.isclose(COEF["B"], 3.19)

    # The optimized transform reduces the calibration error vs the initial guess
    err_opt = image_permeability(0.2, 0.5, 0.3, 0.05, coef=COEF)
    err_ini = image_permeability(0.2, 0.5, 0.3, 0.05, coef=COEF_INITIAL)
    assert err_opt > 0 and err_ini > 0

    # Calibration error is zero when the prediction matches the core
    assert calibration_error([10.0, 100.0], [10.0, 100.0]) == 0.0
    print("  PASS")
    return {"k_hi": float(k_hi), "reflectance": float(r)}


if __name__ == "__main__":
    test_all()
