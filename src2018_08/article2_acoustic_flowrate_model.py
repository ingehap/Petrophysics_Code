"""
Article 2: Improved Aero/Hydro Flow-Rate Model Using Acoustics
Seshadri, Freund, Jha, Venna, Walters, Jagannathan (2018)
DOI: 10.30632/PJV59V4-2018a1

A hydrophone measures the acoustic-pressure amplitude radiated by fluid leaking
through a small restriction (e.g. a casing leak).  This module relates that
amplitude to the leak's volumetric/mass flow rate: the leak velocity from a
Bernoulli (liquid) or choked-nozzle (gas) model, the Mach number it produces,
the aeroacoustic amplitude scaling in powers of Mach number, and a calibrated
inversion that predicts flow rate from the measured amplitude.

Implements:

  - Bernoulli liquid leak rate  Q = Cd*A*sqrt(2*dp/rho)
  - Choked-gas critical pressure ratio and mass rate (adiabatic ideal gas)
  - Leak Mach number  M = Q/(A*c)
  - Aeroacoustic amplitude scaling  p'^2 ~ C2*M^2 + C3*M^3 + C4*M^4
  - Calibrated inversion: flow rate predicted from measured amplitude

Note: this issue's PDF has a text layer but its typeset display-equation glyphs
were dropped in extraction, so the numbered relations (Eqs. 4-16) are faithful
standard-form reconstructions of the acoustic leak-rate model (Bernoulli /
choked nozzle and the monopole-dipole-quadrupole Mach scaling).  SI units.
"""

import numpy as np


# ---------------------------------------------- leak kinematics --------------

def bernoulli_liquid_rate(dp, rho, area, cd=0.62):
    """Liquid leak volumetric rate  Q = Cd*A*sqrt(2*dp/rho)  (Eq. 5).

    dp = upstream-minus-downstream pressure drop (Pa), rho = density (kg/m^3),
    area = leak area (m^2), Cd = discharge coefficient.
    """
    return cd * area * np.sqrt(2.0 * np.asarray(dp, float) / rho)


def critical_pressure_ratio(gamma=1.4):
    """Choked-flow critical pressure ratio  (2/(g+1))^(g/(g-1))  (Eq. 6).

    Below this downstream/upstream ratio the gas leak is choked (sonic).
    """
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def choked_gas_mass_rate(p0, rho0, area, gamma=1.4, cm=0.85):
    """Choked-gas mass rate (Eq. 7)

        mdot = Cm*A*sqrt(gamma*rho0*p0*(2/(g+1))^((g+1)/(g-1))).

    p0, rho0 = stagnation (upstream) pressure and density; choked rate depends
    only on upstream conditions, not on the downstream pressure.
    """
    expo = (gamma + 1.0) / (gamma - 1.0)
    return cm * area * np.sqrt(gamma * rho0 * p0 * (2.0 / (gamma + 1.0)) ** expo)


def mach_number(q, area, c):
    """Leak Mach number  M = (Q/A)/c  (Eq. 9): jet velocity over sound speed."""
    return (np.asarray(q, float) / area) / c


# ---------------------------------------------- acoustics --------------

def acoustic_amplitude_sq(mach, c2, c3, c4):
    """Mean-square acoustic amplitude  p'^2 = C2*M^2 + C3*M^3 + C4*M^4  (Eq. 12).

    The three terms are the monopole (mass-flux fluctuation), dipole (turbulent
    stress on the boundary), and quadrupole (free turbulent mixing) contributions
    of the Ffowcs Williams-Hawkings analogy.
    """
    m = np.asarray(mach, float)
    return c2 * m ** 2 + c3 * m ** 3 + c4 * m ** 4


def calibrate_liquid_inversion(amplitude, rate):
    """Fit the predictive liquid inversion  Q = alpha*p' + beta  (Eqs. 13-14).

    For a liquid leak the quadrupole (M^4) term is dropped, leaving an amplitude
    that is monotonic in Q; a short calibration (here ~1/3 of points would train
    it) gives the two constants used to predict rate from measured amplitude.
    """
    a = np.asarray(amplitude, float)
    alpha, beta = np.polyfit(a, np.asarray(rate, float), 1)
    return alpha, beta


def liquid_rate_from_amplitude(amplitude, alpha, beta):
    """Predict liquid leak rate from measured amplitude  Q = alpha*p' + beta."""
    return alpha * np.asarray(amplitude, float) + beta


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Acoustic Flow-Rate Model")
    print("=" * 60)

    # Bernoulli: a 1 mm^2 leak at 1000 psi (~6.9 MPa) in water
    area = 1e-6
    q = bernoulli_liquid_rate(6.9e6, 1000.0, area)
    print(f"  liquid leak rate       = {q * 6e7:.0f} mL/min")   # m^3/s -> mL/min
    assert q > 0 and bernoulli_liquid_rate(2 * 6.9e6, 1000.0, area) > q

    # Choked air (gamma=1.4): critical ratio ~ 0.528, mass rate positive
    cr = critical_pressure_ratio(1.4)
    print(f"  critical pressure ratio = {cr:.3f}")
    assert np.isclose(cr, 0.5283, atol=1e-3)
    mdot = choked_gas_mass_rate(1.4e7, 165.0, area, 1.4)
    assert mdot > 0

    # Amplitude grows with Mach number (monopole + dipole, liquid)
    machs = mach_number(np.linspace(0.5, 5, 20) * area * 1500.0, area, 1500.0)
    amp = np.sqrt(acoustic_amplitude_sq(machs, c2=1.0, c3=0.5, c4=0.0))
    assert np.all(np.diff(amp) > 0)

    # Calibrated inversion recovers the planted rates from amplitude
    rates = np.linspace(150.0, 6700.0, 30)            # mL/min
    m = mach_number(rates / 6e7, area, 1500.0)
    a = np.sqrt(acoustic_amplitude_sq(m, c2=2.0e9, c3=1.0e9, c4=0.0))
    alpha, beta = calibrate_liquid_inversion(a, rates)
    pred = liquid_rate_from_amplitude(a, alpha, beta)
    err = np.abs(pred - rates) / rates
    print(f"  inversion mean error   = {100 * err.mean():.1f} %")
    assert err.mean() < 0.40                          # paper target: mean <= 40%
    print("  PASS")
    return {"q_liquid": float(q), "crit_ratio": float(cr), "inv_err": float(err.mean())}


if __name__ == "__main__":
    test_all()
