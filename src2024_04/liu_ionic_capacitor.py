"""
liu_ionic_capacitor.py
Implementation of ideas from:
Liu et al., "Microscopic Ionic Capacitor Models for Petrophysics",
Petrophysics, Vol. 65, No. 2 (April 2024), pp. 158-172.

Three microscopic ionic-capacitor models are implemented:
  (I)  intergranular pore capacitor,
  (II) particle with isolated pore capacitor,
  (III) pyrite/graphite/organic conductive-particle capacitor.
The capacitance C = epsilon * A / d depends on the irregular polar area A and
the inter-pole distance d, both functions of pore geometry.
"""
import numpy as np

EPS0 = 8.854e-12  # F/m


def intergranular_capacitor(area, distance, eps_r=80.0):
    """Model I: capacitance of intergranular pore between two grains."""
    return eps_r * EPS0 * np.asarray(area) / np.asarray(distance)


def isolated_pore_capacitor(radius, shell_thickness, eps_r=80.0):
    """Model II: spherical pore inside a particle, treated as a spherical capacitor."""
    r = np.asarray(radius, dtype=float)
    R = r + shell_thickness
    return 4 * np.pi * eps_r * EPS0 * (r * R) / (R - r + 1e-30)


def conductive_particle_capacitor(area, distance, eps_r=80.0, charge_factor=2.0):
    """Model III: pyrite/graphite particle - higher effective area through induced charges."""
    return charge_factor * eps_r * EPS0 * np.asarray(area) / np.asarray(distance)


def time_varying_charge(C, V0=0.1, tau=1.0, t=None):
    """Charges vary with time as the double layer relaxes."""
    if t is None:
        t = np.linspace(0, 5 * tau, 50)
    return t, C * V0 * (1 - np.exp(-t / tau))


def salinity_effect(C, salinity_ppm):
    """Effective capacitance scales with ionic strength (salinity)."""
    return C * (1 + np.log1p(salinity_ppm / 1000.0))


def test_all():
    A = np.array([1e-12, 5e-12, 1e-11])
    d = np.array([1e-7, 2e-7, 5e-7])
    C1 = intergranular_capacitor(A, d)
    assert (C1 > 0).all()
    C2 = isolated_pore_capacitor(np.array([1e-7, 5e-7]), 1e-8)
    assert (C2 > 0).all()
    C3 = conductive_particle_capacitor(A, d)
    assert (C3 > C1).all()
    t, q = time_varying_charge(C1[0])
    assert q[-1] > q[0]
    Cs = salinity_effect(C1, 50000)
    assert (Cs > C1).all()
    print("liu_ionic_capacitor OK  C1[0]=%.3e F" % C1[0])


if __name__ == "__main__":
    test_all()
