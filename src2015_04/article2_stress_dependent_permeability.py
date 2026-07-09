"""
Article 2: Steady-State Stress-Dependent Permeability Measurements of Tight
           Oil-Bearing Rocks
Chhatre, Braun, Sinha, Determan, Passey, Zirkle, Wood, Boros, Berry, Leonardi,
Kudva (2015)
Reference: Petrophysics Vol. 56, No. 2 (April 2015), pp. 116-124
DOI: none assigned (this issue predates SPWLA DOI assignment)

Steady-state liquid permeability of tight oil rocks (Vaca Muerta) is measured as
a function of net confining stress (NCS).  Darcy's law gives the permeability
from flow rate, viscosity, plug geometry and pressure drop; the NCS follows the
Terzaghi relationship (sleeve pressure, pore pressure, Biot coefficient); and
permeability declines exponentially with NCS during both drawdown and buildup.

Implements:

  - Darcy permeability  k = q*mu*L/(A*dP)  (Eq. 2)
  - Net confining stress  sigma_NCS = sigma_S - eta*Pp  (Terzaghi; Eq. 3)
  - Exponential stress-dependent permeability  k = k0*exp(-gamma*NCS)  (Eq. 5)
  - Fit of (k0, gamma) from measured k vs. NCS

Note: this issue's PDF has a text layer; the Darcy, Terzaghi and
exponential-decline relations (Eqs. 2-3, 5) are transcribed from the body, while
the typeset glyphs were dropped and reconstructed in standard form.  Permeability
in nD/mD (consistent), stress/pressure in psi, viscosity in cP, lengths in cm.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Darcy --------------

def darcy_permeability(flow_rate, viscosity, length, area, dp):
    """Darcy permeability for laminar flow through a core plug (Eq. 2)

        k = q*mu*L/(A*dP),

    with q the volumetric flow rate, mu the viscosity, L and A the plug length
    and cross-section, and dP the pressure drop.
    """
    return petrolib.flow_transport.darcy_permeability(
        flow_rate, mu=viscosity, length=length, area=area, dp=dp)


# ---------------------------------------------- net confining stress --------------

def net_confining_stress(sleeve_pressure, pore_pressure, biot=1.0):
    """Terzaghi net confining stress (Eq. 3)

        sigma_NCS = sigma_S - eta*Pp,

    with sigma_S the hydrostatic sleeve pressure, Pp the mean pore pressure and
    eta the Biot coefficient (assumed 1 in the paper).
    """
    return petrolib.flow_transport.net_confining_stress(
        sleeve_pressure, pore_pressure=pore_pressure, biot=biot)


# ---------------------------------------------- stress-dependent permeability --------------

def stress_dependent_permeability(k0, gamma, ncs):
    """Exponential permeability decline with net confining stress (Eq. 5)

        k = k0*exp(-gamma*NCS),

    with k0 the zero-stress permeability and gamma the stress-sensitivity
    coefficient.
    """
    return petrolib.flow_transport.stress_permeability(k0, gamma=gamma, ncs=ncs)


def fit_stress_permeability(ncs, k):
    """Fit (k0, gamma) from measured k vs. NCS via a line in (NCS, ln k)

        ln k = ln k0 - gamma*NCS.

    Returns (k0, gamma).
    """
    return petrolib.flow_transport.fit_stress_permeability(ncs, k)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Stress-Dependent Permeability (Tight Oil)")
    print("=" * 60)

    # Darcy permeability scales with flow rate and inversely with pressure drop
    k = darcy_permeability(flow_rate=1e-6, viscosity=1.0, length=2.5, area=5.0, dp=500.0)
    assert k > 0 and darcy_permeability(2e-6, 1.0, 2.5, 5.0, 500.0) > k

    # Net confining stress: NCS = sleeve - Biot*pore pressure
    assert np.isclose(net_confining_stress(5000.0, 2000.0, biot=1.0), 3000.0)
    assert net_confining_stress(5000.0, 2000.0, biot=0.7) > 3000.0

    # Permeability declines exponentially with NCS
    k_lo = stress_dependent_permeability(100.0, 3e-4, 2000.0)
    k_hi = stress_dependent_permeability(100.0, 3e-4, 6000.0)
    print(f"  k @2000/6000 psi NCS   = {k_lo:.2f} / {k_hi:.2f} nD")
    assert k_lo > k_hi > 0

    # Fit recovers the (k0, gamma) used to synthesize the decline
    ncs = np.array([1000.0, 2000.0, 4000.0, 6000.0, 8000.0])
    k_meas = stress_dependent_permeability(120.0, 2.5e-4, ncs)
    k0_fit, gamma_fit = fit_stress_permeability(ncs, k_meas)
    print(f"  fitted k0 / gamma      = {k0_fit:.2f} / {gamma_fit:.2e}")
    assert np.isclose(k0_fit, 120.0) and np.isclose(gamma_fit, 2.5e-4)
    print("  PASS")
    return {"k": float(k), "k0": float(k0_fit), "gamma": float(gamma_fit)}


if __name__ == "__main__":
    test_all()
