"""
Article 4: Microstructural Investigation of Stress-Dependent Permeability in
           Tight-Oil Rocks
King, Sansone, Kortunov, Xu, Callen, Chhatre, Sahoo, Buono (2018)
DOI: 10.30632/petro_059_1_a3

Low-field NMR and X-ray microtomography are combined to explain why the
permeability of tight-oil rocks falls with net confining stress.  The
permeability follows an exponential decline with stress characterized by a
permeability exponent, the matrix gas permeability follows from a
diffusion-coefficient relation, and gas slippage is handled with a Klinkenberg
correction.

Implements:

  - Net confining stress  NCS = overburden - pore pressure
  - Exponential stress-dependent permeability  k = ki*exp(-gamma*(NCS - NCSi))
  - Matrix gas permeability  k = D*mu/B
  - Klinkenberg apparent permeability  ka = k_inf*(1 + b/P)

Note: this issue's PDF has a text layer; the matrix-permeability and Klinkenberg
forms are literally present in the text, while the exponential permeability model
(Eq. 1) lost its glyph in extraction and is a faithful standard-form
reconstruction.  Stress/pressure consistent units; k in m^2 (or as supplied).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- stress dependence --------------

def net_confining_stress(overburden, pore_pressure):
    """Net confining stress  NCS = overburden - pore pressure."""
    return petrolib.flow_transport.net_confining_stress(overburden, pore_pressure=pore_pressure)


def stress_dependent_permeability(ki, gamma, ncs, ncs_i):
    """Exponential stress-dependent permeability (Eq. 1)

        k = ki*exp(-gamma*(NCS - NCSi)),

    ki = permeability at the initial stress NCSi, gamma = permeability exponent.
    """
    return petrolib.flow_transport.stress_permeability(ki, gamma=gamma, ncs=ncs, ncs0=ncs_i)


def matrix_gas_permeability(diffusivity, viscosity, bulk_modulus):
    """Matrix gas permeability  k = D*mu/B  (B = pressure for an ideal gas)."""
    return diffusivity * viscosity / bulk_modulus


def klinkenberg(k_inf, b, pressure):
    """Klinkenberg apparent permeability  ka = k_inf*(1 + b/P)."""
    return petrolib.flow_transport.klinkenberg_apparent(k_inf, b=b, p_mean=pressure)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Stress-Dependent Permeability")
    print("=" * 60)

    # Net stress = overburden minus pore pressure
    assert net_confining_stress(8000.0, 3000.0) == 5000.0

    # Permeability declines exponentially with net confining stress
    ncs = np.array([1000.0, 3000.0, 6000.0])
    k = stress_dependent_permeability(1e-18, 2.3e-4, ncs, ncs_i=1000.0)
    print(f"  k(NCS) ratio end/start = {k[-1] / k[0]:.3f}")
    assert k[0] > k[1] > k[2] and np.isclose(k[0], 1e-18)

    # Matrix gas permeability in the nanodarcy range
    k_mat = matrix_gas_permeability(1e-10, 2.3e-5, 1e6)   # D in m^2/s, mu Pa.s, B Pa
    print(f"  matrix gas k           = {k_mat:.3e} m^2")
    assert k_mat > 0

    # Klinkenberg slip raises apparent k at low pressure
    assert klinkenberg(0.8e-18, 1.4e6, 5e5) > klinkenberg(0.8e-18, 1.4e6, 2e6)
    print("  PASS")
    return {"k_at_6000": float(k[-1]), "matrix_k": float(k_mat)}


if __name__ == "__main__":
    test_all()
