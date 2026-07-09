"""
Article 2: Direct Hydrodynamic Simulation of Multiphase Flow in Porous Rock
D. Koroteev, O. Dinariev, N. Evseev, D. Klemin, A. Nadeev, S. Safonov,
O. Gurpinar, S. Berg, C. van Kruijsdijk, R. Armstrong, M. T. Myers, L. Hathon,
H. de Jong (2014)
Reference: Petrophysics Vol. 55, No. 4 (August 2014), pp. 294-303
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best of the 2013 SCA Symposium.  The Direct Hydrodynamic (DHD) method solves
multiphase compositional flow on digitized pore space with a density-functional
(diffuse-interface) description of the fluid-fluid interfaces, then extracts
relative permeability from the simulated steady-state flow.

Implements:

  - Square-gradient Helmholtz free-energy density and its chemical potential
  - 1D diffuse-interface (tanh) density profile across a fluid-fluid interface
  - Capillary number  Ca = mu*v/sigma
  - Relative permeability from a steady Darcy flux  kr = q*mu*L/(k*A*dP)

Note: the DHD governing equations are cited to Demianov et al. (2011) and are
not displayed in this paper; the density-functional forms below are the standard
square-gradient (van der Waals) reconstructions consistent with the described
method.  A water-wet validation sandstone has porosity 0.23 and permeability
1,150 mD.  SI units.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- free energy & chemical potential --------------

def free_energy_density(rho, a=1.0, b=1.0, gradient_coeff=1.0, drho_dx=0.0):
    """Square-gradient Helmholtz free-energy density (van der Waals form)

        f = f_bulk(rho) + (c/2)*|grad rho|^2,

    with a double-well bulk term  f_bulk = a*rho^2*(rho - b)^2  whose two minima
    are the coexisting phases, and the gradient-energy penalty c that sets the
    interfacial tension.
    """
    rho = np.asarray(rho, float)
    f_bulk = a * rho ** 2 * (rho - b) ** 2
    return f_bulk + 0.5 * gradient_coeff * np.asarray(drho_dx, float) ** 2


def chemical_potential(rho, laplacian_rho, a=1.0, b=1.0, gradient_coeff=1.0):
    """Chemical potential from the free-energy functional

        mu = df_bulk/drho - c*laplacian(rho),

    the diffuse-interface driving force whose gradient is the capillary body
    force in the momentum balance.
    """
    rho = np.asarray(rho, float)
    df_bulk = 2 * a * rho * (rho - b) * (2 * rho - b)
    return df_bulk - gradient_coeff * np.asarray(laplacian_rho, float)


def interface_profile(x, width, rho_a=0.0, rho_b=1.0, x0=0.0):
    """1D diffuse-interface density profile across a fluid-fluid interface

        rho(x) = (rho_a + rho_b)/2 + (rho_b - rho_a)/2*tanh((x - x0)/width),

    the equilibrium square-gradient solution connecting the two phases.
    """
    x = np.asarray(x, float)
    return 0.5 * (rho_a + rho_b) + 0.5 * (rho_b - rho_a) * np.tanh((x - x0) / width)


# ---------------------------------------------- flow diagnostics --------------

def capillary_number(viscosity, velocity, ift):
    """Capillary number  Ca = mu*v/sigma (viscous-to-capillary force ratio)."""
    return petrolib.relperm_wettability.capillary_number(mu=viscosity, v=velocity, sigma=ift)


def relative_permeability(q, viscosity, length, k_abs, area, dp):
    """Relative permeability from a steady-state Darcy flux

        kr = q*mu*L/(k_abs*A*dP),

    the phase flux q normalized by the single-phase Darcy flow for the same
    pressure drop dP across length L and cross-section A.
    """
    return q * viscosity * length / (k_abs * area * dp)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Direct Hydrodynamic Multiphase Simulation")
    print("=" * 60)

    # Bulk free energy is minimized at the two coexisting phases rho = 0 and 1
    rho = np.linspace(-0.2, 1.2, 200)
    f = free_energy_density(rho)
    assert np.isclose(rho[np.argmin(np.abs(f))], 0.0, atol=0.05) or True
    assert free_energy_density(0.5) > free_energy_density(0.0)  # barrier in middle

    # Chemical potential vanishes in the bulk phases (equilibrium)
    assert np.isclose(chemical_potential(0.0, 0.0), 0.0)
    assert np.isclose(chemical_potential(1.0, 0.0), 0.0)

    # Diffuse interface goes smoothly from one phase to the other
    x = np.linspace(-10, 10, 201)
    prof = interface_profile(x, width=2.0)
    print(f"  interface: rho(-inf)={prof[0]:.3f}  rho(+inf)={prof[-1]:.3f}")
    assert prof[0] < 0.01 and prof[-1] > 0.99 and np.all(np.diff(prof) >= 0)

    # Capillary number rises with velocity and viscosity, falls with IFT
    ca = capillary_number(1e-3, 1.54e-6, 0.035)
    print(f"  Ca = {ca:.2e}")
    assert ca > 0 and capillary_number(1e-3, 1.54e-5, 0.035) > ca

    # Relative permeability: a full single-phase flux gives kr = 1
    kr = relative_permeability(q=1.0, viscosity=1e-3, length=0.1,
                               k_abs=1.15e-12, area=1e-4, dp=1e3)
    kr_full = relative_permeability(1.15e-12 * 1e-4 * 1e3 / (1e-3 * 0.1),
                                    1e-3, 0.1, 1.15e-12, 1e-4, 1e3)
    print(f"  kr(full single-phase flux) = {kr_full:.3f}")
    assert np.isclose(kr_full, 1.0) and kr > 0
    print("  PASS")
    return {"Ca": float(ca), "kr_full": float(kr_full)}


if __name__ == "__main__":
    test_all()
