"""
Article 5: Quantifying the Impact of Petrophysical Properties on Spatial
           Distribution of Contrasting Nanoparticle Agents in the Near-Wellbore
           Region
Kai Cheng, Aderonke Aderibigbe, Masoud Alfi, Zoya Heidari, John Killough (2014)
Reference: Petrophysics Vol. 55, No. 5 (October 2014), pp. 447-460
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best of the 2014 SPWLA Annual Logging Symposium.  Contrast-agent nanoparticles
injected from the borehole migrate into the near-wellbore region by advection,
diffusion-dispersion and first-order deposition (colloid filtration).  A
transport simulator quantifies how porosity, permeability and fractures control
the invasion depth and the spatial concentration distribution.

Implements:

  - Stokes-Einstein Brownian diffusion  Dd = kb*T/(3*pi*mu*dp)  (Eq. 12)
  - Millington-Quirk effective diffusion  Dd_eff = phi_w^(10/3)/phi^2*Dd (Eq. 13)
  - Pore (interstitial) velocity  v_w = u_w/(phi*Sw)  (Eq. 8)
  - Combined diffusion-dispersion coefficient  D* = alphaL*v_w + Dd_eff (Eqs. 9, 16)
  - First-order deposition rate  q_dep = k_dep*C  (Eq. 10)
  - 1D advection-dispersion-filtration transport solver (Eqs. 11, 25)

Note: this is the most equation-rich article, but the display-equation bodies
were dropped in extraction; the forms below are reconstructed from the surviving
variable definitions and nomenclature (Yao, 1971; Millington & Quirk, 1961).
The deposition coefficient k_dep = 2.3e-6 1/s is the paper's history-matched
value.  SI units throughout (m, s, kg).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

KB = 1.380649e-23  # Boltzmann constant, J/K


# ---------------------------------------------- diffusion & dispersion --------------

def stokes_einstein(temperature, viscosity, particle_diameter):
    """Stokes-Einstein Brownian diffusion coefficient (Eq. 12)

        Dd = kb*T/(3*pi*mu*dp),

    inversely proportional to particle diameter dp and fluid viscosity mu.
    """
    # library takes a radius (6*pi*mu*r); dp/2 reproduces the 3*pi*mu*dp form.
    return petrolib.flow_transport.stokes_einstein(temperature, viscosity, particle_diameter / 2.0)


def millington_quirk(dd, phi, sw):
    """Millington-Quirk effective diffusion in porous media (Eqs. 13-14)

        Dd_eff = phi_w^(10/3)/phi^2*Dd,   phi_w = phi*Sw,

    the tortuosity/constriction reduction of the free-fluid diffusion.
    """
    return petrolib.flow_transport.millington_quirk(dd, phi, sw)


def pore_velocity(darcy_velocity, phi, sw):
    """Interstitial (pore) velocity (Eq. 8)

        v_w = u_w/(phi*Sw).
    """
    return petrolib.flow_transport.pore_velocity(darcy_velocity, phi, sw)


def dispersion_coefficient(v_w, dd_eff, dispersivity):
    """Combined diffusion-dispersion coefficient (Eqs. 9, 16)

        D* = alphaL*v_w + Dd_eff,

    mechanical dispersion (longitudinal dispersivity alphaL) plus effective
    molecular diffusion.
    """
    return dispersivity * v_w + dd_eff


def deposition_rate(k_dep, concentration):
    """First-order deposition (colloid filtration) rate (Eq. 10)

        q_dep = k_dep*C.
    """
    return k_dep * np.asarray(concentration, float)


# ---------------------------------------------- 1D transport solver --------------

def transport_1d(c0, length, n_cells, total_time, v_w, d_star, k_dep,
                 phi, sw, n_steps=None):
    """Explicit finite-difference 1D advection-dispersion-filtration transport
    (Eqs. 11, 25)

        d(phi*Sw*C)/dt = d/dx(phi*Sw*D* dC/dx) - d(C*v_w)/dx - k_dep*phi*Sw*C,

    with a fixed injection concentration c0 at x = 0 (upwind advection, central
    diffusion, explicit deposition).  Returns (x, C) at ``total_time``.
    """
    # phi and Sw cancel out of every term of this PDE as written, so the
    # library advection-dispersion-reaction solver reproduces it exactly.
    return petrolib.flow_transport.advect_disperse_1d(
        c0, length=length, n_cells=n_cells, t_total=total_time,
        v=v_w, D=d_star, k_rxn=k_dep, n_steps=n_steps)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Nanoparticle Transport in Near-Wellbore")
    print("=" * 60)

    # Stokes-Einstein: smaller particles diffuse faster
    d1 = stokes_einstein(300.0, 1e-3, 1e-9)     # 1 nm
    d100 = stokes_einstein(300.0, 1e-3, 100e-9)  # 100 nm
    print(f"  Dd(1nm)={d1:.2e}  Dd(100nm)={d100:.2e} m2/s")
    assert d1 > d100 and np.isclose(d1 / d100, 100.0)

    # Millington-Quirk reduces the free-fluid diffusion
    dd_eff = millington_quirk(d1, phi=0.2, sw=1.0)
    assert dd_eff < d1

    # Pore velocity exceeds the Darcy velocity
    v = pore_velocity(7e-6 * 0.2, phi=0.2, sw=1.0)
    assert np.isclose(v, 7e-6)

    # Dispersion dominates diffusion by orders of magnitude (paper's finding)
    d_star = dispersion_coefficient(v, dd_eff, dispersivity=0.01)
    print(f"  D* = {d_star:.2e} m2/s  (dispersion >> diffusion)")
    assert d_star > 100 * dd_eff

    # Deposition is first order in concentration
    assert np.isclose(deposition_rate(2.3e-6, 0.5), 1.15e-6)

    # Transport: a concentration front advances and deposition reduces the
    # plateau below the injected value
    k_dep = 2.3e-6
    x, c = transport_1d(c0=1.0, length=0.5, n_cells=100, total_time=3600.0,
                        v_w=v, d_star=d_star, k_dep=k_dep, phi=0.2, sw=1.0)
    print(f"  near-well C={c[0]:.3f}  front max={c.max():.3f}  far C={c[-1]:.3e}")
    assert c[0] > c[-1]                     # invaded near well, clean far field
    assert 0.0 <= c[-1] < c[0] <= 1.0
    # a less permeable (slower) case invades less deeply
    x2, c2 = transport_1d(1.0, 0.5, 100, 3600.0, v / 10, d_star / 10, k_dep, 0.2, 1.0)
    assert np.sum(c2 > 0.01) < np.sum(c > 0.01)
    print("  PASS")
    return {"Dd_1nm": float(d1), "D_star": float(d_star), "C_near": float(c[0])}


if __name__ == "__main__":
    test_all()
