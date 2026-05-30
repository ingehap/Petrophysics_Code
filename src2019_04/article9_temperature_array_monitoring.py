"""
Article 9: Monitoring Core Measurements With High-Resolution Temperature Arrays
Howard, Hester (2019)
DOI: 10.30632/PJV60N2-2019a7

A high-resolution temperature array along a core monitors processes (e.g.
hydrate formation/dissociation, fluid fronts) through their thermal signatures:
exothermic/endothermic reactions and fluids of differing thermal properties
create temperature anomalies that the array localizes in space and time.  Heat
transfer is modeled by 1D conduction with a moving source/front.

Implements:

  - 1D transient heat-conduction step (explicit finite difference)
  - Thermal diffusivity  alpha = k/(rho*cp)
  - Front localization from the temperature-gradient peak
  - Stability (CFL) number for the explicit scheme

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard heat-conduction / front-detection relations the
paper applies.
"""

import numpy as np


# ---------------------------------------------- heat conduction ---------

def thermal_diffusivity(k, rho, cp):
    """Thermal diffusivity  alpha = k/(rho*cp)  (m^2/s)."""
    return k / (rho * cp)


def cfl_number(alpha, dt, dx):
    """Explicit-scheme stability number  alpha*dt/dx^2 (must be <= 0.5)."""
    return alpha * dt / dx ** 2


def conduction_step(T, alpha, dt, dx, source=None):
    """One explicit finite-difference step of 1D heat conduction (insulated ends)."""
    T = np.asarray(T, float).copy()
    lap = np.zeros_like(T)
    lap[1:-1] = T[2:] - 2.0 * T[1:-1] + T[:-2]
    lap[0] = T[1] - T[0]
    lap[-1] = T[-2] - T[-1]
    Tn = T + alpha * dt / dx ** 2 * lap
    if source is not None:
        Tn += np.asarray(source, float) * dt
    return Tn


def front_location(T, dx):
    """Locate a thermal front as the position of maximum |temperature gradient|."""
    grad = np.abs(np.gradient(np.asarray(T, float), dx))
    return int(np.argmax(grad))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: High-Resolution Temperature-Array Monitoring")
    print("=" * 60)

    alpha = thermal_diffusivity(2.0, 2300.0, 900.0)
    print(f"  thermal diffusivity    = {alpha:.2e} m^2/s")
    assert alpha > 0

    # Stable explicit scheme requires CFL <= 0.5
    dx = 0.005
    dt = 0.4 * dx ** 2 / alpha
    cfl = cfl_number(alpha, dt, dx)
    print(f"  CFL number             = {cfl:.2f}")
    assert cfl <= 0.5

    # A step temperature profile diffuses and smooths over time (peak gradient
    # decreases), conserving total heat in an insulated rod
    n = 60
    T = np.zeros(n); T[:30] = 1.0
    g0 = np.max(np.abs(np.gradient(T, dx)))
    heat0 = T.sum()
    for _ in range(200):
        T = conduction_step(T, alpha, dt, dx)
    g1 = np.max(np.abs(np.gradient(T, dx)))
    print(f"  max gradient before/after = {g0:.1f} / {g1:.1f}")
    assert g1 < g0                                # front smooths by conduction
    assert abs(T.sum() - heat0) < 1e-6            # insulated -> heat conserved

    # An exothermic source creates a local hot spot the array localizes
    T2 = np.full(n, 20.0)
    src = np.zeros(n); src[40] = 50.0             # heat release at cell 40
    for _ in range(50):
        T2 = conduction_step(T2, alpha, dt, dx, source=src)
    hot = int(np.argmax(T2))
    print(f"  hot-spot cell          = {hot}  (source at 40)")
    assert abs(hot - 40) <= 1
    # the front of the diffused step sits near the original interface
    Tstep = np.zeros(n); Tstep[:25] = 1.0
    Tstep = conduction_step(Tstep, alpha, dt, dx)
    assert abs(front_location(Tstep, dx) - 25) <= 2
    print("  PASS")
    return {"alpha": float(alpha), "cfl": float(cfl), "hot_cell": hot}


if __name__ == "__main__":
    test_all()
