"""
Article 5: How Can Microfracturing Improve Reservoir Management?
Malik, Jones, Boratko (2016)
Reference: Petrophysics Vol. 57, No. 5 (October 2016), pp. 492-507
DOI: none assigned (this issue predates SPWLA DOI assignment)

Microfracturing isolates a ~1 m interval between inflatable packers, raises the
pressure until a tensile fracture forms, then runs injection/shut-in cycles to
extend the fracture and read the minimum in-situ stress.  Breakdown and fracture
reopening pressures relate the wellbore tangential (Kirsch) stress to the two
horizontal stresses; the instantaneous shut-in pressure (ISIP) and the
G-function/square-root-of-time closure analysis give the closure pressure, which
equals the minimum stress.  The vertical stress is the integrated overburden and
the horizontal stress follows an Eaton-type poroelastic model.

Implements:

  - Vertical (overburden) stress from a density profile
  - Eaton-type minimum horizontal stress with pore pressure
  - Kirsch breakdown pressure (with tensile strength) and reopening pressure
  - Maximum horizontal stress inverted from breakdown / reopening
  - Net pressure = ISIP - closure, and G-function-time closure picking

Note: this issue is a procedural/case-study paper; the relations below are the
standard hydraulic-fracturing stress mechanics it relies on (Kirsch, 1898;
Hubbert & Willis, 1957; Eaton, 1969; Haimson & Fairhurst, 1970; Nolte, 1979).
Stresses/pressures in psi, depths in ft, densities in g/cm^3.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

PSI_PER_FT_PER_GCC = 0.433    # psi/ft per g/cm^3 (overburden gradient)


# ---------------------------------------------- stress profile --------------

def vertical_stress(depths, densities):
    """Vertical (overburden) stress from the cumulative density load

        sigma_v(z) = 0.433 * cumulative integral of rho dz,

    with depths in ft (increasing) and bulk densities in g/cm^3.  Returns the
    overburden stress (psi) at each depth.
    """
    depths = np.asarray(depths, float)
    densities = np.asarray(densities, float)
    dz = np.diff(depths, prepend=depths[0])
    return PSI_PER_FT_PER_GCC * np.cumsum(densities * dz)


def min_horizontal_stress(sigma_v, pore_pressure, nu, biot=1.0, tectonic=0.0):
    """Eaton-type minimum horizontal stress with pore pressure

        sigma_h = nu/(1-nu)*(sigma_v - alpha*Pp) + alpha*Pp + tectonic.
    """
    return petrolib.acoustic_geomech.min_horizontal_stress(
        sigma_v, pore_pressure, nu, biot=biot, tectonic=tectonic)


# ---------------------------------------------- breakdown / reopening --------------

def breakdown_pressure(sigma_h, sigma_H, pore_pressure, tensile_strength):
    """Kirsch breakdown pressure for a vertical wellbore (impermeable wall)

        Pb = 3*sigma_h - sigma_H - Pp + T0,

    the pressure at which the wellbore tangential (hoop) stress reaches the rock
    tensile strength T0 and a fracture initiates (Hubbert & Willis, 1957).
    """
    return petrolib.acoustic_geomech.breakdown_pressure(
        sigma_h, sigma_H, pore_pressure, tensile_strength=tensile_strength)


def reopening_pressure(sigma_h, sigma_H, pore_pressure):
    """Fracture reopening pressure (no tensile strength on reopening)

        Pr = 3*sigma_h - sigma_H - Pp.
    """
    return petrolib.acoustic_geomech.reopening_pressure(sigma_h, sigma_H, pore_pressure)


def shmax_from_reopening(pr, sigma_h, pore_pressure):
    """Maximum horizontal stress inverted from the reopening pressure

        sigma_H = 3*sigma_h - Pr - Pp.
    """
    return petrolib.acoustic_geomech.shmax_from_reopening(pr, sigma_h, pore_pressure)


# ---------------------------------------------- closure / net pressure --------------

def net_pressure(isip, closure_pressure):
    """Net fracturing pressure  Pnet = ISIP - Pclosure."""
    return isip - closure_pressure


def gfunction_closure(g_time, pressure):
    """Closure pressure from a G-function leakoff plot (Nolte, 1979).

    Under normal leakoff the shut-in pressure declines linearly in G-function
    time; closure is the departure from that line.  This returns the
    straight-line-fit pressure at G = 0 (the closure-pressure estimate) from the
    early, linear portion of (G, P).
    """
    g = np.asarray(g_time, float)
    p = np.asarray(pressure, float)
    slope, intercept = np.polyfit(g, p, 1)
    return float(intercept)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Microfracturing In-Situ Stress")
    print("=" * 60)

    # Overburden stress increases monotonically with depth (~1 psi/ft typical)
    depths = np.linspace(0.0, 10000.0, 101)
    dens = np.full_like(depths, 2.31)            # ~1.0 psi/ft gradient
    sv = vertical_stress(depths, dens)
    print(f"  overburden @10000 ft   = {sv[-1]:.0f} psi")
    assert sv[-1] > sv[0] and np.isclose(sv[-1], PSI_PER_FT_PER_GCC * 2.31 * 10000.0, rtol=1e-2)

    # Minimum horizontal stress lies between pore pressure and overburden
    sigma_v, pp, nu = 9000.0, 4500.0, 0.25
    sh = min_horizontal_stress(sigma_v, pp, nu, biot=1.0)
    print(f"  sigma_h                = {sh:.0f} psi")
    assert pp < sh < sigma_v

    # Breakdown exceeds reopening by the tensile strength
    sigma_H = 7000.0
    pb = breakdown_pressure(sh, sigma_H, pp, tensile_strength=800.0)
    pr = reopening_pressure(sh, sigma_H, pp)
    print(f"  breakdown / reopening  = {pb:.0f} / {pr:.0f} psi")
    assert np.isclose(pb - pr, 800.0)

    # Inverting the reopening pressure recovers sigma_H
    assert np.isclose(shmax_from_reopening(pr, sh, pp), sigma_H)

    # Net pressure is ISIP minus closure; closure equals the minimum stress
    assert np.isclose(net_pressure(6200.0, 5800.0), 400.0)

    # G-function closure picks the linear-decline intercept
    g = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    p = 5800.0 - 300.0 * g                        # linear leakoff toward closure
    pc = gfunction_closure(g, p)
    print(f"  G-function closure     = {pc:.0f} psi")
    assert np.isclose(pc, 5800.0)
    print("  PASS")
    return {"sigma_v": float(sv[-1]), "sigma_h": float(sh), "Pb": float(pb), "Pclosure": pc}


if __name__ == "__main__":
    test_all()
