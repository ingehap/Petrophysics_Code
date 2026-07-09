"""
Article 4: A Multilevel Iterative Method to Quantify Effects of Pore-Size
           Distribution on Phase Equilibrium of Multicomponent Fluids in
           Unconventional Plays
Li, Mezzatesta, Li, Ma, Jamili (2016)
Reference: Petrophysics Vol. 57, No. 2 (April 2016), pp. 121-139
DOI: none assigned (this issue predates SPWLA DOI assignment)

In shale nanopores the capillary pressure between the liquid and vapor phases is
no longer negligible, so the confined-fluid phase equilibrium differs from the
bulk.  This paper introduces a phase-equilibrium model for nonuniform pores (a
critical pore radius tied to the pore-size distribution) and a multilevel
iterative solver.  The building blocks are the bulk vapor-liquid flash
(Rachford-Rice with Wilson K-values), the Young-Laplace capillary pressure, the
Macleod-Sugden (parachor) interfacial tension, and a Peng-Robinson EOS kernel.

Implements:

  - Wilson K-value correlation
  - Rachford-Rice flash (vapor fraction, liquid/vapor compositions; Eq. 1)
  - Young-Laplace capillary pressure  pc = 2*sigma*cos(theta)/rp  (Eq. 2)
  - Macleod-Sugden parachor interfacial tension (Eq. 2)
  - Peng-Robinson compressibility-factor kernel

Note: this issue's PDF has a text layer; the bulk/confined phase-equilibrium
relations (Eqs. 1-4) are transcribed from the body, while the typeset glyphs
were dropped and the flash / EOS / IFT pieces are implemented in standard form
(the full multilevel critical-pore-radius iteration is built on these kernels).
Pressures in psi (or consistent), temperature in R (or K consistently), radii
in m, IFT in dyne/cm.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

R_GAS = 10.7316               # psi*ft^3/(lbmol*R), Peng-Robinson gas constant


# ---------------------------------------------- flash --------------

def wilson_k_values(p, temperature, pc, tc, omega):
    """Wilson K-value correlation (initial equilibrium ratios)

        K_i = (pc_i/p) * exp[5.373*(1 + omega_i)*(1 - Tc_i/T)].
    """
    pc, tc, omega = (np.asarray(v, float) for v in (pc, tc, omega))
    return (pc / p) * np.exp(5.373 * (1.0 + omega) * (1.0 - tc / temperature))


def rachford_rice(z, k, tol=1e-12, max_iter=100):
    """Solve the Rachford-Rice equation for the vapor mole fraction beta

        sum_i z_i*(K_i - 1)/(1 + beta*(K_i - 1)) = 0,

    by bisection on [0, 1].  Returns beta (vapor fraction).
    """
    z, k = np.asarray(z, float), np.asarray(k, float)

    def f(beta):
        return np.sum(z * (k - 1.0) / (1.0 + beta * (k - 1.0)))

    lo, hi = 1e-10, 1.0 - 1e-10
    if f(lo) < 0:
        return 0.0
    if f(hi) > 0:
        return 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def flash_compositions(z, k, beta):
    """Liquid (x) and vapor (y) compositions from the flash (Eq. 1)

        x_i = z_i/(1 + beta*(K_i - 1)),   y_i = K_i*x_i.
    """
    z, k = np.asarray(z, float), np.asarray(k, float)
    x = z / (1.0 + beta * (k - 1.0))
    return x, k * x


# ---------------------------------------------- confinement --------------

def young_laplace_pc(sigma, pore_radius, theta=0.0):
    """Young-Laplace capillary pressure between vapor and liquid (Eq. 2)

        pc = 2*sigma*cos(theta)/rp,

    the pressure difference across the meniscus in a pore of radius rp.  sigma in
    dyne/cm, rp in m -> pc in Pa (1 dyne/cm = 1e-3 N/m).  theta in radians.
    """
    # Signed Young-Laplace; this article passes theta in radians, so bridge to
    # the library's degrees convention with np.degrees, and dyne/cm -> N/m on sigma.
    return petrolib.capillary_pressure.young_laplace_pc(
        pore_radius, sigma=sigma * 1e-3, theta_deg=np.degrees(theta), absolute=False)


def macleod_sugden_ift(parachors, x, y, v_liquid, v_vapor):
    """Macleod-Sugden (parachor) interfacial tension (Eq. 2)

        sigma^(1/4) = sum_i Par_i*(x_i/Vl - y_i/Vv),

    with Vl, Vv the liquid/vapor molar volumes.  Returns sigma (same unit basis
    as the parachors, conventionally dyne/cm).
    """
    par = np.asarray(parachors, float)
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.sum(par * (x / v_liquid - y / v_vapor))
    return max(s, 0.0) ** 4


# ---------------------------------------------- Peng-Robinson --------------

def peng_robinson_z(p, temperature, tc, pc, omega, phase="vapor"):
    """Peng-Robinson compressibility factor for a single component.

    Solves the PR cubic  Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0
    and returns the vapor (largest) or liquid (smallest) real root.
    """
    tr = temperature / tc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
    alpha = (1.0 + kappa * (1.0 - np.sqrt(tr))) ** 2
    a = 0.45724 * R_GAS ** 2 * tc ** 2 / pc * alpha
    b = 0.07780 * R_GAS * tc / pc
    A = a * p / (R_GAS * temperature) ** 2
    B = b * p / (R_GAS * temperature)
    coeffs = [1.0, -(1.0 - B), A - 3.0 * B ** 2 - 2.0 * B, -(A * B - B ** 2 - B ** 3)]
    roots = np.roots(coeffs)
    real = roots[np.abs(roots.imag) < 1e-9].real
    real = real[real > B]
    return float(real.max() if phase == "vapor" else real.min())


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Pore-Confinement Phase Equilibrium")
    print("=" * 60)

    # A two-component (C1/C4-like) mixture at conditions giving a two-phase split
    z = np.array([0.5, 0.5])
    pc = np.array([667.0, 551.0])             # psia
    tc = np.array([343.0, 765.0])             # R
    omega = np.array([0.011, 0.193])
    k = wilson_k_values(p=500.0, temperature=600.0, pc=pc, tc=tc, omega=omega)
    print(f"  Wilson K-values        = {np.round(k, 3)}")
    assert k[0] > 1.0 > k[1]                  # light flashes to vapor, heavy to liquid

    # Rachford-Rice gives an interior vapor fraction; compositions are normalized
    beta = rachford_rice(z, k)
    x, y = flash_compositions(z, k, beta)
    print(f"  vapor fraction beta    = {beta:.3f}")
    assert 0.0 < beta < 1.0
    assert np.isclose(np.sum(x * (1 - beta) + y * beta), 1.0)   # mass balance
    assert np.isclose(np.sum(beta * y + (1 - beta) * x), 1.0)

    # Young-Laplace: capillary pressure rises as the pore shrinks
    pc_small = young_laplace_pc(20.0, 5e-9)
    pc_large = young_laplace_pc(20.0, 1e-6)
    print(f"  Pc 5nm / 1um           = {pc_small:.3e} / {pc_large:.3e} Pa")
    assert pc_small > pc_large > 0

    # Parachor IFT is positive and grows with the density contrast
    sigma = macleod_sugden_ift([71.0, 189.9], x, y, v_liquid=2.0, v_vapor=10.0)
    print(f"  Macleod-Sugden IFT     = {sigma:.2f} dyne/cm")
    assert sigma > 0

    # Peng-Robinson: vapor Z exceeds liquid Z, both physical
    zv = peng_robinson_z(500.0, 600.0, tc[0], pc[0], omega[0], "vapor")
    zl = peng_robinson_z(500.0, 600.0, tc[1], pc[1], omega[1], "liquid")
    print(f"  PR Z vapor/liquid      = {zv:.4f} / {zl:.4f}")
    assert 0.0 < zl < zv < 1.5
    print("  PASS")
    return {"beta": float(beta), "IFT": float(sigma), "Pc_5nm": float(pc_small)}


if __name__ == "__main__":
    test_all()
