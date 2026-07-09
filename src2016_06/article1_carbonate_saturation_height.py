"""
Article 1: Heterogeneous Carbonate Reservoirs: Ensuring Consistency of Subsurface
           Models by Maximizing the use of Saturation-Height Models and Dynamic
           Data
Hulea, Frese, Ramaswami (2016)
Reference: Petrophysics Vol. 57, No. 3 (June 2016), pp. 223-232
DOI: none assigned (this issue predates SPWLA DOI assignment)

Saturation-height models (SHM) and dynamic data are combined to keep carbonate
subsurface models consistent.  A Brooks-Corey SHM (entry pressure, irreducible
water, shape factor) is fit to MICP curves and its parameters correlated to
porosity/permeability; where capillary data are scarce the SHM can be calibrated
to dynamic data.  Permeability is predicted by averaging routine-core-analysis
data (arithmetic / geometric / harmonic) per rock type and calibrated against
wireline-formation-tester mobility (Mdd = k/viscosity).

Implements:

  - Brooks-Corey saturation-height model  Sw = Swi + (1-Swi)*(Pce/Pc)^(1/N)  (Eq. 1)
  - Buoyancy capillary pressure from height above the free-water level
  - Permeability averaging per rock type (arithmetic / geometric / harmonic)
  - Lucia-type permeability transform  log k = a*log(phi) + b
  - WFT mobility-to-permeability  k = Mdd*viscosity

Note: this issue's PDF has a text layer; the Brooks-Corey SHM (Eq. 1) and the
averaging / Lucia / mobility relations are transcribed from the body, while the
typeset glyphs were dropped and reconstructed in standard form.  Pc in bar (or
consistent), permeability in mD, porosity as a fraction, mobility in mD/cP.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

GRAV_PSI_PER_FT = 0.433       # psi/ft per unit specific gravity (water column)


# ---------------------------------------------- saturation height --------------

def brooks_corey_sw(pc, pce, swi, n):
    """Brooks-Corey saturation-height model (Eq. 1)

        Sw = Swi + (1 - Swi)*(Pce/Pc)^(1/N)   for Pc >= Pce,  else Sw = 1,

    with capillary entry pressure Pce, irreducible water Swi and shape factor N.
    """
    # 1/N (reciprocal) convention: the library's pore-size index lam = 1/N
    # (section 9 hazard; lam and N are NOT interchangeable).
    return petrolib.capillary_pressure.brooks_corey_sw(
        pc, pc_entry=pce, lam=1.0 / n, swirr=swi)


def buoyancy_pc(height_above_fwl, sg_water, sg_hc):
    """Buoyancy capillary pressure at a height above the free-water level

        Pc = 0.433*(SGw - SGhc)*HAFWL    [psi],

    the equilibrium relation Pc = (rho_w - rho_hc)*g*h (psi/ft form).
    """
    return petrolib.capillary_pressure.buoyancy_pc_gradient(
        height_above_fwl, sg_water=sg_water, sg_hc=sg_hc,
        gradient_psi_per_ft=GRAV_PSI_PER_FT)


# ---------------------------------------------- permeability --------------

def permeability_average(k_values, method="geometric"):
    """Average a set of rock-type permeabilities (arithmetic / geometric /
    harmonic); geometric is the paper's preferred carbonate transform."""
    k = np.asarray(k_values, float)
    if method == "arithmetic":
        return float(np.mean(k))
    if method == "geometric":
        return float(np.exp(np.mean(np.log(k))))
    if method == "harmonic":
        return float(len(k) / np.sum(1.0 / k))
    raise ValueError(f"unknown method: {method}")


def lucia_permeability(phi, a, b):
    """Lucia-type permeability transform  log10(k) = a*log10(phi) + b  ->
    k = 10^(a*log10(phi) + b)."""
    return 10.0 ** (a * np.log10(np.asarray(phi, float)) + b)


def mobility_to_permeability(mobility, viscosity):
    """Permeability from WFT mobility  k = Mdd*viscosity  (Mdd = k/viscosity)."""
    return mobility * viscosity


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Carbonate Saturation-Height Models")
    print("=" * 60)

    # Brooks-Corey: Sw = 1 below entry pressure, decreasing toward Swi above it
    assert np.isclose(brooks_corey_sw(0.5, pce=1.0, swi=0.15, n=2.0), 1.0)
    sw_lo = brooks_corey_sw(2.0, pce=1.0, swi=0.15, n=2.0)
    sw_hi = brooks_corey_sw(20.0, pce=1.0, swi=0.15, n=2.0)
    print(f"  Sw @2/20 bar           = {sw_lo:.3f} / {sw_hi:.3f}")
    assert 0.15 < sw_hi < sw_lo < 1.0

    # As Pc -> infinity Sw approaches the irreducible saturation
    assert np.isclose(brooks_corey_sw(1e6, 1.0, 0.15, 2.0), 0.15, atol=1e-3)

    # Buoyancy Pc grows with height above the FWL
    assert buoyancy_pc(300.0, 1.05, 0.3) > buoyancy_pc(50.0, 1.05, 0.3) > 0

    # Permeability averages are ordered harmonic <= geometric <= arithmetic
    ks = [1.0, 10.0, 100.0]
    ka = permeability_average(ks, "arithmetic")
    kg = permeability_average(ks, "geometric")
    kh = permeability_average(ks, "harmonic")
    print(f"  k arith/geo/harm       = {ka:.1f} / {kg:.1f} / {kh:.2f} mD")
    assert kh <= kg <= ka and np.isclose(kg, 10.0)

    # Lucia transform increases permeability with porosity
    assert lucia_permeability(0.25, a=8.0, b=2.0) > lucia_permeability(0.10, a=8.0, b=2.0)

    # Mobility-to-permeability inverts Mdd = k/viscosity
    assert np.isclose(mobility_to_permeability(50.0, 0.4), 20.0)
    print("  PASS")
    return {"Sw_hi": float(sw_hi), "kg": kg, "ka": ka}


if __name__ == "__main__":
    test_all()
