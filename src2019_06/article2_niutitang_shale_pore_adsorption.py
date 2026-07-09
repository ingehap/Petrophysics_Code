"""
Article 2: Composition of the Shales in Niutitang Formation at Huijunba Syncline
           and its Influence on Microscopic Pore Structure and Gas Adsorption
Fu, Xu, Tian, Qin, Yang (2019)
DOI: 10.30632/PJV60N3-2019a1

Shale composition (organic carbon, clay, quartz) controls the microscopic pore
structure and gas-adsorption capacity.  The pore structure is characterized from
low-pressure N2 adsorption with the BET surface area and the FHH fractal
dimension, and the methane adsorption capacity follows a Langmuir isotherm that
scales with TOC and clay content.

Implements:

  - BET surface area from the linearized isotherm
  - FHH fractal dimension from the log-log adsorption slope
  - Langmuir methane adsorption capacity
  - Adsorption capacity vs composition (TOC + clay)

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the journal metadata and these are faithful
standard-form reconstructions (BET, FHH, Langmuir) of the pore-structure /
adsorption characterization the paper applies.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

NA = 6.022e23
N2_CROSS_NM2 = 0.162
V_MOLAR_STP = 22414.0


# ---------------------------------------------- BET ---------------------

def bet_surface_area(p_rel, v_ads, cross_nm2=N2_CROSS_NM2):
    """Monolayer volume Vm, BET constant C, and specific surface area (m^2/g).

    Linearizes  x/(v(1-x)) = 1/(Vm*C) + (C-1)/(Vm*C)*x  over the BET range.
    """
    x = np.asarray(p_rel, float); v = np.asarray(v_ads, float)
    y = x / (v * (1.0 - x))
    lf = petrolib.inversion_numerics.fitting.fit_line(x, y)
    slope, intercept = lf.slope, lf.intercept
    vm = 1.0 / (slope + intercept)
    c = slope / intercept + 1.0
    ssa = (vm / V_MOLAR_STP) * NA * (cross_nm2 * 1e-18)
    return float(vm), float(c), float(ssa)


# ---------------------------------------------- FHH fractal -------------

def fhh_fractal_dimension(p_rel, v_ads):
    """Frenkel-Halsey-Hill fractal dimension from ln(V) vs ln(ln(1/x)).

        ln V = A*ln(ln(1/x)) + const,  D = 3 + A  (surface fractal, A in (-1,0))
    """
    x = np.asarray(p_rel, float); v = np.asarray(v_ads, float)
    X = np.log(np.log(1.0 / x))
    Y = np.log(v)
    A = petrolib.inversion_numerics.fitting.fit_line(X, Y).slope
    return 3.0 + A


# ---------------------------------------------- adsorption --------------

def langmuir_adsorption(VL, PL, P):
    """Langmuir methane adsorption  V = VL*P/(PL + P)  (cm^3/g)."""
    return petrolib.geochem_fluids.adsorption.langmuir(P, VL, PL)


def langmuir_volume_from_composition(toc, clay_frac, a=1.6, b=0.2):
    """Langmuir volume VL scaling with TOC and clay  VL = a*TOC + b*clay (cm^3/g).

    Organic carbon is the dominant adsorbent; clay micropores add a smaller term.
    """
    return a * toc * 100.0 + b * clay_frac * 100.0    # toc, clay as fractions


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Niutitang Shale Pore Structure & Adsorption")
    print("=" * 60)

    # BET round-trip: plant Vm, C, forward an isotherm, recover them
    vm_t, c_t = 4.0, 90.0
    x = np.linspace(0.05, 0.30, 8)
    v = vm_t * c_t * x / ((1 - x) * (1 + (c_t - 1) * x))
    vm, c, ssa = bet_surface_area(x, v)
    print(f"  BET Vm/C/SSA           = {vm:.2f} / {c:.0f} / {ssa:.2f} m2/g")
    assert abs(vm - vm_t) < 1e-2 and ssa > 0.01

    # FHH: a rougher (more fractal) surface gives D closer to 3
    xr = np.linspace(0.5, 0.95, 12)
    # synthesize an FHH-consistent isotherm with A = -0.4 (D = 2.6)
    v_fhh = np.exp(-0.4 * np.log(np.log(1.0 / xr)) + 1.0)
    D = fhh_fractal_dimension(xr, v_fhh)
    print(f"  FHH fractal dimension  = {D:.2f}")
    assert 2.0 < D < 3.0 and abs(D - 2.6) < 0.05

    # Langmuir adsorption: half VL at PL, plateau at high P
    assert abs(langmuir_adsorption(10.0, 4.0, 4.0) - 5.0) < 1e-9
    assert langmuir_adsorption(10.0, 4.0, 1e4) > langmuir_adsorption(10.0, 4.0, 4.0)

    # Adsorption capacity rises with TOC and clay content
    vl_rich = langmuir_volume_from_composition(0.06, 0.30)
    vl_lean = langmuir_volume_from_composition(0.02, 0.30)
    print(f"  VL TOC-rich / lean     = {vl_rich:.2f} / {vl_lean:.2f} cm3/g")
    assert vl_rich > vl_lean
    print("  PASS")
    return {"BET_SSA": ssa, "FHH_D": float(D), "VL_rich": float(vl_rich)}


if __name__ == "__main__":
    test_all()
