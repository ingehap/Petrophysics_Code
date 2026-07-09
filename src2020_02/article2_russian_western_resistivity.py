"""
Article 2: Comparison of the Russian and Western Resistivity Logs in Typical
           Western Siberian Reservoir Environments - A Numerical Study
Epov, Sukhorukova, Nechaev, Petrov, Rabinovich, Weston, Tyurin, Wang,
Abubakar, Claverie (2020)
DOI: 10.30632/PJV61N1-2020a1

A finite-element forward-modeling comparison of Russian electromagnetic /
electrical tools (VEMKZ/VIKIZ induction, BKZ lateral, laterolog) against Western
array-induction / laterolog / triaxial tools in anisotropic, invaded clastic
reservoirs.  Galvanic tools solve a Poisson problem for the potential; induction
tools solve the curl-curl equation; the formation anisotropy and the EM skin
depth control what each tool sees.

Implements:

  - Galvanic apparent resistivity  rho_a = k*U/I
  - Layered apparent resistivity (parallel / series anisotropy)
  - Anisotropy coefficient  lambda = sqrt(rho_v/rho_h)
  - EM skin depth  delta = 503*sqrt(rho/f)  (induction DOI vs frequency)

Note: this issue's PDF text layer kept the equation numbers and variable
definitions but dropped the typeset glyphs, so these are the standard
galvanic / induction / anisotropy relations anchored to those definitions.
Paper anchors: VEMKZ 0.875-14 MHz, IK induction 50-100 kHz, mud R0 = 1.4 ohm-m.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- galvanic ----------------

def apparent_resistivity_galvanic(U, I, k):
    """Galvanic (laterolog/BKZ) apparent resistivity  rho_a = k*U/I."""
    return k * np.asarray(U, float) / np.asarray(I, float)


def horizontal_resistivity(layer_res, layer_frac):
    """Parallel (horizontal-current) resistivity  1/Rh = sum(v_i/R_i)."""
    r = np.asarray(layer_res, float); v = np.asarray(layer_frac, float)
    return 1.0 / np.sum(v / r)


def vertical_resistivity(layer_res, layer_frac):
    """Series (vertical-current) resistivity  Rv = sum(v_i*R_i)."""
    r = np.asarray(layer_res, float); v = np.asarray(layer_frac, float)
    return float(np.sum(v * r))


# ---------------------------------------------- anisotropy --------------

def anisotropy_coefficient(rho_h, rho_v):
    """Anisotropy coefficient  lambda = sqrt(rho_v/rho_h)."""
    return petrolib.em_dielectric.anisotropy_coefficient(rho_h, rho_v)


# ---------------------------------------------- induction ---------------

def skin_depth(rho, freq_hz):
    """EM skin depth  delta = sqrt(2*rho/(w*mu0)) = 503*sqrt(rho/f)  (m)."""
    return petrolib.em_dielectric.skin_depth(rho, freq_hz)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Russian vs Western Resistivity Logs")
    print("=" * 60)

    # Galvanic apparent resistivity scales with the measured potential
    ra = apparent_resistivity_galvanic(U=2.0, I=1.0, k=5.0)
    assert abs(ra - 10.0) < 1e-12

    # Laminated anisotropy: Rv >= Rh always, and lambda >= 1
    res = np.array([1.0, 50.0])              # shale / sand
    frac = np.array([0.5, 0.5])
    Rh = horizontal_resistivity(res, frac)
    Rv = vertical_resistivity(res, frac)
    lam = anisotropy_coefficient(Rh, Rv)
    print(f"  Rh / Rv / lambda       = {Rh:.2f} / {Rv:.2f} / {lam:.2f}")
    assert Rv > Rh and lam > 1.0
    # an isotropic bed has lambda = 1
    assert abs(anisotropy_coefficient(10.0, 10.0) - 1.0) < 1e-12

    # Skin depth: the 503*sqrt(rho/f) rule and frequency dependence
    assert abs(skin_depth(1.0, 1.0) - 503.3) < 1.0
    d_vemkz = skin_depth(10.0, 14e6)         # 14 MHz Russian induction (shallow)
    d_ik = skin_depth(10.0, 50e3)            # 50 kHz induction (deeper)
    print(f"  skin depth 14MHz/50kHz = {d_vemkz:.3f} / {d_ik:.2f} m")
    assert d_ik > d_vemkz                    # lower frequency reads deeper

    # Higher mud/formation resistivity -> deeper penetration at fixed frequency
    assert skin_depth(50.0, 1e6) > skin_depth(5.0, 1e6)
    print("  PASS")
    return {"Rh": Rh, "Rv": Rv, "lambda": float(lam),
            "skin_14MHz": float(d_vemkz)}


if __name__ == "__main__":
    test_all()
