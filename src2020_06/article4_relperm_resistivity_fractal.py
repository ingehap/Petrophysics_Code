"""
Article 4: Evaluation of Relative Permeability From Resistivity Data for Fractal
           Porous Media
Shi, Meng, Liu, Zhang, Wang (2020)
DOI: 10.30632/PJV61N3-2020a4

Analytical fractal expressions are derived for the resistivity index and the
two-phase relative permeability from pore-size and fluid-distribution fractal
geometry (pore-size PDF + Hagen-Poiseuille + Darcy + Archie), then combined to
predict relative permeability directly from resistivity-index data.

Implements:

  - Pore-size fractal PDF  f(r) = Df*rmin^Df*r^(-Df-1)            (Eq. 1)
  - Pore fractal dimension  Df = De - ln(phi)/ln(rmin/rmax)       (Eq. 2)
  - Archie resistivity index  I = Ro/Rt = Sw^(-n)                 (Eq. 11)
  - Brooks-Corey / fractal wetting & nonwetting rel-perm          (Eqs. 22, 24)
  - Relative permeability from the resistivity index              (Eq. 23)

Note: this issue's PDF text layer kept the equation numbers and variable
definitions but dropped the typeset glyphs, so these are the standard fractal /
Brooks-Corey closed forms (with the fractal-Brooks-Corey link lambda = De - Df)
anchored to those definitions.  Paper anchors: base case phi = 0.2,
rmax/rmin = 1000; limestone Dt = 2.25 (phi = 0.19); Boise Dt = 2.1 (phi = 0.32).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------------------------------------------- fractal geometry --------

def fractal_dimension(phi, rmin, rmax, De=3.0):
    """Pore fractal dimension  Df = De - ln(phi)/ln(rmin/rmax)  (Eq. 2)."""
    return De - np.log(phi) / np.log(rmin / rmax)


def pore_size_pdf(r, rmin, Df):
    """Pore-size probability density  f(r) = Df*rmin^Df*r^(-Df-1)  (Eq. 1)."""
    r = np.asarray(r, float)
    return Df * rmin ** Df * r ** (-Df - 1.0)


def bc_lambda(Df, De=3.0):
    """Brooks-Corey pore-size-distribution index from the fractal dimension.

    lambda = De - Df  (0 < lambda < 1 for 2 < Df < 3 in 3-D).
    """
    return De - Df


# ---------------------------------------------- resistivity -------------

def resistivity_index(sw, n=2.0):
    """Archie resistivity index  I = Ro/Rt = Sw^(-n)  (Eq. 11)."""
    # NOTE: despite the name this is the power law in Sw — it maps to
    # resistivity_index_from_sw.
    return petrolib.saturation_resistivity.resistivity_index_from_sw(sw, n=n)


# ---------------------------------------------- relative permeability ---

def bc_krw(swe, lam):
    """Brooks-Corey wetting-phase rel-perm  krw = Swe^((2+3*lam)/lam)  (Eq. 22)."""
    swe = np.asarray(swe, float)
    return swe ** ((2.0 + 3.0 * lam) / lam)


def bc_krnw(swe, lam):
    """Brooks-Corey nonwetting rel-perm  krnw = (1-Swe)^2*(1-Swe^((2+lam)/lam))  (Eq. 24)."""
    swe = np.asarray(swe, float)
    return (1.0 - swe) ** 2 * (1.0 - swe ** ((2.0 + lam) / lam))


def krw_from_resistivity_index(I, n, swr, lam):
    """Wetting-phase rel-perm from the resistivity index  (Eq. 23).

    Inverts I = Sw^-n for Sw, normalizes, and applies the fractal Brooks-Corey
    wetting-phase model so krw is obtained directly from resistivity data.
    """
    sw = np.asarray(I, float) ** (-1.0 / n)
    swe = np.clip((sw - swr) / (1.0 - swr), 0.0, 1.0)
    return bc_krw(swe, lam)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Relative Permeability From Resistivity (Fractal)")
    print("=" * 60)

    # Fractal dimension for the base case (phi=0.2, rmax/rmin=1000)
    Df = fractal_dimension(0.2, rmin=1.0, rmax=1000.0, De=3.0)
    print(f"  fractal dimension Df   = {Df:.3f}")
    assert 2.0 < Df < 3.0 and abs(Df - 2.767) < 0.01

    # Pore-size PDF integrates to ~1 over [rmin, inf)
    r = np.linspace(1.0, 5000.0, 400000)
    integral = _trapezoid(pore_size_pdf(r, 1.0, Df), r)
    print(f"  PDF integral           = {integral:.3f}")
    assert abs(integral - 1.0) < 0.02

    # Resistivity index: unity at Sw=1, rises as Sw drops
    assert abs(resistivity_index(1.0, 2.0) - 1.0) < 1e-12
    assert resistivity_index(0.5, 2.0) > resistivity_index(0.8, 2.0)

    # Brooks-Corey endpoints and monotonicity
    lam = bc_lambda(Df)
    print(f"  Brooks-Corey lambda    = {lam:.3f}")
    assert abs(bc_krw(1.0, lam) - 1.0) < 1e-9 and abs(bc_krw(0.0, lam)) < 1e-12
    assert abs(bc_krnw(0.0, lam) - 1.0) < 1e-9 and abs(bc_krnw(1.0, lam)) < 1e-12
    swe = np.linspace(0.0, 1.0, 11)
    assert np.all(np.diff(bc_krw(swe, lam)) >= -1e-12)       # krw increases
    assert np.all(np.diff(bc_krnw(swe, lam)) <= 1e-12)       # krnw decreases

    # Relative permeability straight from the resistivity index: higher I
    # (lower water saturation) -> lower wetting-phase rel-perm
    n, swr = 2.0, 0.15
    krw_lowI = krw_from_resistivity_index(1.5, n, swr, lam)    # high Sw
    krw_hiI = krw_from_resistivity_index(10.0, n, swr, lam)    # low Sw
    print(f"  krw(I=1.5) / krw(I=10) = {krw_lowI:.3e} / {krw_hiI:.3e}")
    assert krw_lowI > krw_hiI

    # Real-rock fractal dimensions stay physical (limestone / Boise sandstone)
    Df_ls = fractal_dimension(0.19, 1.0, 1000.0, De=3.0)
    Df_boise = fractal_dimension(0.32, 1.0, 1000.0, De=3.0)
    print(f"  Df limestone / Boise   = {Df_ls:.3f} / {Df_boise:.3f}")
    assert 2.0 < Df_ls < 3.0 and 2.0 < Df_boise < 3.0
    print("  PASS")
    return {"Df": float(Df), "lambda": float(lam), "krw_lowI": float(krw_lowI)}


if __name__ == "__main__":
    test_all()
