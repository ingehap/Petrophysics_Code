"""
Article 5: Coherent Interpretation of Wideband Electromagnetic Measurements in
           the Millihertz to Gigahertz Frequency Range
Seleznev, Hou, Freed, Habashy, Feng, Fellah, Xu, Nadeev (2018)
DOI: 10.30632/PJV59N3-2018a4

A single differential-effective-medium (DEM) complex-permittivity model fits a
rock's electromagnetic response coherently from spectral induced polarization
(mHz-kHz) to dielectric dispersion (MHz-GHz).  Each phase contributes a complex
permittivity (real dielectric part plus a conductivity term that dominates at
low frequency); grains are added incrementally into the brine host through a
Bruggeman mixing law, and the low-frequency limit recovers Archie's law.

Implements:

  - Radial frequency  omega = 2*pi*f
  - Complex permittivity  eps* = eps + i*sigma/(omega*eps0)
  - Depolarization factors of a spheroid (sphere limit L = 1/3)
  - Bruggeman symmetric effective-medium mixing of two phases
  - Archie formation factor / tortuosity  F = phi^(-m),  alpha = phi^(1-m)

Note: this issue's PDF has a text layer but its typeset display-equation glyphs
were dropped in extraction; the Archie limit (F = phi^-m) survived inline, and
the DEM / complex-permittivity relations (Eqs. 3-17) are faithful standard-form
reconstructions.  SI units; eps0 = vacuum permittivity.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

EPS0 = 8.8541878128e-12      # vacuum permittivity (F/m)


# ---------------------------------------------- complex permittivity --------------

def radial_frequency(f):
    """Radial frequency  omega = 2*pi*f."""
    return 2.0 * np.pi * np.asarray(f, float)


def complex_permittivity(eps_rel, sigma, omega):
    """Complex permittivity  eps* = eps + i*sigma/(omega*eps0)  (Eqs. 3-4, 10-12).

    The conductivity term dominates the imaginary part at low frequency and
    fades at high frequency.
    """
    return eps_rel + 1j * sigma / (omega * EPS0)


def depolarization_spheroid(aspect):
    """Depolarization factors (Lx, Ly, Lz) of a spheroid (Eq. 10).

    aspect = c/a is the polar/equatorial semi-axis ratio (1 = sphere -> 1/3
    each).  Oblate platelets (aspect < 1, e.g. clay) raise the symmetry-axis
    factor Lz; prolate needles (aspect > 1) lower it.  Closed-form integrals;
    the three factors always sum to 1.
    """
    return petrolib.em_dielectric.depolarization_spheroid(aspect)


# ---------------------------------------------- effective medium --------------

def bruggeman_two_phase(eps1, eps2, frac1):
    """Bruggeman symmetric effective permittivity of two phases (spheres)

        f1*(e1-e)/(e1+2e) + f2*(e2-e)/(e2+2e) = 0,

    solved in closed form (the quadratic root with the physical sign).  Works for
    real permittivities/conductivities and reduces to e1 (f1=1) or e2 (f1=0).
    """
    f2 = 1.0 - frac1
    # Quadratic 2e^2 - b*e - e1*e2 = 0 with b = (2f1-f2)e1 + (2f2-f1)e2:
    b = (2.0 * frac1 - f2) * eps1 + (2.0 * f2 - frac1) * eps2
    e = (b + np.sqrt(b ** 2 + 8.0 * eps1 * eps2)) / 4.0
    return e


def archie_formation_factor(phi, m=2.0):
    """Archie formation factor  F = phi^(-m)  (Eq. 17)."""
    return petrolib.saturation_resistivity.formation_factor(phi, m=m)


def archie_tortuosity(phi, m=2.0):
    """Archie tortuosity factor  alpha = F*phi = phi^(1-m)."""
    return np.asarray(phi, float) ** (1.0 - m)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Wideband EM DEM Permittivity")
    print("=" * 60)

    # Conductivity term dominates the imaginary permittivity at low frequency
    lo = complex_permittivity(20.0, 1.0, radial_frequency(1e-2))
    hi = complex_permittivity(20.0, 1.0, radial_frequency(1e9))
    print(f"  Im(eps*) at 0.01Hz/1GHz = {lo.imag:.3e} / {hi.imag:.3e}")
    assert lo.imag > hi.imag and np.isclose(hi.real, 20.0)

    # Depolarization factors sum to 1; sphere is 1/3 each
    lx, ly, lz = depolarization_spheroid(1.0)
    assert np.isclose(lx + ly + lz, 1.0) and np.isclose(lz, 1.0 / 3.0)
    lx2, _, lz2 = depolarization_spheroid(0.3)          # oblate clay platelet
    assert np.isclose(2 * lx2 + lz2, 1.0) and lz2 > 1.0 / 3.0

    # Bruggeman mixing lies between the endpoints and is exact at f=0/1
    em = bruggeman_two_phase(80.0, 5.0, 0.3)            # brine + rock matrix
    print(f"  Bruggeman eff. eps     = {em:.3f}")
    assert 5.0 < em < 80.0
    assert np.isclose(bruggeman_two_phase(80.0, 5.0, 1.0), 80.0)
    assert np.isclose(bruggeman_two_phase(80.0, 5.0, 0.0), 5.0)

    # Archie: F rises and tortuosity falls as porosity falls
    assert archie_formation_factor(0.1) > archie_formation_factor(0.3)
    assert np.isclose(archie_formation_factor(0.25, 2.0) * 0.25, archie_tortuosity(0.25, 2.0))
    print("  PASS")
    return {"eff_eps": float(em), "Lz_oblate": float(lz2)}


if __name__ == "__main__":
    test_all()
