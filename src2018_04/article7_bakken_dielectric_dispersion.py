"""
Article 7: Bakken Petroleum System Characterization Using Dielectric-Dispersion
           Logs
Han, Misra (2018)
DOI: 10.30632/PJV59N2-2018a6

Multifrequency (10 MHz-1 GHz) dielectric-dispersion logs are inverted with an
electromagnetic mixing model to estimate water saturation, water salinity, the
cementation index, and a homogeneity index.  The Lichtenecker-Rother power-law
mixing law combines the matrix, water, and hydrocarbon permittivities with an
exponent alpha that encodes the geometrical arrangement (alpha = 0.5 recovers
the homogeneous CRI model); the complex water permittivity carries the
conductivity term that dominates at low frequency.

Implements:

  - Lichtenecker-Rother mixing  eps^a = (1-phi)*eps_m^a + phi*Sw*eps_w^a + phi*(1-Sw)*eps_o^a
  - Complex water permittivity  eps*_w = eps_w' - i*Cw/(omega*eps0)
  - Inversion of the mixing law for water saturation
  - CRI (homogeneous) limit at alpha = 0.5

Note: this issue's PDF has a text layer but the mixing-law expressions (Eqs.
1-4) lost their typeset glyphs in extraction, so the relations are faithful
standard-form reconstructions from the surviving variable definitions.  alpha in
[-1, 1]; permittivities relative; omega = 2*pi*f.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

EPS0 = 8.8541878128e-12      # vacuum permittivity (F/m)


# ---------------------------------------------- mixing --------------

def lichtenecker_rother(phi, sw, eps_m, eps_w, eps_o=2.2, alpha=0.5):
    """Lichtenecker-Rother power-law permittivity (Eq. 1)

        eps = [ (1-phi)*eps_m^a + phi*Sw*eps_w^a + phi*(1-Sw)*eps_o^a ]^(1/a).

    alpha = geometrical/homogeneity index: 0.5 -> homogeneous CRI model, +1 ->
    layers parallel to the field, -1 -> perpendicular.
    """
    return petrolib.em_dielectric.crim(
        phi, sw, eps_w=eps_w, eps_hc=eps_o, eps_matrix=eps_m, alpha=alpha
    )


def complex_water_permittivity(eps_w_real, cw, omega):
    """Complex water permittivity  eps*_w = eps_w' - i*Cw/(omega*eps0)  (Eq. 3)."""
    return petrolib.em_dielectric.complex_permittivity(
        eps_w_real, sigma=cw, freq_hz=omega / (2.0 * np.pi)
    )


def water_saturation(eps_meas, phi, eps_m, eps_w, eps_o=2.2, alpha=0.5):
    """Invert the Lichtenecker-Rother law for water saturation

        Sw = [ eps_meas^a - (1-phi)*eps_m^a - phi*eps_o^a ] / [ phi*(eps_w^a - eps_o^a) ],

    clipped to [0, 1].
    """
    return petrolib.em_dielectric.sw_from_permittivity(
        eps_meas, phi, eps_w=eps_w, eps_hc=eps_o, eps_matrix=eps_m, alpha=alpha, clip=True
    )


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Bakken Dielectric Dispersion")
    print("=" * 60)

    # Higher water saturation raises the bulk permittivity
    e_dry = lichtenecker_rother(0.10, 0.3, eps_m=5.0, eps_w=60.0)
    e_wet = lichtenecker_rother(0.10, 0.9, eps_m=5.0, eps_w=60.0)
    print(f"  eps Sw=0.3 / 0.9       = {e_dry:.3f} / {e_wet:.3f}")
    assert e_wet > e_dry

    # alpha = 0.5 recovers the CRIM/CRI (square-root) mixing
    cri = ((1 - 0.1) * np.sqrt(5.0) + 0.1 * 0.5 * np.sqrt(60.0) + 0.1 * 0.5 * np.sqrt(2.2)) ** 2
    assert np.isclose(lichtenecker_rother(0.1, 0.5, 5.0, 60.0), cri)

    # Complex water permittivity: conductivity term dominates the imag part at low f
    ew = complex_water_permittivity(78.0, 10.0, 2 * np.pi * 1e7)
    assert ew.real == 78.0 and ew.imag < 0

    # Forward then invert recovers the planted water saturation
    eps_meas = lichtenecker_rother(0.12, 0.45, 5.0, 60.0, alpha=0.5)
    sw = water_saturation(eps_meas, 0.12, 5.0, 60.0, alpha=0.5)
    print(f"  recovered Sw           = {sw:.3f}  (true 0.450)")
    assert np.isclose(sw, 0.45)
    print("  PASS")
    return {"eps_wet": float(e_wet), "Sw": float(sw)}


if __name__ == "__main__":
    test_all()
