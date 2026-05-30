"""
Article 6: Experimental Method for Time-Lapse Micro-CT Imaging of Mud-Filtrate
           Invasion and Mudcake Deposition
Schroeder, Torres-Verdin (2019)
DOI: 10.30632/PJV60N5-2019a6

A high-resolution micro-CT setup images mud-filtrate invasion and mudcake
buildup on a core under simulated wellbore conditions.  CT measures the X-ray
linear attenuation coefficient (Beer-Lambert law) at many projections; the
reconstructed voxel attenuations give porosity and saturation, and time-lapse
scans track the invasion front and the growing mudcake thickness.

Implements:

  - Beer-Lambert attenuation  I = I0*exp(-mu*x)
  - Linear attenuation coefficient from intensity
  - CT porosity / saturation from voxel attenuation endpoints
  - Mudcake thickness growth (sqrt-of-time filtration) and invasion front

Note: this issue's PDF has a text layer but the paper is an imaging/methods
study with no numbered equations; these are the standard Beer-Lambert / CT and
filtration relations the method is grounded in.  Paper anchors: 42.5-um voxels,
150 F, 100-psi overbalance, ~60-minute continuous scans.
"""

import numpy as np


# ---------------------------------------------- Beer-Lambert ------------

def beer_lambert(I0, mu, x):
    """Transmitted X-ray intensity  I = I0*exp(-mu*x)."""
    return I0 * np.exp(-np.asarray(mu, float) * x)


def attenuation_from_intensity(I0, I, x):
    """Linear attenuation coefficient  mu = ln(I0/I)/x."""
    return np.log(I0 / np.asarray(I, float)) / x


# ---------------------------------------------- CT petrophysics ---------

def ct_porosity(mu_voxel, mu_grain, mu_fluid):
    """Porosity from voxel attenuation  phi = (mu_grain - mu)/(mu_grain - mu_fluid)."""
    return (mu_grain - np.asarray(mu_voxel, float)) / (mu_grain - mu_fluid)


def ct_saturation(mu_voxel, mu_dry, mu_sat):
    """Fluid saturation from time-lapse attenuation change.

        S = (mu - mu_dry)/(mu_sat - mu_dry)
    where mu_dry/mu_sat are the dry and fully-saturated voxel attenuations.
    """
    return np.clip((np.asarray(mu_voxel, float) - mu_dry) / (mu_sat - mu_dry),
                   0.0, 1.0)


# ---------------------------------------------- filtration --------------

def mudcake_thickness(t, rate_const):
    """Static-filtration mudcake thickness  h = rate_const*sqrt(t)."""
    return rate_const * np.sqrt(np.asarray(t, float))


def invasion_front(t, q_flux, phi):
    """Radial-ish invasion front depth  x = q_flux*t/phi  (volume balance)."""
    return q_flux * np.asarray(t, float) / phi


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Time-Lapse Micro-CT Invasion & Mudcake")
    print("=" * 60)

    # Beer-Lambert round-trip
    I0, mu, x = 1.0, 50.0, 0.02
    I = beer_lambert(I0, mu, x)
    print(f"  transmitted fraction   = {I:.4f}")
    assert abs(attenuation_from_intensity(I0, I, x) - mu) < 1e-9
    # denser/thicker -> less transmission
    assert beer_lambert(I0, 80.0, x) < I

    # CT porosity from voxel attenuation between grain and fluid endpoints
    phi = ct_porosity(mu_voxel=0.7 * 1.0 + 0.3 * 0.2, mu_grain=1.0, mu_fluid=0.2)
    print(f"  CT porosity            = {phi:.3f}")
    assert abs(phi - 0.3) < 1e-9

    # CT saturation rises from 0 (dry) to 1 (saturated)
    assert abs(ct_saturation(0.2, 0.2, 0.6)) < 1e-9
    assert abs(ct_saturation(0.6, 0.2, 0.6) - 1.0) < 1e-9
    assert abs(ct_saturation(0.4, 0.2, 0.6) - 0.5) < 1e-9

    # Mudcake grows as sqrt(time); invasion front advances with time
    t = np.array([1.0, 4.0, 9.0, 16.0])
    h = mudcake_thickness(t, rate_const=1e-4)
    print(f"  mudcake h(t) ratio 4/1 = {h[1]/h[0]:.1f}  (sqrt -> 2)")
    assert abs(h[1] / h[0] - 2.0) < 1e-9          # quadruple time -> double cake
    assert np.all(np.diff(invasion_front(t, 1e-6, 0.2)) > 0)
    print("  PASS")
    return {"phi": float(phi), "mudcake_16": float(h[-1])}


if __name__ == "__main__":
    test_all()
