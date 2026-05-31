"""
Article 1: Untangling Acoustic Anisotropy
Market, Mejia, Mutlu, Shahri, Tudge (2015)
Reference: Petrophysics Vol. 56, No. 5 (October 2015), pp. 420-439
DOI: none assigned (this issue predates SPWLA DOI assignment)

A tutorial/review on separating the causes of acoustic anisotropy (intrinsic
bedding, stress-induced, fractures, geometric/bed-boundary).  Azimuthal sonic
data are processed by Alford rotation (crossed dipole) or azimuthal binning;
the fast/slow shear velocities, fast-shear azimuth and the shear anisotropy
magnitude are the primary outputs, cross-checked against compressional,
Stoneley, density and image responses.

Implements:

  - Shear anisotropy magnitude from fast/slow velocities (and from slowness)
  - Alford rotation of crossed-dipole (XX, XY, YX, YY) to fast/slow waveforms
  - Fast-shear azimuth by cross-component energy minimization
  - Thomsen shear-anisotropy parameter gamma from stiffnesses

Note: this is a tutorial/review; the relations below are the standard
azimuthal-sonic anisotropy tools it describes (Alford, 1986; Thomsen, 1986).
Velocities in m/s, slowness in us/ft, angles in radians unless noted.
"""

import numpy as np


# ---------------------------------------------- anisotropy magnitude --------------

def shear_anisotropy_velocity(v_fast, v_slow):
    """Shear anisotropy magnitude from velocities  100*(Vfast - Vslow)/Vslow  [%]."""
    return 100.0 * (v_fast - v_slow) / v_slow


def shear_anisotropy_slowness(dt_fast, dt_slow):
    """Shear anisotropy magnitude from slowness  100*(DTslow - DTfast)/DTfast  [%]

    (slowness is the inverse of velocity, so the slow shear has the larger DT).
    """
    return 100.0 * (dt_slow - dt_fast) / dt_fast


# ---------------------------------------------- Alford rotation --------------

def alford_rotation(xx, xy, yx, yy, theta):
    """Rotate crossed-dipole data to the principal (fast/slow) frame

        FP    = cos^2 t*XX + sin t cos t*(XY+YX) + sin^2 t*YY
        SP    = sin^2 t*XX - sin t cos t*(XY+YX) + cos^2 t*YY
        cross = sin t cos t*(YY-XX) + cos^2 t*XY - sin^2 t*YX,

    where t is the fast-shear azimuth.  Returns (FP, SP, cross).
    """
    c, s = np.cos(theta), np.sin(theta)
    xx, xy, yx, yy = (np.asarray(v, float) for v in (xx, xy, yx, yy))
    fp = c ** 2 * xx + s * c * (xy + yx) + s ** 2 * yy
    sp = s ** 2 * xx - s * c * (xy + yx) + c ** 2 * yy
    cross = s * c * (yy - xx) + c ** 2 * xy - s ** 2 * yx
    return fp, sp, cross


def fast_shear_azimuth(xx, xy, yx, yy, n_theta=361):
    """Fast-shear azimuth by minimizing the rotated cross-component energy
    (Alford energy-minimization method) over theta in [0, pi)."""
    thetas = np.linspace(0.0, np.pi, n_theta)
    energy = [np.sum(alford_rotation(xx, xy, yx, yy, t)[2] ** 2) for t in thetas]
    return float(thetas[int(np.argmin(energy))])


# ---------------------------------------------- Thomsen gamma --------------

def thomsen_gamma(c66, c44):
    """Thomsen shear-anisotropy parameter  gamma = (C66 - C44)/(2*C44)."""
    return (c66 - c44) / (2.0 * c44)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Untangling Acoustic Anisotropy")
    print("=" * 60)

    # Anisotropy magnitude is zero when fast == slow, positive otherwise
    assert shear_anisotropy_velocity(3000.0, 3000.0) == 0.0
    aniso = shear_anisotropy_velocity(3200.0, 3000.0)
    print(f"  shear anisotropy       = {aniso:.2f} %")
    assert aniso > 0
    # Velocity and slowness definitions agree (slow shear has larger DT)
    assert np.isclose(shear_anisotropy_slowness(100.0, 106.667), aniso, atol=0.05)

    # Synthesize a fast/slow medium with a known azimuth and recover it by Alford
    t = np.linspace(0, 1, 200)
    fast_src = np.sin(2 * np.pi * 5 * t) * np.exp(-3 * t)
    az = np.radians(40.0)
    c, s = np.cos(az), np.sin(az)
    xx = c ** 2 * fast_src
    yy = s ** 2 * fast_src
    xy = yx = s * c * fast_src
    az_fit = fast_shear_azimuth(xx, xy, yx, yy)
    print(f"  fast azimuth true/fit  = 40.0 / {np.degrees(az_fit):.1f} deg")
    assert np.isclose(np.degrees(az_fit), 40.0, atol=1.0)

    # Rotation at the fitted azimuth recovers the fast waveform, near-zero cross
    fp, sp, cross = alford_rotation(xx, xy, yx, yy, az_fit)
    assert np.allclose(fp, fast_src, atol=1e-6) and np.sum(cross ** 2) < 1e-6

    # Thomsen gamma is positive for a VTI shale (C66 > C44)
    g = thomsen_gamma(12.0, 8.0)
    print(f"  Thomsen gamma          = {g:.3f}")
    assert g > 0
    print("  PASS")
    return {"anisotropy_pct": float(aniso), "azimuth_deg": float(np.degrees(az_fit)), "gamma": float(g)}


if __name__ == "__main__":
    test_all()
