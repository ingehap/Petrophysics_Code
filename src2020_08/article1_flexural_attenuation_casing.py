"""
Article 1: A Study of the Flexural Attenuation Technique Through Laboratory
           Measurements and Numerical Simulations
Sirevaag, Johansen, Larsen, Holt (2020)
DOI: 10.30632/PJV61N4-2020a1

An ultrasonic pitch-catch bench (scaled from the field geometry) excites the
zero-order antisymmetric flexural mode A0 in a fluid-submerged steel casing and
measures its attenuation between receivers to characterize the material in the
annulus behind the casing.  Snell's law sets the optimal incidence angle, the
amplitude ratio between receivers gives the attenuation, and the third-interface
echo (TIE) timing yields the annulus thickness and casing eccentricity.

Implements:

  - Plane-wave phase shift / phase velocity  dphi = w*dt - w*dz/vphi   (Eqs. 1-2)
  - Snell optimal incidence angle  sin(theta) = vf/vphi                (Eq. 3)
  - Attenuation in dB  20*log10(A1/A2) and coefficient alpha           (Eq. 4)
  - Annulus traveltime / thickness from the TIE  x_a = s_a*cos(theta)  (Eqs. 5-7)
  - Cosine eccentricity fit  dt(az) = A*cos(az + phi) + t_avg          (Eq. 8)

Note: this issue's PDF text layer kept the equation numbers and variable
definitions but dropped the typeset glyphs, so these are the standard
plane-wave / Snell / spectral-ratio forms anchored to those definitions.
Paper anchors reproduced: fluid 1325 m/s, A0 phase velocity 2650 m/s -> the
30 deg incidence angle.  SI units; velocities m/s, distances m, times s.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

V_FLUID = 1325.0         # m/s, kerosene in the borehole
V_A0 = 2650.0            # m/s, A0 flexural phase velocity (lab)


# ---------------------------------------------- phase / group -----------

def phase_shift(omega, dt, dz, vphi):
    """Plane-wave phase shift  dphi = w*dt - k*dz,  k = w/vphi  (Eq. 1)."""
    return omega * dt - omega * dz / vphi


def phase_velocity(dz, dt):
    """Phase velocity from inter-receiver delay  vphi = dz/dt  (Eq. 1, solved)."""
    return dz / dt


def group_velocity(omega, k):
    """Group velocity  vg = dw/dk  from a dispersion curve  (Eq. 2)."""
    return np.gradient(np.asarray(omega, float), np.asarray(k, float))


# ---------------------------------------------- Snell -------------------

def snell_angle(vf=V_FLUID, vphi=V_A0):
    """Optimal incidence angle  theta = arcsin(vf/vphi)  (Eq. 3), in degrees."""
    return np.degrees(np.arcsin(vf / vphi))


# ---------------------------------------------- attenuation -------------

def attenuation_db(a1, a2):
    """Attenuation between receivers  20*log10(A1/A2)  (dB)  (Eq. 4)."""
    return petrolib.integrity_drilling.attenuation_db(a1, a2)


def attenuation_coefficient(a1, a2, dx):
    """Spatial attenuation coefficient from A = A0*exp(-alpha*x)  (1/m)."""
    return petrolib.integrity_drilling.attenuation_coefficient(a1, a2, dx)


# ---------------------------------------------- annulus / TIE -----------

def annulus_slant_path(dt, v_annulus, z_cas, v_group_cas):
    """Slant path in the annulus from the TIE time difference  (Eqs. 5-6).

        dt = 2*s_a/v_annulus - z_cas/v_group_cas
    Solve for s_a (the slant distance the wave travels in the annulus).
    """
    return 0.5 * v_annulus * (dt + z_cas / v_group_cas)


def annulus_thickness(s_a, theta_deg):
    """Annulus thickness  x_a = s_a*cos(theta)  (Eq. 7)."""
    return s_a * np.cos(np.radians(theta_deg))


def fit_eccentricity(az_deg, dt):
    """Least-squares cosine fit  dt(az) = A*cos(az + phi) + t_avg  (Eq. 8).

    Returns (A, phi_deg, t_avg).
    """
    az = np.radians(np.asarray(az_deg, float))
    dt = np.asarray(dt, float)
    # linear in [cos(az), sin(az), 1]:  A*cos(az+phi) = c1*cos(az) + c2*sin(az)
    M = np.vstack([np.cos(az), np.sin(az), np.ones_like(az)]).T
    c1, c2, t_avg = np.linalg.lstsq(M, dt, rcond=None)[0]
    A = np.hypot(c1, c2)
    phi = np.degrees(np.arctan2(-c2, c1))
    return float(A), float(phi), float(t_avg)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Flexural Attenuation Technique")
    print("=" * 60)

    # Snell's law reproduces the paper's 30 deg incidence angle
    theta = snell_angle()
    print(f"  optimal incidence angle = {theta:.1f} deg  (expect 30)")
    assert abs(theta - 30.0) < 1e-6

    # Phase velocity from a 3-mm receiver spacing and its delay
    dz = 3.0e-3
    dt = dz / V_A0
    print(f"  phase velocity         = {phase_velocity(dz, dt):.0f} m/s")
    assert abs(phase_velocity(dz, dt) - V_A0) < 1e-6

    # Group velocity from a synthetic (linear) dispersion -> constant slope
    k = np.linspace(100, 200, 11)
    omega = 2650.0 * k                     # nondispersive: vg = vphi
    vg = group_velocity(omega, k)
    assert np.allclose(vg, 2650.0)

    # Attenuation: exponential decay recovers the planted coefficient
    alpha_true = 50.0                      # 1/m
    x = 0.02                               # 2 cm between receivers
    a1, a2 = 1.0, np.exp(-alpha_true * x)
    print(f"  attenuation            = {attenuation_db(a1, a2):.2f} dB")
    assert abs(attenuation_coefficient(a1, a2, x) - alpha_true) < 1e-9
    assert attenuation_db(a1, a2) > 0      # amplitude drops with distance

    # Annulus thickness from a TIE time difference (round-trip)
    s_a = annulus_slant_path(dt=1.41e-6, v_annulus=1310.0, z_cas=15e-3,
                             v_group_cas=3100.0)
    x_a = annulus_thickness(s_a, theta)
    print(f"  annulus slant / thick  = {s_a*1e3:.2f} / {x_a*1e3:.2f} mm")
    assert s_a > 0 and 0 < x_a < s_a

    # Cosine eccentricity fit recovers planted amplitude / phase
    az = np.arange(10, 360, 20.0)
    dt_meas = 0.8 * np.cos(np.radians(az + 35.0)) + 5.0
    A, phi, t_avg = fit_eccentricity(az, dt_meas)
    print(f"  eccentricity A / phi   = {A:.3f} / {phi:.1f} deg")
    assert abs(A - 0.8) < 1e-6 and abs(t_avg - 5.0) < 1e-6
    assert abs(((phi - 35.0 + 180) % 360) - 180) < 1e-3
    print("  PASS")
    return {"theta": theta, "alpha": alpha_true, "x_a_mm": x_a * 1e3,
            "ecc_A": A}


if __name__ == "__main__":
    test_all()
