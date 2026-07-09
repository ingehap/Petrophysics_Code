"""
Article 2: New 4.75-in. Ultrasonic LWD Technology Provides High-Resolution
           Caliper and Imaging in Oil-Based and Water-Based Muds
Li, Lee, Coates, Jin, Wong (2019)
DOI: 10.30632/PJV60N6-2019a2

A pulse-echo ultrasonic LWD tool with four azimuthal transducers (90 deg apart)
on a 4.75-in. collar fires and listens to its own echo: the echo two-way travel
time times the mud sound speed gives the standoff (hence borehole radius and
caliper), and the echo amplitude (reflection coefficient) gives a borehole
image.  The four transducers let a least-squares fit remove tool eccentering.

Implements:

  - Pulse-echo standoff  standoff = c_mud * t_echo / 2
  - Borehole radius and caliper from standoff and collar radius
  - Acoustic impedance  Z = rho*c  and reflection coefficient R
  - Eccentering recovery from the 4 azimuthal standoffs (cosine fit)

Note: this is a hardware/field paper with essentially no numbered closed-form
equations; this module implements the standard pulse-echo standoff / impedance
relations the tool is built on.  Paper anchors: 4 transducers, 4.75-in. collar,
SNR > 10 dB beyond 2-in. standoff.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- standoff / caliper ------

def standoff(c_mud, t_echo):
    """Pulse-echo standoff  standoff = c_mud * t_echo / 2  (m).  t in s, c in m/s."""
    return c_mud * np.asarray(t_echo, float) / 2.0


def borehole_radius(standoff_m, collar_radius_m):
    """Borehole radius = collar radius + transducer standoff."""
    return collar_radius_m + np.asarray(standoff_m, float)


def caliper(standoffs, collar_radius_m):
    """Diameter from two opposed transducers  D = 2*collar + so_a + so_b."""
    so = np.asarray(standoffs, float)
    return 2.0 * collar_radius_m + so[0] + so[2]      # opposed pair (0 and 180)


# ---------------------------------------------- impedance / image -------

def acoustic_impedance(rho, c):
    """Acoustic impedance  Z = rho*c  (Rayl)."""
    return petrolib.acoustic_geomech.acoustic_impedance(rho, c)


def reflection_coefficient(Z1, Z2):
    """Normal-incidence reflection coefficient  R = (Z2-Z1)/(Z2+Z1)."""
    return petrolib.acoustic_geomech.reflection_coefficient(Z1, Z2)


# ---------------------------------------------- eccentering -------------

def fit_eccentering(azimuths_deg, standoffs):
    """Recover (mean_standoff, eccentricity, azimuth) from azimuthal standoffs.

        so(az) = so_mean + e*cos(az - phi)
    Least-squares fit on [1, cos(az), sin(az)].  Returns (so_mean, e, phi_deg).
    """
    az = np.radians(np.asarray(azimuths_deg, float))
    so = np.asarray(standoffs, float)
    M = np.vstack([np.ones_like(az), np.cos(az), np.sin(az)]).T
    c0, c1, c2 = np.linalg.lstsq(M, so, rcond=None)[0]
    e = np.hypot(c1, c2)
    phi = np.degrees(np.arctan2(c2, c1))
    return float(c0), float(e), float(phi)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Ultrasonic LWD Caliper & Imaging")
    print("=" * 60)

    c_mud = 1200.0                        # m/s in OBM
    # Standoff round-trip: 2-in. (0.0508 m) standoff -> echo time and back
    so = 0.0508
    t = 2.0 * so / c_mud
    print(f"  echo time @2in standoff = {t*1e6:.1f} us")
    assert abs(standoff(c_mud, t) - so) < 1e-12

    # Caliper from opposed transducers (collar 4.75 in. dia -> radius 0.0603 m)
    collar_r = 4.75 * 0.0254 / 2.0
    D = caliper([so, so, so, so], collar_r)
    print(f"  caliper                = {D/0.0254:.2f} in")
    assert abs(D - (2 * collar_r + 2 * so)) < 1e-12

    # Reflection: a hard formation reflects more strongly than a soft one
    Zmud = acoustic_impedance(1100.0, c_mud)
    R_hard = reflection_coefficient(Zmud, acoustic_impedance(2550.0, 5500.0))
    R_soft = reflection_coefficient(Zmud, acoustic_impedance(2100.0, 2400.0))
    print(f"  R hard / soft          = {R_hard:.3f} / {R_soft:.3f}")
    assert R_hard > R_soft > 0

    # Eccentering: plant a known eccentricity and azimuth, recover from 4 (or
    # more) azimuthal standoffs
    az = np.array([0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0])
    so_mean, e_true, phi_true = 0.04, 0.012, 35.0
    measured = so_mean + e_true * np.cos(np.radians(az - phi_true))
    so0, e, phi = fit_eccentering(az, measured)
    print(f"  recovered so/e/phi     = {so0:.3f} / {e:.4f} / {phi:.1f}")
    assert abs(so0 - so_mean) < 1e-9 and abs(e - e_true) < 1e-9
    assert abs(((phi - phi_true + 180) % 360) - 180) < 1e-6
    print("  PASS")
    return {"caliper_in": float(D / 0.0254), "R_hard": float(R_hard),
            "ecc": e}


if __name__ == "__main__":
    test_all()
