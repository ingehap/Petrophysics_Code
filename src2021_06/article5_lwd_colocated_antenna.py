"""
Article 5: First LWD Co-Located Antenna Sensors for Real-Time Anisotropy and
           Dip Angle Determination, Yielding Better Look-Ahead Detection
Bittar, Wu, Ma, Pan, Fan, Griffing, Lozinsky (2021)
DOI: 10.30632/PJV62N3-2021a4

The first LWD resistivity tool with co-located (concentric, same-depth) tilted
antennas, so the full 3x3 magnetic-field tensor is measured at one point.  This
enables real-time horizontal/vertical resistivity anisotropy, relative dip, and
azimuthal/geosignal directional responses.

Implements:

  - Tilted-antenna magnetic moment projected onto tool axes
  - Magnetic tensor coupling  V = m_R . H . m_T
  - Magnetic-field tensor assembly (3x3)
  - Propagation-resistivity attenuation & phase shift (skin depth)
  - Apparent resistivity from phase shift / attenuation

Note: the paper introduces the tool and validates it by 3D/1D EM modeling but
prints no closed-form equations; the tilted-coil projection and tensor coupling
are transcribed from its figures, and the attenuation/phase apparent-resistivity
relations are the standard propagation-resistivity forms it relies on (flagged).
Resistivity in ohm-m, frequency in Hz, lengths in metres, angles in degrees.
"""

import numpy as np

MU0 = 4e-7 * np.pi


# ---------------------------------------------- tilted-coil geometry ----

def tilted_moment(tilt_deg, azimuth_deg=0.0):
    """Unit magnetic moment of a tilted antenna projected onto (x, y, z).

    At azimuth 0 a 45-deg coil gives (sin t, 0, cos t) = T'_x x + T'_z z;
    rotating the tool spins the transverse part around z (Figs. 4-5).
    """
    t, a = np.radians(tilt_deg), np.radians(azimuth_deg)
    return np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)])


def tensor_coupling(m_R, H, m_T):
    """Tilted-coil voltage from the magnetic tensor  V = m_R . H . m_T."""
    return float(np.asarray(m_R) @ np.asarray(H) @ np.asarray(m_T))


def assemble_tensor(Hxx, Hyy, Hzz, Hxy=0.0, Hxz=0.0, Hyz=0.0,
                    Hyx=None, Hzx=None, Hzy=None):
    """Assemble the full 3x3 magnetic-field tensor."""
    Hyx = Hxy if Hyx is None else Hyx
    Hzx = Hxz if Hzx is None else Hzx
    Hzy = Hyz if Hzy is None else Hzy
    return np.array([[Hxx, Hxy, Hxz], [Hyx, Hyy, Hyz], [Hzx, Hzy, Hzz]])


# ---------------------------------------------- propagation resistivity -

def skin_depth(rho, freq_hz):
    """EM skin depth  delta = sqrt(2*rho/(omega*mu0))  (metres)."""
    omega = 2 * np.pi * freq_hz
    return np.sqrt(2.0 * rho / (omega * MU0))


def phase_shift_deg(rho, freq_hz, spacing_m):
    """Phase shift across the receiver spacing  = spacing/delta  (degrees)."""
    return np.degrees(spacing_m / skin_depth(rho, freq_hz))


def attenuation_db(rho, freq_hz, spacing_m):
    """Amplitude attenuation across the spacing  = 8.686*spacing/delta  (dB)."""
    return 8.686 * spacing_m / skin_depth(rho, freq_hz)


def resistivity_from_phase(phase_deg, freq_hz, spacing_m):
    """Invert the phase shift for apparent resistivity (ohm-m)."""
    omega = 2 * np.pi * freq_hz
    delta = spacing_m / np.radians(phase_deg)
    return delta ** 2 * omega * MU0 / 2.0


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: LWD Co-Located Antenna Anisotropy / Dip")
    print("=" * 60)

    # Tilted-coil projection: 45 deg at azimuth 0 -> equal x and z components
    m = tilted_moment(45.0, 0.0)
    print(f"  45deg coil moment      = {np.round(m, 3)}")
    assert abs(m[0] - np.sqrt(0.5)) < 1e-9 and abs(m[2] - np.sqrt(0.5)) < 1e-9
    assert abs(m[1]) < 1e-12
    # rotating to azimuth 90 swaps transverse into y
    m90 = tilted_moment(45.0, 90.0)
    assert abs(m90[1] - np.sqrt(0.5)) < 1e-9 and abs(m90[0]) < 1e-9

    # Coupling of co-axial coils (both along z) selects Hzz
    H = assemble_tensor(1.018, 1.011, 0.9964, Hxz=0.0351, Hzx=0.0351)
    z = np.array([0.0, 0.0, 1.0])
    assert abs(tensor_coupling(z, H, z) - 0.9964) < 1e-9
    # near-isotropic case: Hxx ~ Hyy, off-diagonals small
    assert abs(H[0, 0] - H[1, 1]) < 0.02 and abs(H[0, 1]) < 1e-6

    # Propagation resistivity round-trips through phase shift
    rho, f, dz = 10.0, 2e6, 0.61          # 10 ohm-m, 2 MHz, 24-in spacing
    ph = phase_shift_deg(rho, f, dz)
    att = attenuation_db(rho, f, dz)
    rho_rec = resistivity_from_phase(ph, f, dz)
    print(f"  phase / attenuation    = {ph:.1f} deg / {att:.2f} dB")
    print(f"  recovered resistivity  = {rho_rec:.3f} ohm-m  (true {rho})")
    assert abs(rho_rec - rho) < 1e-6

    # More conductive formation -> larger phase shift and attenuation
    assert phase_shift_deg(1.0, f, dz) > phase_shift_deg(100.0, f, dz)
    assert attenuation_db(1.0, f, dz) > attenuation_db(100.0, f, dz)
    print("  PASS")
    return {"moment": m.tolist(), "phase_deg": ph, "rho_rec": rho_rec}


if __name__ == "__main__":
    test_all()
