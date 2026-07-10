"""
Article 4: Ultrasonic Logging of Creeping Shale
Diez, Johansen, Larsen (2022)
DOI: 10.30632/PJV63N1-2022a4

Laboratory pressure-cell monitoring of creeping Pierre II shale bonding to a
steel pipe ("casing") using two ultrasonic techniques - pulse echo (PE,
normal incidence) and pitch catch (PC, 29-degree leaky-Lamb wave).  Both
observables are mapped to the acoustic impedance of the material behind the
pipe via empirical bench calibrations.

Implements:

  - Group delay  tau(w) = -dphi/dw,  phi = arg(S_P / S_N)        (Eq. 1)
  - Thickness-resonance frequency  f_min = 0.95 * v_p / (2 d)    (Eq. 2)
  - PE impedance calibration  |tau_min| = a*Z + b                (Eq. 3)
  - PC impedance calibration  alpha = a*Z + b                    (Eq. 4)
  - Attenuation rate  alpha = (E_T - E) / L                      (aux)
  - Normal-incidence reflection coefficient                     (aux)

The bench-calibration slopes/intercepts are image-rendered in the paper and
were not in the machine-readable text; they are fit here to the numerical
anchor pairs quoted in the article body and flagged as reconstructions.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# Characteristic acoustic impedances (MRayl)
Z_PIPE = 43.5                # steel pipe ("casing")
Z_SHALE = 4.2               # Pierre II shale at consolidation stress
Z_KEROSENE = 1.1            # gap fill before creep closes the annulus

F_MIN_CORRECTION = 0.95     # S1-mode negative-group-velocity correction (Eq. 2)


# ---------------------------------------------- Eq. 1: group delay -------

def group_delay(S_P, S_N, freqs):
    """Group delay tau(w) = -dphi/dw with phi = arg(S_P / S_N)  (Eq. 1).

    S_P, S_N : complex spectra of the processing and normalization windows.
    freqs    : frequency axis (Hz) matching the spectra.
    Returns (tau, w) with tau in seconds on the midpoint frequency grid.
    """
    phi = np.unwrap(np.angle(np.asarray(S_P) / np.asarray(S_N)))
    w = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    tau = -np.diff(phi) / np.diff(w)
    w_mid = 0.5 * (w[1:] + w[:-1])
    return tau, w_mid


def group_delay_parameter(S_P, S_N, freqs):
    """The group-delay parameter |tau_min| (magnitude of the minimum, Eq. 1)."""
    tau, _ = group_delay(S_P, S_N, freqs)
    return float(abs(tau.min()))


# ---------------------------------------------- Eq. 2: f_min ------------

def thickness_resonance_freq(v_p, d, correction=F_MIN_CORRECTION):
    """f_min = 0.95 * v_p / (2 d)  (Eq. 2).  v_p [m/s], d [m] -> Hz."""
    return petrolib.integrity_drilling.casing_resonance_frequency(d, v=v_p, correction=correction)


def casing_velocity_from_fmin(f_min, d, correction=F_MIN_CORRECTION):
    """Invert Eq. 2 for the casing compressional velocity v_p [m/s]."""
    return 2.0 * d * f_min / correction


# ---------------------------------------------- Eqs. 3-4: calibrations --

def _fit_line(x_pts, y_pts):
    """Two-or-more point least-squares line  y = a*x + b -> (a, b)."""
    lf = petrolib.inversion_numerics.fitting.fit_line(x_pts, y_pts)
    return float(lf.slope), float(lf.intercept)


# PE bench anchors quoted in the paper: (|tau_min| [us], Z [MRayl])
_PE_TAU_US = [0.66, 0.80]
_PE_Z = [1.36, 0.80]
PE_SLOPE, PE_INTERCEPT = _fit_line(_PE_TAU_US, _PE_Z)   # Z = a*tau + b


def impedance_from_pe(tau_min_us):
    """PE impedance from group-delay parameter (Eq. 3).  tau in us -> MRayl."""
    return PE_SLOPE * float(tau_min_us) + PE_INTERCEPT


# PC bench anchors: (attenuation alpha [dB/m], Z [MRayl]); higher Z leaks
# more energy so impedance rises with attenuation.
_PC_ALPHA = [200.0, 700.0]
_PC_Z = [1.0, 3.6]
PC_SLOPE, PC_INTERCEPT = _fit_line(_PC_ALPHA, _PC_Z)    # Z = a*alpha + b


def impedance_from_pc(alpha_db_m):
    """PC impedance from Lamb-wave attenuation rate (Eq. 4).  dB/m -> MRayl."""
    return PC_SLOPE * float(alpha_db_m) + PC_INTERCEPT


# ---------------------------------------------- auxiliary --------------

def attenuation_rate(E_transmitted_db, E_measured_db, length_m):
    """alpha = (E_T - E) / L  [dB/m]."""
    return (E_transmitted_db - E_measured_db) / length_m


def reflection_coefficient(z_behind, z_pipe=Z_PIPE):
    """Normal-incidence reflection coefficient R = (Z2 - Z1)/(Z2 + Z1)."""
    return petrolib.acoustic_geomech.reflection_coefficient(z_pipe, z_behind)


def _synthetic_pe_spectra(d, v_p, freqs, gap_delay_s):
    """Build toy normalization / processing spectra with a thickness echo.

    The processing window adds a reverberation echo delayed by the pipe
    round-trip; this produces a group-delay minimum near f_min.
    """
    w = 2 * np.pi * freqs
    S_N = np.ones_like(freqs, dtype=complex)
    rt = 2.0 * d / v_p                       # pipe round-trip time
    S_P = S_N + 0.6 * np.exp(-1j * w * (rt + gap_delay_s))
    return S_P, S_N


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Ultrasonic Logging of Creeping Shale")
    print("=" * 60)

    d = 0.55e-3                  # pipe wall thickness, 0.55 mm
    f_lab = 5e6                  # ~5 MHz lab pulse

    # Eq. 2: implied steel velocity from f_min = f_lab is ~5.8 km/s
    v_p = casing_velocity_from_fmin(f_lab, d)
    f_min = thickness_resonance_freq(v_p, d)
    print(f"  v_p from f_min         = {v_p:.0f} m/s")
    print(f"  f_min (Eq. 2)          = {f_min/1e6:.2f} MHz")
    assert 5.0e3 < v_p < 6.5e3
    assert abs(f_min - f_lab) / f_lab < 1e-6

    # Eq. 1: group delay shows a clear minimum near f_min
    freqs = np.linspace(3e6, 7e6, 800)
    S_P, S_N = _synthetic_pe_spectra(d, v_p, freqs, gap_delay_s=0.0)
    tau, w_mid = group_delay(S_P, S_N, freqs)
    f_at_min = w_mid[np.argmin(tau)] / (2 * np.pi)
    print(f"  group-delay min at     = {f_at_min/1e6:.2f} MHz")
    assert 3e6 < f_at_min < 7e6

    # Eq. 3 (PE): anchor points are reproduced by the fitted calibration
    z1 = impedance_from_pe(0.66)
    z2 = impedance_from_pe(0.80)
    print(f"  PE: |tau|=0.66us -> Z  = {z1:.2f} MRayl")
    print(f"  PE: |tau|=0.80us -> Z  = {z2:.2f} MRayl")
    assert abs(z1 - 1.36) < 0.02 and abs(z2 - 0.80) < 0.02
    assert PE_SLOPE < 0          # impedance falls as group delay rises

    # attenuation rate -> Eq. 4 (PC)
    alpha = attenuation_rate(E_transmitted_db=14.0, E_measured_db=-12.0,
                             length_m=0.031)
    z_pc = impedance_from_pc(alpha)
    print(f"  PC: alpha={alpha:.0f} dB/m -> Z = {z_pc:.2f} MRayl")
    assert alpha > 0 and PC_SLOPE > 0

    # reflection coefficient: gap (kerosene) ~ -0.95, bonded shale ~ -0.82
    R_gap = reflection_coefficient(Z_KEROSENE)
    R_bond = reflection_coefficient(Z_SHALE)
    print(f"  R (kerosene gap)       = {R_gap:.2f}")
    print(f"  R (bonded shale)       = {R_bond:.2f}")
    assert abs(R_gap + 0.95) < 0.02
    assert abs(R_bond + 0.82) < 0.02
    assert R_bond > R_gap        # bonding raises (less negative) R
    print("  PASS")
    return {"v_p": v_p, "f_min": f_min, "R_gap": R_gap, "R_bond": R_bond}


if __name__ == "__main__":
    test_all()
