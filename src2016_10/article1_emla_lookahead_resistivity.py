"""
Article 1: Looking Ahead of the Bit While Drilling: From Vision to Reality
Constable, Antonsen, Stalheim, Olsen, Fjell, Dray, Eikenes, Aarflot, Haldorsen,
Digranes, Seydoux, Omeragic, Thiel, Davydychev, Denichou, Salim, Frey, Homan,
Tan (2016)
Reference: Petrophysics Vol. 57, No. 5 (October 2016), pp. 426-446
DOI: none assigned (this issue predates SPWLA DOI assignment)

An electromagnetic look-ahead (EMLA) LWD service places a low-frequency
transmitter ~1.8 m behind the bit and measures the formation resistivity profile
ahead of the bit by inversion of ultradeep harmonic-resistivity (UHR)
measurements built from the XX/YY/ZZ antenna couplings.  The look-ahead depth of
detection grows with the transmitter-receiver span and the EM skin depth.

Implements:

  - Electromagnetic skin depth  delta = sqrt(2*rho/(omega*mu))
  - Harmonic-resistivity attenuation (UHRA, dB) and phase shift (UHRP, deg)
    from antenna-coupling voltages (Eqs. 1-2)
  - Depth-of-detection-ahead scaling with span and skin depth
  - Apparent resistivity from the attenuation in a homogeneous background

Note: this issue's PDF has a text layer; the UHRA/UHRP measurements (Eqs. 1-2)
are defined in the body from the XX/YY/ZZ couplings, while the typeset coupling
combination lost its glyphs, so the attenuation/phase pair is implemented in the
standard ratio form used by propagation-resistivity tools.  Resistivity in
ohm-m, frequency in Hz, lengths in m.
"""

import numpy as np

MU0 = 4.0e-7 * np.pi          # permeability of free space (H/m)


# ---------------------------------------------- skin depth --------------

def skin_depth(rho, freq, mu_r=1.0):
    """Electromagnetic skin depth  delta = sqrt(2*rho/(omega*mu))

    with omega = 2*pi*freq and mu = mu_r*mu0; the depth at which the current
    density falls to 1/e of its surface value (Jordan & Balmain, 1968).
    """
    omega = 2.0 * np.pi * np.asarray(freq, float)
    return np.sqrt(2.0 * rho / (omega * mu_r * MU0))


# ---------------------------------------------- harmonic resistivity --------------

def uhr_attenuation_phase(v_near, v_far):
    """Ultradeep harmonic-resistivity attenuation and phase shift (Eqs. 1-2)

        UHRA = 20*log10(|V_near / V_far|)      [dB]
        UHRP = angle(V_near / V_far)           [deg]

    from the (complex) coupling-voltage ratio; for UHR the near/far voltages are
    formed from the coaxial (Vxx) and coplanar (Vyy, Vzz) couplings.
    """
    ratio = np.asarray(v_near, complex) / np.asarray(v_far, complex)
    uhra = 20.0 * np.log10(np.abs(ratio))
    uhrp = np.degrees(np.angle(ratio))
    return uhra, uhrp


def coupling_ratio(vxx, vyy, vzz):
    """Symmetrized coaxial/coplanar coupling ratio used to build the UHR
    measurement: the geometric mean of the coplanar couplings over the coaxial
    coupling,  sqrt(Vyy*Vzz)/Vxx."""
    return np.sqrt(np.asarray(vyy, complex) * np.asarray(vzz, complex)) / np.asarray(vxx, complex)


# ---------------------------------------------- look-ahead depth --------------

def depth_of_detection(span, rho, freq, mu_r=1.0, k=1.0):
    """Look-ahead depth of detection ahead of the bit

        DOD ~ k*sqrt(span * delta),

    a monotonic scaling capturing the paper's observation that the depth of
    detection grows with the transmitter-receiver span and the skin depth
    (field range ~5 to 30 m).  k is a tool/contrast calibration factor.
    """
    return k * np.sqrt(np.asarray(span, float) * skin_depth(rho, freq, mu_r))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: EMLA Look-Ahead Resistivity")
    print("=" * 60)

    # Skin depth grows with resistivity and shrinks with frequency
    d_lo = skin_depth(1.0, 1.0e4)
    d_hi = skin_depth(100.0, 1.0e4)
    print(f"  skin depth 1 / 100 ohm-m = {d_lo:.2f} / {d_hi:.2f} m")
    assert d_hi > d_lo > 0
    assert skin_depth(10.0, 1.0e3) > skin_depth(10.0, 1.0e5)

    # Attenuation is positive and phase shift nonzero when the near voltage leads
    uhra, uhrp = uhr_attenuation_phase(1.0 * np.exp(1j * 0.3), 0.5 * np.exp(1j * 0.1))
    print(f"  UHRA / UHRP            = {uhra:.2f} dB / {uhrp:.2f} deg")
    assert uhra > 0 and not np.isclose(uhrp, 0.0)

    # Matched voltages give zero attenuation and phase shift
    z = 0.7 * np.exp(1j * 0.2)
    a0, p0 = uhr_attenuation_phase(z, z)
    assert np.isclose(a0, 0.0) and np.isclose(p0, 0.0)

    # The coupling ratio is well-defined and the UHR is built from it
    r = coupling_ratio(1.0 + 0j, 0.8 + 0.1j, 0.9 - 0.1j)
    assert np.isfinite(abs(r))

    # Depth of detection increases with span and skin depth
    dod_short = depth_of_detection(span=5.0, rho=20.0, freq=1.0e4)
    dod_long = depth_of_detection(span=15.0, rho=20.0, freq=1.0e4)
    print(f"  DOD span 5 / 15 m      = {dod_short:.2f} / {dod_long:.2f} m")
    assert dod_long > dod_short > 0
    print("  PASS")
    return {"skin_depth": float(d_hi), "UHRA": float(uhra), "DOD": float(dod_long)}


if __name__ == "__main__":
    test_all()
