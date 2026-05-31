"""
Article 4: Efficiency Improvements in Production Profiling Using Ultracompact
           Flow Array Sensing Technology
Abbassi, Tavernier, Donzier, Gysen, Gysen, Chen, Zeid, Cedillo (2018)
DOI: 10.30632/PJV59V4-2018a3

A compact MEMS array tool carries an array of 10-MHz ultrasonic transducers plus
optical/conductivity point probes to profile multiphase flow.  Each transducer
measures the Doppler frequency shift of sound scattered off moving fluid /
bubbles to give a local flow speed; a digital implementation reads the speed
from the position of the peak in the Doppler spectrum, and the conductivity
probes give a local water holdup from the bubble-count statistics.

Implements:

  - Ultrasonic Doppler flow speed  VF = Vs*(Fr - Fe)/(2*Fe*cos(alpha))
  - Digital Doppler speed from the spectral peak position (Appendix 6)
  - Conductivity-probe water holdup from the wetted-fraction count
  - Array averaging of local probe speeds into a profile mean

Note: this issue's PDF has a text layer but its typeset display-equation glyphs
were dropped in extraction, so the Doppler relations (Eq. 1 and Appendix Eq.
A6.1) are faithful standard-form reconstructions from the variables the paper
defines.  SI units (speeds m/s, frequencies Hz).
"""

import numpy as np

CENTRAL_FREQ = 10e6          # 10 MHz transducer
MAXIMUM_SHIFT = 128          # array digitization span (Appendix 6)


# ---------------------------------------------- doppler --------------

def doppler_velocity(fr, fe, sound_speed, alpha_deg):
    """Flow speed from the Doppler shift  VF = Vs*(Fr-Fe)/(2*Fe*cos(alpha))  (Eq. 1).

    Fr/Fe = received/emitted frequency, Vs = sound speed in the continuous
    phase, alpha = beam angle to the flow.  The factor 2 is the round-trip
    (emit + backscatter) shift.
    """
    return sound_speed * (fr - fe) / (2.0 * fe * np.cos(np.radians(alpha_deg)))


def digital_doppler_speed(max_pos, frequency_range, sound_speed, alpha_deg,
                          central_freq=CENTRAL_FREQ, maximum_shift=MAXIMUM_SHIFT):
    """Digital Doppler flow speed from the spectral peak position (Eq. A6.1)

        FS = (MaxPos/MaximumShift)*FrequencyRange*Vs/(2*CentralFreq*cos(alpha)).

    MaxPos is the bin of the peak energy in the Doppler spectrum; dividing by the
    digitization span and scaling by the acquisition FrequencyRange recovers the
    Doppler frequency, then the Doppler relation gives the speed.
    """
    doppler_freq = (max_pos / maximum_shift) * frequency_range
    return doppler_freq * sound_speed / (2.0 * central_freq * np.cos(np.radians(alpha_deg)))


# ---------------------------------------------- probes --------------

def conductivity_holdup(wetted_counts, total_counts):
    """Local water holdup from a conductivity probe = wetted fraction of samples."""
    return np.asarray(wetted_counts, float) / np.asarray(total_counts, float)


def array_mean_speed(local_speeds, areas=None):
    """Area-weighted mean of the array's local Doppler speeds (profile velocity)."""
    v = np.asarray(local_speeds, float)
    if areas is None:
        return float(v.mean())
    a = np.asarray(areas, float)
    return float((v * a).sum() / a.sum())


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Ultracompact Flow Array (Doppler)")
    print("=" * 60)

    # Round-trip a known flow speed through the Doppler relation
    vs, alpha = 1480.0, 30.0
    v_true = 1.2
    fe = CENTRAL_FREQ
    fr = fe + 2.0 * fe * v_true * np.cos(np.radians(alpha)) / vs   # invert Eq. 1
    v = doppler_velocity(fr, fe, vs, alpha)
    print(f"  Doppler speed          = {v:.3f} m/s  (true {v_true})")
    assert np.isclose(v, v_true, rtol=1e-9)

    # No shift -> no flow
    assert np.isclose(doppler_velocity(fe, fe, vs, alpha), 0.0)

    # Digital Doppler: peak position scales the recovered speed linearly
    s1 = digital_doppler_speed(40, 2.0e4, vs, alpha)
    s2 = digital_doppler_speed(80, 2.0e4, vs, alpha)
    print(f"  digital speed 40/80    = {s1:.3f} / {s2:.3f} m/s")
    assert np.isclose(s2, 2.0 * s1)

    # Conductivity holdup and area-weighted profile mean
    yw = conductivity_holdup(np.array([300, 600, 900]), np.array([1000, 1000, 1000]))
    assert np.allclose(yw, [0.3, 0.6, 0.9])
    vbar = array_mean_speed([1.0, 1.4, 1.8], areas=[1, 1, 2])
    print(f"  profile mean speed     = {vbar:.3f} m/s")
    assert np.isclose(vbar, (1.0 + 1.4 + 2 * 1.8) / 4)
    print("  PASS")
    return {"doppler_v": float(v), "profile_v": float(vbar)}


if __name__ == "__main__":
    test_all()
