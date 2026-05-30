"""
Article 5: Application of a Fast NMR T1 Relaxation Time Measurement to
           Sedimentary Rock Cores
Mitchell, Valori (2019)
DOI: 10.30632/PJV60N3-2019a4

Longitudinal (T1) relaxation is informative but slow to measure by the classic
inversion-recovery sequence, which waits ~5*T1 between scans.  A fast T1 method
samples the recovery at a few well-chosen delays and fits the exponential,
recovering T1 in a fraction of the acquisition time with acceptable accuracy.

Implements:

  - Inversion-recovery and saturation-recovery magnetization models
  - Full multi-point T1 fit (log-linearized / least squares)
  - Fast few-point T1 estimate and its accuracy
  - Acquisition-time comparison (fast vs full)

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the journal metadata and these are faithful
standard-form reconstructions of the NMR T1 relaxation / fast-measurement
methods the paper applies.  T1 and times in seconds.
"""

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------- recovery models ---------

def inversion_recovery(t, M0, T1):
    """Inversion-recovery magnetization  M(t) = M0*(1 - 2*exp(-t/T1))."""
    return M0 * (1.0 - 2.0 * np.exp(-np.asarray(t, float) / T1))


def saturation_recovery(t, M0, T1):
    """Saturation-recovery magnetization  M(t) = M0*(1 - exp(-t/T1))."""
    return M0 * (1.0 - np.exp(-np.asarray(t, float) / T1))


# ---------------------------------------------- T1 fitting --------------

def fit_t1_saturation(t, M):
    """Fit (T1, M0) to a saturation-recovery curve by nonlinear least squares.

    Jointly fits M(t) = M0*(1 - exp(-t/T1)); returns (T1, M0).
    """
    t = np.asarray(t, float); M = np.asarray(M, float)
    popt, _ = curve_fit(lambda tt, M0, T1: M0 * (1.0 - np.exp(-tt / T1)),
                        t, M, p0=[M.max(), 1.0], maxfev=10000)
    return float(popt[1]), float(popt[0])


def fast_t1_two_point(t1_delay, M1, t2_delay, M2, M0):
    """Fast two-point T1 from two saturation-recovery samples.

        T1 = (t2 - t1) / ln[(M0 - M1)/(M0 - M2)]
    """
    return (t2_delay - t1_delay) / np.log((M0 - M1) / (M0 - M2))


def acquisition_time(delays, repeats=1, recycle_factor=5.0, T1_guess=1.0):
    """Approximate acquisition time = repeats * sum(delay + recycle*T1_guess)."""
    delays = np.asarray(delays, float)
    return repeats * np.sum(delays + recycle_factor * T1_guess)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Fast NMR T1 Measurement")
    print("=" * 60)

    T1_true, M0 = 0.8, 1.0

    # Inversion recovery crosses zero at t = T1*ln(2)
    t_zero = T1_true * np.log(2.0)
    assert abs(inversion_recovery(t_zero, M0, T1_true)) < 1e-9

    # Saturation recovery reaches ~63% of M0 at t = T1
    assert abs(saturation_recovery(T1_true, M0, T1_true) - M0 * (1 - np.exp(-1))) < 1e-9

    # Full multi-point fit recovers T1 from a saturation-recovery curve
    t = np.linspace(0.05, 4.0, 20)
    M = saturation_recovery(t, M0, T1_true)
    T1_fit, M0_fit = fit_t1_saturation(t, M)
    print(f"  full-fit T1            = {T1_fit:.3f} s  (true {T1_true})")
    assert abs(T1_fit - T1_true) < 0.05

    # Fast two-point estimate is close to the true T1
    t1d, t2d = 0.4, 1.6
    M1 = saturation_recovery(t1d, M0, T1_true)
    M2 = saturation_recovery(t2d, M0, T1_true)
    T1_fast = fast_t1_two_point(t1d, M1, t2d, M2, M0)
    print(f"  fast two-point T1      = {T1_fast:.3f} s")
    assert abs(T1_fast - T1_true) < 0.05

    # The fast method (2 delays) is much quicker than a full 20-point scan
    t_full = acquisition_time(t, T1_guess=T1_true)
    t_fast = acquisition_time([t1d, t2d], T1_guess=T1_true)
    print(f"  acq time full / fast   = {t_full:.1f} / {t_fast:.1f} s  ({t_full/t_fast:.0f}x)")
    assert t_fast < t_full
    print("  PASS")
    return {"T1_fit": float(T1_fit), "T1_fast": float(T1_fast),
            "speedup": float(t_full / t_fast)}


if __name__ == "__main__":
    test_all()
