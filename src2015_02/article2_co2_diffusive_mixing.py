"""
Article 2: CO2 EOR by Diffusive Mixing in Fractured Reservoirs
Eide, Ersland, Brattekas, Haugen, Graue, Ferno (2015)
Reference: Petrophysics Vol. 56, No. 1 (February 2015), pp. 23-31
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best Papers of the 2014 SCA Symposium.  CO2 injected into fractured cores
recovers oil from the matrix by diffusive mixing.  When diffusion controls
production, the matrix saturation change is linear in the square root of time
(Fick's law); the early-time fractional recovery from a matrix block follows the
classic short-time solution, from which an effective matrix diffusion
coefficient can be extracted.

Implements:

  - Diffusion length  L = sqrt(D*t)
  - Square-root-of-time recovery (diffusion-controlled) and its slope fit
  - Fickian early-time fractional recovery from a matrix block
  - Effective diffusion coefficient from the sqrt-time recovery slope

Note: this is an experimental MRI/CT paper; the relations below are the standard
Fickian-diffusion forms it relies on.  SI units: D in m^2/s, length in m, time
in s; recovery as a fraction.
"""

import numpy as np


# ---------------------------------------------- diffusion --------------

def diffusion_length(diffusion_coeff, time):
    """Characteristic diffusion length  L = sqrt(D*t)."""
    return np.sqrt(diffusion_coeff * np.asarray(time, float))


def sqrt_time_recovery(time, rate, rf_max=1.0):
    """Diffusion-controlled recovery linear in sqrt(time)

        RF(t) = min(rate*sqrt(t), rf_max),

    the signature of a Fickian (diffusion-limited) production mechanism.
    """
    rf = rate * np.sqrt(np.asarray(time, float))
    return np.minimum(rf, rf_max)


def fit_sqrt_time_slope(times, recovery):
    """Slope of recovery vs. sqrt(time) (the diffusion-controlled rate).

    A good linear fit of RF against sqrt(t) confirms diffusion as the production
    mechanism.  Returns the slope.
    """
    x = np.sqrt(np.asarray(times, float))
    slope, _ = np.polyfit(x, np.asarray(recovery, float), 1)
    return slope


# ---------------------------------------------- Fickian block recovery --------------

def fick_early_time_recovery(diffusion_coeff, time, half_length):
    """Early-time fractional recovery from a matrix block (1D diffusion)

        Mt/Minf = (2/L)*sqrt(D*t/pi),

    valid until Mt/Minf ~ 0.6; L is the block half-length.  Linear in sqrt(t).
    """
    return (2.0 / half_length) * np.sqrt(diffusion_coeff * np.asarray(time, float) / np.pi)


def diffusion_coefficient_from_slope(slope, half_length):
    """Effective diffusion coefficient from the early-time recovery slope

        D = (slope*L*sqrt(pi)/2)^2,

    inverting the early-time Fickian recovery (slope = recovery per sqrt(t)).
    """
    return (slope * half_length * np.sqrt(np.pi) / 2.0) ** 2


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: CO2 EOR by Diffusive Mixing")
    print("=" * 60)

    # Diffusion length grows as sqrt(D*t)
    assert np.isclose(diffusion_length(1e-9, 1e6), np.sqrt(1e-3))

    # Recovery is linear in sqrt(t) and capped at rf_max
    t = np.array([1e3, 4e3, 9e3, 1.6e4])
    rf = sqrt_time_recovery(t, rate=2e-3, rf_max=0.6)
    print(f"  RF(sqrt-t) @t          = {np.round(rf, 3)}")
    assert np.all(np.diff(rf) >= 0) and rf[-1] <= 0.6
    # The slope fit recovers the diffusion rate (in the unsaturated regime)
    slope = fit_sqrt_time_slope(t[:3], rf[:3])
    assert np.isclose(slope, 2e-3, rtol=1e-6)

    # Early-time block recovery is linear in sqrt(t) and inverse in block size
    r_small = fick_early_time_recovery(1e-9, 1e5, half_length=0.01)
    r_big = fick_early_time_recovery(1e-9, 1e5, half_length=0.05)
    print(f"  block recovery 1/5 cm  = {r_small:.3f} / {r_big:.3f}")
    assert r_small > r_big > 0

    # Diffusion coefficient round-trips through the recovery slope
    d_true, half = 2e-9, 0.02
    s = (2.0 / half) * np.sqrt(d_true / np.pi)        # slope of RF vs sqrt(t)
    assert np.isclose(diffusion_coefficient_from_slope(s, half), d_true)
    print("  PASS")
    return {"slope": float(slope), "r_small": float(r_small)}


if __name__ == "__main__":
    test_all()
