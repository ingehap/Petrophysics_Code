"""
Article 1: Automatically Quantifying Wireline and LWD Pressure-Test Quality
Proett, Musharfi, Gill, Ma, Meridji, Eyuboglu (2015)
Reference: Petrophysics Vol. 56, No. 2 (April 2015), pp. 101-115
DOI: none assigned (this issue predates SPWLA DOI assignment)

A spreadsheet workflow scores formation-tester (wireline/LWD) pretest quality.
The drawdown mobility comes from the pseudosteady hemispherical-flow model
(probe flow coefficient, flow rate, drawdown differential); pressure and
temperature stability are the slope of a least-squares fit to the last seconds
of buildup; a relative radius of investigation follows from the mobility and
drawdown duration.

Implements:

  - Drawdown mobility  M = Cpf*q/dP  (Moran et al., 1961; Eq. 1)
  - Pretest flow rate  q = V/tp
  - Linear least-squares regression slope/intercept and residual std (Eqs. 2-5)
  - Pressure (or temperature) stability from the buildup slope
  - Relative radius of investigation (Eqs. 14-15)

Note: this issue's PDF has a text layer; the drawdown-mobility, LSR and ROI
relations are transcribed from the body, while the typeset glyphs were dropped
and reconstructed in standard form.  Mobility in mD/cP, pressure in psi, volume
in cc, time in s, rate in cc/s.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- mobility --------------

def pretest_flow_rate(volume, drawdown_time):
    """Pretest flow rate  q = V/tp  (pretest volume over drawdown time)."""
    return volume / drawdown_time


def drawdown_mobility(cpf, flow_rate, dp_drawdown):
    """Pseudosteady hemispherical-flow drawdown mobility (Moran et al., 1961; Eq. 1)

        M_sdd = Cpf * q / dP_dd,

    with Cpf the probe flow coefficient, q the drawdown flow rate and dP_dd the
    drawdown differential (Pstop - Pdd).  Valid roughly over 1-1000 mD/cP.
    """
    return cpf * flow_rate / dp_drawdown


# ---------------------------------------------- stability (LSR) --------------

def linear_regression(x, y):
    """Least-squares slope and intercept of  y = a + b*x  (Eqs. 2-4).  Returns
    (slope, intercept)."""
    lf = petrolib.inversion_numerics.fitting.fit_line(x, y)
    return lf.slope, lf.intercept


def regression_residual_std(x, y):
    """Standard deviation of the LSR residuals (Eq. 5), a QC noise measure."""
    b, a = linear_regression(x, y)
    resid = np.asarray(y, float) - (a + b * np.asarray(x, float))
    return float(np.std(resid, ddof=2)) if len(resid) > 2 else float(np.std(resid))


def pressure_stability(times, pressures):
    """Pressure (or temperature) stability = slope of the LSR over the buildup
    window (ideally near zero at a stable buildup)."""
    slope, _ = linear_regression(times, pressures)
    return slope


# ---------------------------------------------- radius of investigation --------------

def radius_of_investigation(mobility, drawdown_time, k_const=1.0):
    """Relative radius of investigation (Eqs. 14-15)

        ROI ~ k_const*sqrt(M_sdd * t_dd),

    increasing with the drawdown mobility and drawdown duration; a relative,
    test-parameter-dependent measure rather than an absolute depth.
    """
    return k_const * np.sqrt(mobility * np.asarray(drawdown_time, float))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Pressure-Test Quality")
    print("=" * 60)

    # Drawdown mobility from probe coefficient, flow rate and differential
    q = pretest_flow_rate(volume=10.0, drawdown_time=20.0)
    m = drawdown_mobility(cpf=5500.0, flow_rate=q, dp_drawdown=200.0)
    print(f"  flow rate / mobility   = {q:.3f} cc/s / {m:.1f} mD/cP")
    assert np.isclose(q, 0.5) and m > 0
    # Mobility falls with a larger drawdown differential (tighter rock)
    assert drawdown_mobility(5500.0, q, 800.0) < m

    # LSR recovers a known slope/intercept; residual std ~ injected noise
    t = np.linspace(0, 60, 60)
    rng = np.random.default_rng(0)
    p = 5000.0 + 0.002 * t + rng.normal(0, 0.5, t.size)
    slope, intercept = linear_regression(t, p)
    print(f"  buildup slope          = {slope:.4f} psi/s")
    assert np.isclose(slope, 0.002, atol=5e-3) and np.isclose(intercept, 5000.0, atol=0.5)
    assert regression_residual_std(t, p) > 0

    # A near-zero stability slope indicates a well-stabilized buildup
    stable = pressure_stability(t, np.full_like(t, 5000.0))
    assert np.isclose(stable, 0.0, atol=1e-9)

    # Radius of investigation grows with mobility and drawdown time
    roi = radius_of_investigation(m, drawdown_time=20.0)
    print(f"  relative ROI           = {roi:.2f}")
    assert roi > 0 and radius_of_investigation(m, 40.0) > roi
    print("  PASS")
    return {"mobility": float(m), "slope": float(slope), "ROI": float(roi)}


if __name__ == "__main__":
    test_all()
