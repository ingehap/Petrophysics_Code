"""
Article 3: Refining Interpretation Models of Multiphase Flow for Existing and
           Next-Generation Production Logging Sensors
Manzar, Sun, Chace (2018)
DOI: 10.30632/PJV59V4-2018a2

Array production-logging tools (array resistance/capacitance/spinner) sample
local water holdup and velocity across the pipe.  A simple linear mixing law
maps the array-resistance response to a "linear" water holdup, but the true
holdup departs from it; the paper adds a nonlinear, velocity-dependent
correction calibrated in a multiphase flow loop, then integrates the corrected
local holdups and velocities over the pipe cross-section to allocate per-phase
flow rates.

Implements:

  - Linear water holdup from array resistance  Yw = (R - Rhc)/(Rw - Rhc)
  - Nonlinear, velocity-dependent holdup correction (piecewise, Eqs. 6-14)
  - Cross-section integration of holdup  Yp = sum(Yp_i*a_i)/sum(a_i)
  - Per-phase flow-rate allocation  qp = sum(Yp_i*V_i*a_i)

Note: this issue's PDF has a text layer, and the mid-range correction (Eq. 7)
survived verbatim; the low-holdup velocity curves (Eqs. 9-13) lost their printed
coefficients in extraction and are reconstructed as monotone calibration curves
between the two reported flow-loop velocities (0.79 and 1.92 m/s).  SI units.
"""

import numpy as np


# ---------------------------------------------- local holdup --------------

def linear_water_holdup(r, r_hc, r_w):
    """Linear water holdup from an array-resistance response (Eq. 1)

        Yw = (R - Rhc)/(Rw - Rhc),

    clipped to [0, 1].  Rhc/Rw are the responses in pure hydrocarbon / water.
    """
    yw = (np.asarray(r, float) - r_hc) / (r_w - r_hc)
    return np.clip(yw, 0.0, 1.0)


def _low_holdup_curve(yw_lin, slope):
    """Reconstructed monotone calibration curve at one flow-loop velocity."""
    return np.clip(slope * yw_lin, 0.0, 1.0)


def nonlinear_holdup(yw_lin, velocity):
    """Velocity-dependent nonlinear holdup correction (Eqs. 6-14).

      - Yw_lin >= 0.96            : no correction (Eq. 6).
      - 0.65 < Yw_lin < 0.96      : Yw = 1.26*Yw_lin^2 - 0.31*Yw_lin + 0.05 (Eq. 7).
      - Yw_lin <= 0.65            : interpolate the two flow-loop curves by the
                                    local mixture velocity (Eqs. 8-14): the
                                    0.79 m/s curve for V<=0.79, the 1.92 m/s
                                    curve for V>1.92, linear interpolation between.
    """
    y = float(yw_lin)
    if y >= 0.96:
        return y
    if y > 0.65:
        return 1.26 * y ** 2 - 0.31 * y + 0.05
    yw1 = _low_holdup_curve(y, slope=0.85)            # 0.79 m/s (more slip)
    yw2 = _low_holdup_curve(y, slope=1.05)            # 1.92 m/s (well mixed)
    if velocity <= 0.79:
        return yw1
    if velocity > 1.92:
        return yw2
    frac = (velocity - 0.79) / (1.92 - 0.79)
    return yw1 + frac * (yw2 - yw1)


# ---------------------------------------------- profile reconstruction --------------

def average_holdup(holdups, areas):
    """Cross-section average holdup  Yp = sum(Yp_i*a_i)/sum(a_i)  (Eq. 15)."""
    h = np.asarray(holdups, float)
    a = np.asarray(areas, float)
    return float((h * a).sum() / a.sum())


def phase_flow_rate(holdups, velocities, areas):
    """Phase flow rate  qp = sum(Yp_i*V_i*a_i)  (Eq. 16), summing layered flux."""
    h = np.asarray(holdups, float)
    v = np.asarray(velocities, float)
    a = np.asarray(areas, float)
    return float((h * v * a).sum())


def water_cut(q_water, q_oil):
    """Water cut  WC = q_w/(q_w + q_o)."""
    return q_water / (q_water + q_oil)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Multiphase PL Holdup Correction")
    print("=" * 60)

    # Linear holdup interpolates the pure-fluid responses and clips
    assert np.isclose(linear_water_holdup(5.0, r_hc=10.0, r_w=2.0), 0.625)
    assert linear_water_holdup(11.0, 10.0, 2.0) == 0.0   # below hydrocarbon end

    # Eq. 7 (verbatim) governs the mid-range; high holdup is uncorrected
    assert nonlinear_holdup(0.98, 1.0) == 0.98
    mid = nonlinear_holdup(0.80, 1.0)
    print(f"  Eq.7 mid-range Yw      = {mid:.3f}")
    assert np.isclose(mid, 1.26 * 0.8 ** 2 - 0.31 * 0.8 + 0.05)

    # Low-holdup correction increases with mixture velocity (less slip)
    slow = nonlinear_holdup(0.50, 0.5)
    fast = nonlinear_holdup(0.50, 2.5)
    print(f"  low-holdup slow/fast   = {slow:.3f} / {fast:.3f}")
    assert fast > slow

    # Cross-section allocation: a uniform profile recovers its own holdup/rate
    areas = np.full(6, 1.0 / 6)
    holds = np.full(6, 0.4)
    vels = np.full(6, 1.5)
    assert np.isclose(average_holdup(holds, areas), 0.4)
    qw = phase_flow_rate(holds, vels, areas)
    qo = phase_flow_rate(1 - holds, vels, areas)
    print(f"  q_water / q_oil        = {qw:.3f} / {qo:.3f}")
    assert np.isclose(qw, 0.4 * 1.5) and np.isclose(water_cut(qw, qo), 0.4)
    print("  PASS")
    return {"mid_holdup": float(mid), "water_cut": float(water_cut(qw, qo))}


if __name__ == "__main__":
    test_all()
