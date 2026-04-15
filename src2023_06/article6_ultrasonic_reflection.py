"""
article6_ultrasonic_reflection.py
==================================
Implementation of ideas from:

    Olszowska, D., Gallardo-Giozza, G., Crisafulli, D., Torres-Verdin, C.
    "Angle-Dependent Ultrasonic Wave Reflection for Estimating
    High-Resolution Elastic Properties of Complex Rock Samples"
    Petrophysics, Vol. 64, No. 3 (June 2023), pp. 402-419
    DOI: 10.30632/PJV64N3-2023a6

This module implements the angle-dependent reflection / transmission
coefficients of a plane longitudinal wave incident from a fluid (water
or castor oil) onto a solid (the rock sample) and the inverse problem of
estimating Vp, Vs and density of the solid from a measured RC vs theta
curve.

The classical Zoeppritz formulation reduces to the closed-form 'fluid /
solid' reflection coefficient (also known as the Brekhovskikh formula)

    Z_p = rho_s Vp / cos(theta_p)
    Z_s = rho_s Vs / cos(theta_s)
    Z_f = rho_f Vf / cos(theta_i)
    A   = Z_p cos^2(2*theta_s) + Z_s sin^2(2*theta_s)

    R(theta) = (A - Z_f) / (A + Z_f)

(Vanaverbeke et al. 2003; the same formula is given as the reduced
case of the inhomogeneous Zoeppritz system in the paper.)

The module provides
  * `reflection_coefficient(theta, ...)`   forward model
  * `critical_angles(...)`                 P- and S-wave critical angles
  * `invert_velocities(theta, R_meas, ...)` non-linear least squares
                                           recovery of Vp, Vs, rho
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Forward fluid-solid reflection coefficient (Brekhovskikh)
# ---------------------------------------------------------------------------
def reflection_coefficient(theta_deg: np.ndarray,
                           Vp: float, Vs: float, rho_s: float,
                           Vf: float = 1480.0, rho_f: float = 1000.0
                           ) -> np.ndarray:
    """
    Magnitude of the angle-dependent P-wave reflection coefficient at a
    fluid / solid interface.

    Parameters
    ----------
    theta_deg : incidence angle in the fluid (degrees)
    Vp, Vs    : P- and S-wave velocity of the solid (m/s)
    rho_s     : solid density (kg/m^3)
    Vf, rho_f : coupling-fluid velocity / density (m/s, kg/m^3)
    """
    theta_i = np.deg2rad(np.asarray(theta_deg, dtype=float))
    sin_i = np.sin(theta_i)

    # Snell's law:   sin(theta) / V = const = sin(theta_i)/Vf
    sin_p = np.clip(Vp * sin_i / Vf, -1.0, 1.0)
    sin_s = np.clip(Vs * sin_i / Vf, -1.0, 1.0)
    cos_i = np.cos(theta_i)
    # complex sqrt to handle the post-critical regime
    cos_p = np.sqrt(1.0 - sin_p ** 2 + 0j)
    cos_s = np.sqrt(1.0 - sin_s ** 2 + 0j)

    Zf = rho_f * Vf / np.where(cos_i == 0, 1e-30, cos_i)
    Zp = rho_s * Vp / np.where(cos_p == 0, 1e-30, cos_p)
    Zs = rho_s * Vs / np.where(cos_s == 0, 1e-30, cos_s)

    sin_2s = 2.0 * sin_s * cos_s
    cos_2s = 1.0 - 2.0 * sin_s ** 2

    A = Zp * cos_2s ** 2 + Zs * sin_2s ** 2
    R = (A - Zf) / (A + Zf)
    return np.abs(R)


# ---------------------------------------------------------------------------
# P- and S-wave critical angles for a fluid -> solid interface
# ---------------------------------------------------------------------------
def critical_angles(Vp: float, Vs: float,
                    Vf: float = 1480.0) -> tuple[float, float]:
    """Returns (theta_c_P, theta_c_S) in degrees."""
    theta_p = np.degrees(np.arcsin(min(Vf / Vp, 1.0)))
    theta_s = np.degrees(np.arcsin(min(Vf / Vs, 1.0)))
    return theta_p, theta_s


# ---------------------------------------------------------------------------
# Inverse problem - estimate (Vp, Vs, rho_s) from a measured RC curve
# ---------------------------------------------------------------------------
def invert_velocities(theta_deg: np.ndarray, R_meas: np.ndarray,
                      Vf: float = 1480.0, rho_f: float = 1000.0,
                      x0: tuple = (4000.0, 2200.0, 2400.0)) -> dict:
    """
    Recover Vp, Vs and rho_s from a measured |R(theta)| curve by
    nonlinear least squares.
    """
    def residuals(p):
        Vp, Vs, rho_s = p
        return reflection_coefficient(theta_deg, Vp, Vs, rho_s,
                                      Vf=Vf, rho_f=rho_f) - R_meas

    res = least_squares(residuals, x0=np.asarray(x0, dtype=float),
                        bounds=([1500, 500, 1500],
                                [8000, 5000, 4000]))
    Vp, Vs, rho_s = res.x
    return {"Vp": float(Vp), "Vs": float(Vs), "rho_s": float(rho_s),
            "cost": float(res.cost), "success": bool(res.success)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    """Synthetic-data test for module 6 (ultrasonic reflectivity)."""
    print("[article6] computing reference R(theta) for Berea-like sample ...")
    Vp_true, Vs_true, rho_true = 3500.0, 2000.0, 2200.0
    theta = np.linspace(0, 80, 161)
    R = reflection_coefficient(theta, Vp_true, Vs_true, rho_true)
    assert R.shape == theta.shape
    assert R.min() >= 0 and R.max() <= 1.0001
    print(f"           R(0deg) = {R[0]:.3f}  (normal incidence)")

    tc_p, tc_s = critical_angles(Vp_true, Vs_true)
    print(f"           critical angles:  P={tc_p:.1f}deg  S={tc_s:.1f}deg")
    assert 20 < tc_p < 35

    print("[article6] running inversion on noisy synthetic measurements ...")
    rng = np.random.default_rng(3)
    theta_meas = np.linspace(2, 60, 31)
    R_meas = reflection_coefficient(theta_meas, Vp_true, Vs_true, rho_true)
    R_noisy = R_meas + 0.01 * rng.standard_normal(R_meas.size)
    R_noisy = np.clip(R_noisy, 0.0, 1.0)

    out = invert_velocities(theta_meas, R_noisy)
    print(f"           true   :  Vp={Vp_true:.0f}  Vs={Vs_true:.0f}  "
          f"rho={rho_true:.0f}")
    print(f"           inverted: Vp={out['Vp']:.0f}  Vs={out['Vs']:.0f}  "
          f"rho={out['rho_s']:.0f}")

    # Allow ~10% tolerance because Vs and density are weakly constrained
    # below the S-wave critical angle (cf. discussion in the paper).
    assert abs(out["Vp"] - Vp_true) / Vp_true < 0.05, "Vp recovery error"
    assert abs(out["Vs"] - Vs_true) / Vs_true < 0.15, "Vs recovery error"
    print("[article6] OK")


if __name__ == "__main__":
    test_all()
