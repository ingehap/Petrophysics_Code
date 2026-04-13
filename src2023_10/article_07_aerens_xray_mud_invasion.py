"""
article_07_aerens_xray_mud_invasion.py
======================================
Implementation of ideas from:

    Aerens, P., Espinoza, D. N., and Torres-Verdin, C. (2023).
    "High-Resolution Time-Lapse Monitoring of Mud Invasion in Spatially
    Complex Rocks Using In-Situ X-Ray Radiography."  Petrophysics,
    64(5), 715-740.  DOI: 10.30632/PJV64N5-2023a7

Mud filtrate invades the formation while drilling, leaving a
mud-cake on the borehole wall and altering near-borehole saturation.
The authors use micro-focus X-ray radiography to image saturation
fronts vs time at 10-30 micron resolution, with rectangular core
samples saturated with different connate fluids.

Concepts implemented in this module:

    * Beer-Lambert attenuation:   I = I0 * exp(-mu_eff * x)
    * Conversion of grayscale to volumetric saturation by linear
      interpolation between the dry and fully-water-saturated images
    * 1-D Buckley-Leverett radial invasion (Leverett 1941, fractional
      flow with capillary correction) producing time-evolving
      saturation profiles
    * External mud-cake build-up vs invaded volume (Outmans 1963 /
      Dewan & Chenevert 2001 model)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# X-ray attenuation -> saturation
# ---------------------------------------------------------------------------
def attenuation(I0: float, mu_eff: np.ndarray, thickness: float) -> np.ndarray:
    """Beer-Lambert: I = I0 * exp(-mu_eff * x)."""
    return I0 * np.exp(-mu_eff * thickness)


def grayscale_to_saturation(gray: np.ndarray,
                            gray_dry: np.ndarray,
                            gray_wet: np.ndarray) -> np.ndarray:
    """Linear conversion of pixel grayscale to water saturation.

    A value of 0 -> dry, 1 -> fully water-saturated.  Both endpoint
    images must be acquired on the same sample under identical
    geometry/exposure so that all attenuation differences are
    fluid-related.
    """
    den = gray_dry - gray_wet
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    return np.clip((gray_dry - gray) / den, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Mud-cake growth (external)
# ---------------------------------------------------------------------------
@dataclass
class MudcakeParams:
    fs: float = 0.05            # mud solid fraction
    fc: float = 0.6             # cake solid fraction
    k_mc: float = 1e-19         # mud cake permeability (m^2)
    mu_filt_Pas: float = 1e-3
    dP_Pa: float = 7e5


def mudcake_thickness(t_s: np.ndarray, p: MudcakeParams) -> np.ndarray:
    """External mudcake thickness vs time, dh/dt = (k_mc * dP) /
    (mu * h * (fc/fs - 1)).  Closed-form: h(t) = sqrt(2*k_mc*dP*t /
    (mu*(fc/fs - 1)))."""
    factor = 2.0 * p.k_mc * p.dP_Pa / (p.mu_filt_Pas * (p.fc / p.fs - 1.0))
    return np.sqrt(np.maximum(factor * t_s, 0.0))


def filtrate_volume(t_s: np.ndarray, p: MudcakeParams,
                    area_m2: float) -> np.ndarray:
    """Cumulative filtrate volume per unit area of borehole wall."""
    return mudcake_thickness(t_s, p) * area_m2 * (p.fc / p.fs - 1.0)


# ---------------------------------------------------------------------------
# Buckley-Leverett 1-D filtrate invasion
# ---------------------------------------------------------------------------
def fractional_flow(Sw: np.ndarray,
                    mu_w: float = 1e-3, mu_n: float = 5e-3,
                    Swir: float = 0.15, Snwr: float = 0.10,
                    nw: float = 2.5, nnw: float = 2.0) -> np.ndarray:
    Swe = np.clip((Sw - Swir) / (1.0 - Swir - Snwr), 0.0, 1.0)
    krw = Swe ** nw
    krn = (1.0 - Swe) ** nnw
    return (krw / mu_w) / (krw / mu_w + krn / mu_n + 1e-18)


def saturation_profile(x_m: np.ndarray, t_s: float,
                       qt: float, phi: float = 0.20,
                       Sw_init: float = 0.20,
                       mu_w: float = 1e-3, mu_n: float = 5e-3) -> np.ndarray:
    """Buckley-Leverett saturation profile by Welge tangent construction.

    For each x, solve x/t = (qt/A/phi) * df_w/dSw  for Sw using bisection
    on the rarefaction part of the fractional-flow curve.
    """
    Sw_grid = np.linspace(0.16, 0.89, 200)
    fw = fractional_flow(Sw_grid, mu_w=mu_w, mu_n=mu_n)
    dfdsw = np.gradient(fw, Sw_grid)
    # Identify shock by Welge tangent: dfw/dSw = (fw - fw_init) / (Sw - Sw_init)
    fw_i = fractional_flow(np.array([Sw_init]), mu_w=mu_w, mu_n=mu_n)[0]
    chord = (fw - fw_i) / (Sw_grid - Sw_init + 1e-12)
    i_shock = int(np.argmin(np.abs(dfdsw - chord)))
    Sw_shock = Sw_grid[i_shock]
    df_shock = dfdsw[i_shock]
    out = np.full_like(x_m, Sw_init, dtype=float)
    if t_s <= 0:
        return out
    # rarefaction zone
    speed = (qt / phi) * dfdsw[i_shock:]
    x_front = speed * t_s
    # behind the shock, interpolate Sw vs x
    out_rare = np.interp(x_m, x_front[::-1], Sw_grid[i_shock:][::-1],
                         left=Sw_grid[-1], right=Sw_init)
    x_shock_pos = (qt / phi) * df_shock * t_s
    out[x_m <= x_shock_pos] = out_rare[x_m <= x_shock_pos]
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    rng = np.random.default_rng(4)

    # Beer-Lambert
    I0 = 50000.0
    mu = np.array([0.5, 1.0, 2.0])
    I = attenuation(I0, mu, thickness=1.0)
    assert I[0] > I[1] > I[2]
    assert np.all(I < I0)

    # Grayscale -> Sw round-trip
    gray_dry = np.full(100, 200.0)
    gray_wet = np.full(100, 60.0)
    Sw_true = np.linspace(0, 1, 100)
    gray_obs = gray_dry - Sw_true * (gray_dry - gray_wet)
    Sw_est = grayscale_to_saturation(gray_obs, gray_dry, gray_wet)
    assert np.allclose(Sw_est, Sw_true, atol=1e-9)

    # Mudcake build-up monotonically increasing as sqrt(t)
    t = np.linspace(0, 3600, 200)
    p = MudcakeParams()
    h = mudcake_thickness(t, p)
    assert h[0] == 0.0
    assert np.all(np.diff(h) >= -1e-15)
    # sqrt-time scaling: h(4t) ~ 2 h(t)
    h1 = mudcake_thickness(np.array([900.0]), p)[0]
    h4 = mudcake_thickness(np.array([3600.0]), p)[0]
    assert abs(h4 / h1 - 2.0) < 1e-6

    # Buckley-Leverett: front advances with time
    x = np.linspace(0, 0.05, 200)
    s_short = saturation_profile(x, 30.0, qt=1e-6)
    s_long = saturation_profile(x, 600.0, qt=1e-6)
    front_short = x[np.where(s_short > 0.25)[0][-1]] if np.any(s_short > 0.25) else 0
    front_long = x[np.where(s_long > 0.25)[0][-1]] if np.any(s_long > 0.25) else 0
    assert front_long > front_short, (front_short, front_long)

    # Saturation profile bounded in [Swir, 1-Snwr]
    assert s_long.min() >= 0.16 - 1e-6 and s_long.max() <= 0.91

    print(f"article_07_aerens_xray_mud_invasion: OK  "
          f"(mudcake@1h={h[-1]*1000:.2f} mm, BL front t=600s={front_long*1000:.1f} mm)")


if __name__ == "__main__":
    test_all()
