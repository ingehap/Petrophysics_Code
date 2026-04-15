"""
article9_xray_invasion.py
==========================
Implementation of ideas from:

    Aerens, P., Torres-Verdin, C., Espinoza, N.
    "Experimental Time-Lapse Visualization of Mud-Filtrate Invasion
    and Mudcake Deposition Using X-Ray Radiography"
    Petrophysics, Vol. 64, No. 3 (June 2023), pp. 448-461
    DOI: 10.30632/PJV64N3-2023a9

The paper monitors mud-filtrate invasion in a thin rock slab using
2D X-ray radiography.  The processing chain is:

  1. Beer-Lambert attenuation         I = I0 * exp(-mu_eff * h)
  2. Baseline subtraction              dI = I_dry - I(t)
  3. 2D median filtering (3x3)         to reduce salt-and-pepper noise
  4. Time-lapse interpretation         tracking the invasion front and
                                       mudcake thickness vs time

Mud-filtrate invasion under a constant overbalance pressure is governed
by a leak-off / mudcake-build-up model.  When the mudcake controls the
flux (after a short transient), the cumulative filtrate volume per unit
area follows a square-root-of-time law:

    Vcake(t) ~ sqrt(2 * k_cake * dP * (1 - phi_cake) / (mu * c) * t)

We use a simplified Darcy-flow front-advance model

    x_front(t) = sqrt( 2 * k * dP / (mu * phi) * t )

to predict the position of the invasion front into the rock.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Beer-Lambert
# ---------------------------------------------------------------------------
def beer_lambert(I0: np.ndarray, mu: float | np.ndarray,
                 h: float = 1.0) -> np.ndarray:
    """I = I0 * exp(-mu * h)."""
    return I0 * np.exp(-np.asarray(mu) * h)


def attenuation_map(I_now: np.ndarray, I_dry: np.ndarray) -> np.ndarray:
    """A = -ln(I_now / I_dry)  (positive when fluid invades dry sample)."""
    safe = np.where(I_dry == 0, 1e-30, I_dry)
    return -np.log(np.clip(I_now / safe, 1e-30, None))


# ---------------------------------------------------------------------------
# 2D median filter (3x3) - the paper applies this to all radiographs
# ---------------------------------------------------------------------------
def median_filter_3x3(image: np.ndarray) -> np.ndarray:
    """Pure-numpy 3x3 median filter (with edge replication)."""
    pad = np.pad(image, 1, mode="edge")
    H, W = image.shape
    stack = np.empty((9, H, W))
    k = 0
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            stack[k] = pad[dy:dy + H, dx:dx + W]
            k += 1
    return np.median(stack, axis=0)


# ---------------------------------------------------------------------------
# Darcy invasion-front position
# ---------------------------------------------------------------------------
def invasion_front_position(t: np.ndarray, k: float, dP: float,
                            mu: float, phi: float) -> np.ndarray:
    """
    x(t) = sqrt(2 * k * dP / (mu * phi) * t).

    k   permeability [m^2]
    dP  overbalance pressure [Pa]
    mu  filtrate viscosity [Pa s]
    phi rock porosity (fraction)
    t   time [s]
    """
    coef = 2.0 * k * dP / (mu * phi)
    return np.sqrt(coef * np.asarray(t))


# ---------------------------------------------------------------------------
# Build a time-lapse synthetic radiograph stack
# ---------------------------------------------------------------------------
def synthetic_radiograph_series(times: np.ndarray, slab_shape: tuple,
                                pixel_size: float,
                                k: float, dP: float, mu: float, phi: float,
                                mudcake_growth: float = 1e-5,
                                noise: float = 0.005,
                                rng: np.random.Generator | None = None
                                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a sequence of synthetic 2D radiographs.

    Returns
    -------
    I_dry  baseline radiograph
    stack  array of shape (n_times, H, W) with one radiograph per time
    """
    if rng is None:
        rng = np.random.default_rng(0)

    H, W = slab_shape
    # Smooth dry baseline with a few percent variability
    I_dry = 1.0 + 0.02 * rng.standard_normal((H, W))

    x_front = invasion_front_position(times, k, dP, mu, phi)        # (n,)
    cake = mudcake_growth * np.sqrt(times)                          # (n,)

    # Pixel x-coordinate of the 'front' at each time
    front_pix = (x_front / pixel_size).astype(int)
    cake_pix = np.clip((cake / pixel_size).astype(int), 0, W - 1)

    stack = np.empty((times.size, H, W))
    for i, _ in enumerate(times):
        mu_map = np.zeros((H, W))
        # mudcake (very strong absorber)
        if cake_pix[i] > 0:
            mu_map[:, :cake_pix[i]] += 1.5
        # invaded zone
        end = min(W, cake_pix[i] + max(front_pix[i], 0))
        mu_map[:, cake_pix[i]:end] += 0.4
        I = beer_lambert(I_dry, mu_map)
        I = I + noise * rng.standard_normal(I.shape)
        stack[i] = np.clip(I, 1e-3, None)
    return I_dry, stack


# ---------------------------------------------------------------------------
# Pick the invasion front from a processed radiograph
# ---------------------------------------------------------------------------
def detect_front(att_map: np.ndarray, cake_thresh: float = 1.0,
                 inv_thresh: float = 0.2) -> dict:
    """
    Locate the cake / invaded-zone boundary on a single 2D attenuation
    map by thresholding the column-averaged profile.
    """
    profile = att_map.mean(axis=0)
    cake_idx = np.where(profile >= cake_thresh)[0]
    cake_end = int(cake_idx.max()) if cake_idx.size else 0
    inv_idx = np.where(profile[cake_end:] >= inv_thresh)[0]
    inv_end = int(inv_idx.max() + cake_end) if inv_idx.size else cake_end
    return {"cake_end": cake_end, "front_end": inv_end,
            "front_depth": inv_end - cake_end}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    """Synthetic-data test for module 9 (X-ray invasion)."""
    print("[article9] testing Beer-Lambert and attenuation map ...")
    I_dry = np.full((10, 10), 2.0)
    I_now = beer_lambert(I_dry, mu=0.5, h=1.0)
    A = attenuation_map(I_now, I_dry)
    assert np.allclose(A, 0.5, atol=1e-12)

    print("[article9] testing 3x3 median filter on a salt-and-pepper image")
    img = np.zeros((6, 6))
    img[2, 2] = 100.0    # spike
    out = median_filter_3x3(img)
    assert out[2, 2] == 0.0, "spike should be removed"

    print("[article9] running synthetic time-lapse invasion experiment ...")
    times = np.array([60, 120, 240, 480, 960], dtype=float)   # seconds
    # Tight rock so the front stays well inside the 6-mm slab over the
    # full time window.  Predicted front depth at t=960s is ~ 2.5 mm.
    k = 2e-18           # ~ 0.002 mD
    dP = 100 * 6894.76  # 100 psi -> Pa
    mu = 1e-3           # water-like, Pa s
    phi = 0.20
    px = 30e-6          # 30 micron / pixel
    H, W = 40, 200      # 40 rows, 200 columns -> 6 mm slab depth

    I_dry, stack = synthetic_radiograph_series(times, (H, W), px,
                                               k, dP, mu, phi,
                                               mudcake_growth=2e-6)
    # Apply baseline subtraction and median filtering
    fronts = []
    for i, t in enumerate(times):
        A = attenuation_map(stack[i], I_dry)
        A = median_filter_3x3(A)
        info = detect_front(A, cake_thresh=1.0, inv_thresh=0.15)
        fronts.append(info["front_depth"] * px)
    fronts = np.array(fronts)

    # Predicted depth from the analytic Darcy law
    pred = invasion_front_position(times, k, dP, mu, phi)
    print(f"           t (s)        observed depth (mm)   predicted (mm)")
    for t, o, p in zip(times, fronts, pred):
        print(f"           {t:6.0f}        {o*1e3:>14.3f}      {p*1e3:>10.3f}")

    # Observed front depth must be monotonically non-decreasing (allow
    # one-pixel jitter from the threshold detection).
    assert np.all(np.diff(fronts) >= -px), "front depth must be non-decreasing"
    # Last observed depth must be of the same order of magnitude as the
    # analytic prediction.
    rel = np.abs(fronts[-1] - pred[-1]) / pred[-1]
    assert rel < 0.7, f"observed/predicted disagreement too large ({rel:.2f})"
    print("[article9] OK")


if __name__ == "__main__":
    test_all()
