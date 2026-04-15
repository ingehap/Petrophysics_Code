"""
article4_gas_trapping.py
=========================
Implementation of ideas from:

    Gao, Y., Sorop, T., Brussee, N., van der Linde, H., Coorn, A.,
    Appel, M., Berg, S.
    "Advanced Digital-SCAL Measurements of Gas Trapped in Sandstone"
    Petrophysics, Vol. 64, No. 3 (June 2023), pp. 368-383
    DOI: 10.30632/PJV64N3-2023a4

The paper uses high-resolution micro-CT to study the residual / trapped
gas saturation Sgr.  Two effects matter:

  1. Capillary trapping during water imbibition (Land trapping model)
        Sgr = Sgi / (1 + C * Sgi)
     with the Land coefficient  C = 1/Sgr_max - 1/Sgi_max .
  2. Time-dependent shrinkage of disconnected gas clusters by
     dissolution / Ostwald ripening, well described by an
     exponential decay of the trapped saturation:
        Sgr(t) = Sgr_inf + (Sgr_0 - Sgr_inf) * exp(-t/tau)

This module also includes a simple voxel-segmentation routine that
extracts gas / brine / grain volume fractions from a 3-D synthetic
micro-CT image (3-class Otsu-style thresholding).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Land trapping model (Land, 1968)
# ---------------------------------------------------------------------------
def land_coefficient(sgi_max: float, sgr_max: float) -> float:
    """C = 1/Sgr_max - 1/Sgi_max."""
    if sgr_max <= 0 or sgi_max <= 0:
        raise ValueError("saturations must be positive")
    return 1.0 / sgr_max - 1.0 / sgi_max


def land_trapped_sat(sgi: float | np.ndarray, C: float) -> float | np.ndarray:
    """Trapped saturation from initial gas saturation Sgi and Land C."""
    sgi = np.asarray(sgi, dtype=float)
    return sgi / (1.0 + C * sgi)


# ---------------------------------------------------------------------------
# Dissolution / ripening kinetics
# ---------------------------------------------------------------------------
def ripening_decay(t: np.ndarray, sgr_0: float,
                   sgr_inf: float, tau: float) -> np.ndarray:
    """
    Trapped gas saturation as a function of time when the brine is
    NOT pre-equilibrated (under-saturated brine continues to dissolve
    the trapped gas).

        Sgr(t) = Sgr_inf + (Sgr_0 - Sgr_inf) * exp(-t/tau)
    """
    return sgr_inf + (sgr_0 - sgr_inf) * np.exp(-np.asarray(t) / tau)


# ---------------------------------------------------------------------------
# Three-class segmentation of a synthetic micro-CT image
# ---------------------------------------------------------------------------
def segment_microct(image: np.ndarray) -> dict:
    """
    Very simple 3-class segmentation of a micro-CT volume into
    gas / brine / grain phases.  Two thresholds are computed by
    bisecting the image histogram (a coarse Otsu approximation).

    Returns a dict with the volume fraction of each phase.
    """
    flat = image.ravel()
    t1 = float(np.quantile(flat, 1 / 3))
    t2 = float(np.quantile(flat, 2 / 3))

    gas = flat < t1
    brine = (flat >= t1) & (flat < t2)
    grain = flat >= t2

    n = flat.size
    return {
        "gas_fraction": gas.sum() / n,
        "brine_fraction": brine.sum() / n,
        "grain_fraction": grain.sum() / n,
        "thresholds": (t1, t2),
    }


def gas_saturation(seg: dict) -> float:
    """Gas saturation = gas / (gas + brine)  (i.e., fraction of pore space)."""
    pore = seg["gas_fraction"] + seg["brine_fraction"]
    return 0.0 if pore == 0 else seg["gas_fraction"] / pore


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    """Synthetic-data test for module 4 (gas trapping)."""
    print("[article4] testing Land trapping model ...")
    C = land_coefficient(sgi_max=0.7, sgr_max=0.35)
    assert C > 0, "Land coefficient must be positive"

    sgi = np.array([0.2, 0.4, 0.6, 0.7])
    sgr = land_trapped_sat(sgi, C)
    # Sgr should be monotonically increasing with Sgi
    assert np.all(np.diff(sgr) > 0)
    # Sgr at Sgi_max should equal Sgr_max
    assert abs(sgr[-1] - 0.35) < 1e-6
    print(f"           Sgi={sgi}, Sgr={np.round(sgr, 3)}")

    print("[article4] testing dissolution / ripening kinetics ...")
    t = np.linspace(0, 100, 50)         # arbitrary time units
    s = ripening_decay(t, sgr_0=0.30, sgr_inf=0.05, tau=20.0)
    assert s[0] == 0.30
    assert abs(s[-1] - 0.05) < 0.01
    assert np.all(np.diff(s) <= 0)
    print(f"           t=0  Sgr={s[0]:.3f},  t->inf Sgr -> {s[-1]:.3f}")

    print("[article4] testing 3-class micro-CT segmentation ...")
    rng = np.random.default_rng(0)
    # synthetic 30^3 voxel volume:  20% gas / 20% brine / 60% grain
    vol = np.empty(30 ** 3)
    n = vol.size
    vol[: int(0.2 * n)] = rng.normal(20, 3, int(0.2 * n))   # gas (low CT#)
    vol[int(0.2 * n):int(0.4 * n)] = rng.normal(80, 3, int(0.2 * n))   # brine
    vol[int(0.4 * n):] = rng.normal(160, 5, n - int(0.4 * n))           # grain
    rng.shuffle(vol)
    vol = vol.reshape(30, 30, 30)

    seg = segment_microct(vol)
    sg = gas_saturation(seg)
    print(f"           segmented fractions: gas={seg['gas_fraction']:.2f}  "
          f"brine={seg['brine_fraction']:.2f}  "
          f"grain={seg['grain_fraction']:.2f}")
    print(f"           gas saturation Sg = {sg:.2f}")
    assert 0.4 < sg < 0.6, "expected Sg ~ 0.5 (20%% gas / 20%% brine)"
    print("[article4] OK")


if __name__ == "__main__":
    test_all()
