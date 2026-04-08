"""
article2_claystone_repository.py
================================

Implementation of the main petrophysical techniques from:

    Strobel, J. (2024). "Petrophysical Analyses for Supporting the Search
    for a Claystone-Hosted Nuclear Repository."  Petrophysics 65(3),
    302-316. DOI: 10.30632/PJV65N3-2024a2

The paper describes how the German BGE searches for a high-level nuclear
waste repository in claystone using legacy oilfield logs.  The three
quantitative techniques that we implement are:

1.  Vertical variogram analysis of the gamma-ray log to identify layer
    thicknesses (Fig. 4 of the paper).  The variogram at lag h is

        gamma(h) = 0.5 * E[ (x(z) - x(z+h))^2 ]

    and the Lag1-specific enhanced variances are used with a P10
    threshold to detect layer boundaries.

2.  The "residual" curve technique: a raw GR log is first passed through
    a short (5-point) median filter and then a long (5-8 m) median
    filter; the absolute difference is the residual used to quantify GR
    deflection severity.

3.  An Archie-type effective diffusivity model for clay porosity and
    tortuosity,

        De = D0 * phi^m

    (van Loon and Mibus, 2015) and the associated Waxman-Smits-style
    modified cementation factor m* extracted from a resistivity log.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------
# 1. Vertical variogram of a well log ---------------------------------------
# ---------------------------------------------------------------------------

def vertical_variogram(log: np.ndarray, max_lag: int) -> np.ndarray:
    """Classical experimental variogram gamma(h) for h = 1 .. max_lag samples.

    Returns an array of length `max_lag` where entry h-1 is

        gamma(h) = (1/(2*N)) * sum_i (x(i) - x(i+h))**2

    with N = number of valid pairs at that lag.
    """
    x = np.asarray(log, dtype=float)
    out = np.empty(max_lag)
    for h in range(1, max_lag + 1):
        diff = x[h:] - x[:-h]
        out[h - 1] = 0.5 * np.mean(diff ** 2)
    return out


def detect_layer_boundaries(log: np.ndarray, percentile: float = 90.0,
                            lag: int = 1) -> np.ndarray:
    """Layer-boundary detection via Lag-1 enhanced variance threshold.

    This corresponds to the P10 threshold on the Lag1-specific enhanced
    variance described in Fig. 4b of the paper (a "P10 variance" is the
    90th percentile of the *small* variance values -- i.e. the upper
    tail defines where the curve is locally jagged).

    Parameters
    ----------
    log : 1D log curve.
    percentile : percentile used as the threshold (paper uses 90 = P10 tail).
    lag : the lag (in samples) to use for the local variance estimate.

    Returns
    -------
    Integer indices where the local variance exceeds the threshold.
    """
    x = np.asarray(log, dtype=float)
    local_var = np.zeros_like(x)
    local_var[:-lag] = 0.5 * (x[lag:] - x[:-lag]) ** 2
    thr = np.percentile(local_var, percentile)
    return np.where(local_var > thr)[0]


# ---------------------------------------------------------------------------
# 2. Residual curve for layer classification --------------------------------
# ---------------------------------------------------------------------------

def residual_curve(gr: np.ndarray, short_window: int = 5,
                   long_window: int = 25) -> np.ndarray:
    """Compute the GR "residual" used by Strobel (2024).

    The raw curve is first median-filtered with a short window (service
    company default, ~5 samples), then again with a longer window (5-8 m
    in the paper; we pass the number of samples).  The residual is the
    absolute difference between the two filtered curves.
    """
    x = np.asarray(gr, dtype=float)
    short = median_filter(x, size=short_window, mode="nearest")
    long_ = median_filter(short, size=long_window, mode="nearest")
    return np.abs(short - long_)


def residual_cutoff(residual: np.ndarray, percentile: float = 90.0) -> float:
    """Histogram-derived cutoff to pick "streaks" of low clay content."""
    return float(np.percentile(residual, percentile))


# ---------------------------------------------------------------------------
# 3. Archie-type diffusivity model ------------------------------------------
# ---------------------------------------------------------------------------

def archie_effective_diffusivity(phi: float | np.ndarray, d0: float,
                                 m: float = 2.0) -> np.ndarray:
    """Effective diffusivity  De = D0 * phi^m.

    Parameters
    ----------
    phi : clay porosity (fraction).
    d0  : free-water diffusivity of the tracer (m^2/s).
    m   : cementation exponent (Archie-type), 1.8-2.5 for claystones.
    """
    phi_arr = np.asarray(phi, dtype=float)
    return d0 * np.power(phi_arr, m)


def modified_cementation_factor(rt: np.ndarray, rw: float,
                                phi: np.ndarray, a: float = 1.0) -> np.ndarray:
    """Extract m* (modified cementation factor) from resistivity and porosity.

    Starting from the classical Archie equation in a water-filled shaly
    interval,  Rt = a * Rw / phi^m  =>  m = log(a*Rw/Rt) / log(phi).
    The value obtained in this clay context is called m* because it is
    elevated by the cation-exchange-capacity of the clays.
    """
    rt_a = np.asarray(rt, dtype=float)
    phi_a = np.asarray(phi, dtype=float)
    # mask avoids log(<=0) warnings
    ok = (rt_a > 0) & (phi_a > 0) & (phi_a < 1)
    m_star = np.full_like(rt_a, np.nan)
    m_star[ok] = np.log(a * rw / rt_a[ok]) / np.log(phi_a[ok])
    return m_star


# ---------------------------------------------------------------------------
# Test harness ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def _synthetic_gr_log(n: int = 500, seed: int = 0) -> tuple[np.ndarray, list[int]]:
    """Build a synthetic GR log made of several layers plus noise."""
    rng = np.random.default_rng(seed)
    boundaries = [0, 80, 180, 260, 340, 430, n]
    means = [50, 110, 60, 130, 70, 120]
    log = np.empty(n)
    for a, b, mu in zip(boundaries[:-1], boundaries[1:], means):
        log[a:b] = mu
    log = log + rng.normal(0, 4, size=n)
    return log, boundaries[1:-1]


def test_all(verbose: bool = True) -> None:
    rng = np.random.default_rng(1)

    # (a) Variogram -- for a layered log the experimental variogram should
    # rise and plateau near the stationary variance.
    log, true_boundaries = _synthetic_gr_log()
    gamma = vertical_variogram(log, max_lag=30)
    assert gamma[0] < gamma[-1], "Variogram should increase"
    assert np.all(np.isfinite(gamma))

    # (b) Layer detection picks up most boundaries (within a few samples).
    picks = detect_layer_boundaries(log, percentile=95.0)
    hits = 0
    for b in true_boundaries:
        if np.any(np.abs(picks - b) <= 3):
            hits += 1
    assert hits >= len(true_boundaries) - 1, (
        f"Only found {hits} of {len(true_boundaries)} boundaries")

    # (c) Residual curve spikes at layer edges.
    res = residual_curve(log, short_window=5, long_window=25)
    cutoff = residual_cutoff(res, percentile=90.0)
    for b in true_boundaries:
        window = res[max(0, b - 15):b + 15]
        assert window.max() >= cutoff * 0.5, \
            f"Residual did not spike near boundary {b}"

    # (d) Archie-type diffusivity monotone in phi and m.
    phi = np.linspace(0.05, 0.3, 10)
    de = archie_effective_diffusivity(phi, d0=2.3e-9, m=2.1)
    assert np.all(np.diff(de) > 0), "De must increase with phi"
    assert archie_effective_diffusivity(0.2, 2.3e-9, m=2.5) < \
           archie_effective_diffusivity(0.2, 2.3e-9, m=1.5), \
           "Higher m -> lower De at fixed phi<1"

    # (e) Modified cementation factor round-trip on synthetic shaly sand.
    phi = np.array([0.1, 0.15, 0.2, 0.25])
    rw = 0.05
    m_true = 2.3
    rt = rw / phi ** m_true                      # noise-free
    m_rec = modified_cementation_factor(rt, rw, phi)
    assert np.allclose(m_rec, m_true, atol=1e-6), (
        f"m* round-trip failed: {m_rec}")

    if verbose:
        print("Article 2 (Claystone repository): all tests passed.")
        print(f"  variogram gamma(1)  = {gamma[0]:7.2f}")
        print(f"  variogram gamma(30) = {gamma[-1]:7.2f}")
        print(f"  layer boundary hits = {hits}/{len(true_boundaries)}")
        print(f"  residual cutoff P90 = {cutoff:6.2f}")
        print(f"  recovered m*        = {m_rec.mean():.3f} (true {m_true})")


if __name__ == "__main__":
    test_all()
