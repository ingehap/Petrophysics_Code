"""
article5_lwd_image_deeplearning.py
==================================

Implementation of the deep-learning workflow for LWD image interpretation
from:

    Molossi, A., Roncoroni, G., and Pipan, M. (2024).
    "Efficient Logging-While-Drilling Image Logs Interpretation Using
    Deep Learning."  Petrophysics 65(3), 365-387.
    DOI: 10.30632/PJV65N3-2024a5

The paper combines a U-Net ("PickNet") that segments geological edges in
a 20x16 LWD azimuthal density image with a fully-connected network
("FitNet") that fits sinusoids to the segmented edges.  Since the actual
paper trains on 10^6 synthetic images, we do something equivalent that
does not require a GPU:

* `SyntheticLWDGenerator` reproduces the image synthesis recipe of
  Appendix 1: for every image, place a random number of density
  contrasts along random sinusoidal paths and add Gaussian noise.

* `pick_edges`  approximates what the U-Net PickNet does: it finds
  per-azimuth depth positions of strong vertical gradients.  This is
  the deterministic analogue of the binary segmentation mask used in
  the paper.

* `fit_sinusoid`  performs the least-squares fit the FitNet learns,
  giving amplitude A, phase, and mean depth y0 of a sinusoid

        y(x) = y0 + A * sin(2*pi*x/W + phase).

* `dip_from_amplitude`  converts amplitude to dip magnitude using the
  paper's Eq. 1:   dip = arctan(A / D)   where D is borehole diameter.

* `mc_dropout_uncertainty`  provides a simple Monte-Carlo-dropout style
  uncertainty estimate (Gal & Ghahramani, 2016) by randomly omitting a
  subset of the picked points from the fit, as the CNN does with
  dropped neurons.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Synthetic LWD image generator ---------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class LWDImage:
    image: np.ndarray          # (n_depth, n_azimuth)
    mask: np.ndarray           # binary mask of edges, same shape
    sinusoids: list[tuple[float, float, float]]  # list of (y0, A, phase)


def generate_synthetic_lwd(n_depth: int = 20, n_azimuth: int = 16,
                           max_edges: int = 3, noise: float = 0.1,
                           seed: int | None = None) -> LWDImage:
    """Generate one synthetic LWD image with 0..max_edges sinusoidal edges.

    Each edge is a sinusoid  y(x) = y0 + A*sin(2*pi*x/W + phase)  drawn on
    a depth-x-azimuth grid.  The image is smoothly varying between edges
    and steps across them; additive Gaussian noise provides robustness
    for the subsequent edge detector.
    """
    rng = np.random.default_rng(seed)
    n_edges = int(rng.integers(0, max_edges + 1))

    # Random levels for each stratigraphic layer
    levels = rng.uniform(0.2, 0.8, size=n_edges + 1)
    # Random sinusoid parameters
    sinusoids: list[tuple[float, float, float]] = []
    y0_candidates = np.sort(rng.uniform(2, n_depth - 3, size=n_edges))
    for y0 in y0_candidates:
        A = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        sinusoids.append((float(y0), float(A), float(phase)))

    image = np.zeros((n_depth, n_azimuth))
    mask = np.zeros_like(image, dtype=int)

    x = np.arange(n_azimuth)
    # Build per-column boundary positions for each sinusoid
    boundaries = np.zeros((n_edges, n_azimuth))
    for e, (y0, A, phase) in enumerate(sinusoids):
        boundaries[e] = y0 + A * np.sin(2 * np.pi * x / n_azimuth + phase)

    for col in range(n_azimuth):
        col_bounds = np.sort(boundaries[:, col]) if n_edges else np.array([])
        # layer id at each depth
        for depth in range(n_depth):
            layer = int(np.searchsorted(col_bounds, depth))
            image[depth, col] = levels[layer]
        # Mark edge pixels (closest integer depth to each boundary)
        for yb in col_bounds:
            d = int(round(yb))
            if 0 <= d < n_depth:
                mask[d, col] = 1

    image += rng.normal(0, noise, size=image.shape)
    return LWDImage(image=image, mask=mask, sinusoids=sinusoids)


# ---------------------------------------------------------------------------
# PickNet analogue: edge detection -------------------------------------------
# ---------------------------------------------------------------------------

def pick_edges(image: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """For every azimuth column, return the depth index of the strongest
    vertical gradient -- a deterministic stand-in for the binary mask the
    U-Net PickNet produces.

    Returns an integer array of length n_azimuth with the picked depth
    (or -1 if no gradient exceeded the threshold).
    """
    img = np.asarray(image, dtype=float)
    grad = np.abs(np.diff(img, axis=0))
    if threshold is None:
        threshold = 0.15 * (img.max() - img.min())
    picks = np.full(img.shape[1], -1, dtype=int)
    for col in range(img.shape[1]):
        g = grad[:, col]
        if g.max() > threshold:
            picks[col] = int(np.argmax(g))
    return picks


# ---------------------------------------------------------------------------
# FitNet analogue: sinusoid fit ---------------------------------------------
# ---------------------------------------------------------------------------

def fit_sinusoid(picks: np.ndarray, n_azimuth: int) -> tuple[float, float, float]:
    """Linear least-squares fit of  y = y0 + A*sin(theta + phase).

    Expanding,  y = y0 + a*sin(theta) + b*cos(theta)  with  A = sqrt(a^2+b^2)
    and phase = atan2(b, a).

    `picks` contains depth values; entries with value < 0 are ignored.
    """
    mask = picks >= 0
    if mask.sum() < 3:
        raise ValueError("need at least 3 valid picks to fit")
    col_idx = np.arange(n_azimuth)[mask]
    y = picks[mask].astype(float)
    theta = 2 * np.pi * col_idx / n_azimuth
    G = np.column_stack([np.ones_like(theta), np.sin(theta), np.cos(theta)])
    coef, *_ = np.linalg.lstsq(G, y, rcond=None)
    y0, a, b = coef
    amp = float(np.sqrt(a * a + b * b))
    phase = float(np.arctan2(b, a))
    return float(y0), amp, phase


def dip_from_amplitude(amplitude_samples: float, borehole_diameter_m: float,
                       sample_spacing_m: float = 0.05) -> float:
    """Eq. 1 of Molossi et al.: dip = arctan(peak-to-peak / D) ~ arctan(2A/D).

    `amplitude_samples`  is the sinusoid amplitude in depth samples, and
    `sample_spacing_m` converts samples to metres.  The paper uses a
    5 cm sampling rate.
    """
    A_m = amplitude_samples * sample_spacing_m
    return float(np.degrees(np.arctan(2 * A_m / borehole_diameter_m)))


# ---------------------------------------------------------------------------
# Monte-Carlo dropout uncertainty on the dip estimate -----------------------
# ---------------------------------------------------------------------------

def mc_dropout_uncertainty(picks: np.ndarray, n_azimuth: int,
                           borehole_diameter_m: float,
                           sample_spacing_m: float = 0.05,
                           dropout: float = 0.25, n_passes: int = 50,
                           seed: int | None = None) -> tuple[float, float]:
    """Estimate mean & std of dip magnitude by repeated fits with random
    subsets of picks dropped -- an MC-dropout-style uncertainty (Gal &
    Ghahramani, 2016).  Returns (mean_dip_deg, std_dip_deg)."""
    rng = np.random.default_rng(seed)
    dips = []
    for _ in range(n_passes):
        keep = rng.random(len(picks)) > dropout
        trial = np.where(keep, picks, -1)
        if (trial >= 0).sum() >= 3:
            try:
                _, A, _ = fit_sinusoid(trial, n_azimuth)
                dips.append(dip_from_amplitude(A, borehole_diameter_m,
                                               sample_spacing_m))
            except ValueError:
                continue
    if not dips:
        return float("nan"), float("nan")
    return float(np.mean(dips)), float(np.std(dips))


# ---------------------------------------------------------------------------
# Test harness ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_all(verbose: bool = True) -> None:
    rng_seed = 7

    # (a) Synthetic data generator produces a 2D image with the expected shape
    sample = generate_synthetic_lwd(n_depth=20, n_azimuth=16, max_edges=1,
                                    noise=0.02, seed=rng_seed)
    assert sample.image.shape == (20, 16)
    assert sample.mask.shape == (20, 16)
    assert len(sample.sinusoids) <= 1

    # (b) Recover a single sinusoid: generate an image that definitely
    # contains exactly one clear edge, pick it, fit it, and check the
    # amplitude.  We force exactly one edge by re-seeding until we get it.
    for attempt in range(20):
        sample = generate_synthetic_lwd(max_edges=1, noise=0.01,
                                        seed=rng_seed + attempt)
        if len(sample.sinusoids) == 1:
            break
    assert len(sample.sinusoids) == 1, "could not produce 1-edge image"
    y0_true, A_true, phase_true = sample.sinusoids[0]
    picks = pick_edges(sample.image)
    y0_est, A_est, phase_est = fit_sinusoid(picks, n_azimuth=16)
    assert abs(A_est - A_true) < 1.5, f"A: {A_est} vs {A_true}"
    assert abs(y0_est - y0_true) < 1.5, f"y0: {y0_est} vs {y0_true}"

    # (c) Dip magnitude consistency: larger amplitude -> larger dip
    dip_small = dip_from_amplitude(0.5, borehole_diameter_m=0.2)
    dip_large = dip_from_amplitude(3.0, borehole_diameter_m=0.2)
    assert dip_large > dip_small
    assert 0 < dip_large < 90

    # (d) MC dropout produces a reasonable uncertainty estimate
    mean_dip, std_dip = mc_dropout_uncertainty(
        picks, n_azimuth=16, borehole_diameter_m=0.2, n_passes=30, seed=0)
    assert not np.isnan(mean_dip) and not np.isnan(std_dip)
    assert std_dip >= 0

    if verbose:
        print("Article 5 (LWD deep learning): all tests passed.")
        print(f"  true  A = {A_true:.3f}, phase = {phase_true:+.2f}")
        print(f"  est   A = {A_est:.3f}, phase = {phase_est:+.2f}")
        print(f"  dip (A=0.5) = {dip_small:.2f} deg")
        print(f"  dip (A=3.0) = {dip_large:.2f} deg")
        print(f"  MC dip      = {mean_dip:.2f} +/- {std_dip:.2f} deg")


if __name__ == "__main__":
    test_all()
