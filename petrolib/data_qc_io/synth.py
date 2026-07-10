"""Synthetic test data: blocky logs, log suites, shifted pairs, spectra,
sphere packs, and disk images.

Generators are deterministic by default (``rng=None`` seeds ``default_rng``
with the source article's seed) so examples and tests reproduce; pass a seed
or ``np.random.Generator`` to vary.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]
_Int = NDArray[np.int_]

#: Curve order of :func:`log_suite` columns.
LOG_SUITE_CURVES = ("GR", "RD", "RHOB", "DT", "NPHI")

#: Per-facies (rows) mean levels for the :data:`LOG_SUITE_CURVES` (columns):
#: granitic, shale, fractured granite, weathered.  Source: src2022_08/article5.
LOG_SUITE_FACIES_PROPS = np.array(
    [
        [22.0, 80.0, 2.45, 75.0, 0.08],
        [75.0, 10.0, 2.20, 95.0, 0.30],
        [35.0, 200.0, 2.60, 65.0, 0.05],
        [50.0, 30.0, 2.40, 80.0, 0.15],
    ]
)


def _arr(x: ArrayLike) -> _Float:
    return np.asarray(x, np.float64)


def _rng(rng: np.random.Generator | int | None, default_seed: int = 0) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(default_seed if rng is None else rng)


def blocky_log(
    n: int,
    n_beds: int = 8,
    *,
    level_range: tuple[float, float] = (20.0, 150.0),
    noise: float = 2.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[_Float, _Int]:
    """Blocky (square) log of ``n_beds`` constant levels plus Gaussian noise.

    Returns ``(curve, bounds)`` where ``bounds`` are the ``n_beds - 1``
    interior boundary indices, drawn without replacement away from the ends
    (margin ``max(1, n//15)``).  Levels are uniform on ``level_range``.
    Composed from the corpus generators: random breakpoints per
    src2022_08/article5, curve+bounds return per src2024_06/article2,
    ``noise=2.0`` per src2022_02/article2.
    """
    if n_beds < 1:
        raise ValueError("blocky_log: n_beds must be >= 1")
    gen = _rng(rng)
    margin = max(1, n // 15)
    interior = np.arange(margin, n - margin)
    if interior.size < n_beds - 1:
        raise ValueError("blocky_log: n too small for n_beds")
    bounds = np.sort(gen.choice(interior, n_beds - 1, replace=False)).astype(int)
    edges = np.concatenate(([0], bounds, [n]))
    levels = gen.uniform(level_range[0], level_range[1], n_beds)
    curve = np.zeros(n)
    for a, b, mu in zip(edges[:-1], edges[1:], levels, strict=True):
        curve[a:b] = mu
    return curve + gen.normal(0.0, noise, n), bounds


def log_suite(
    n: int = 600,
    n_facies: int = 4,
    *,
    rng: np.random.Generator | int | None = None,
) -> tuple[dict[str, _Float], _Int, _Int]:
    """Five-log synthetic well (GR, RD, RHOB, DT, NPHI) with facies blocks.

    Returns ``(logs, boundaries, facies_labels)``: a dict of the
    :data:`LOG_SUITE_CURVES`, the interior breakpoint indices, and the
    per-sample facies index.  Each block draws each curve from
    ``N(mu, 0.05*|mu| + 1e-3)`` with ``mu`` from
    :data:`LOG_SUITE_FACIES_PROPS`; breakpoints are drawn from
    ``arange(40, n-40)`` without replacement.  Source: src2022_08/article5
    (synth_log_suite, verbatim draw sequence).
    """
    if not 1 <= n_facies <= len(LOG_SUITE_FACIES_PROPS):
        raise ValueError(f"log_suite: n_facies must be in 1..{len(LOG_SUITE_FACIES_PROPS)}")
    gen = _rng(rng)
    bp_true = sorted(gen.choice(np.arange(40, n - 40), n_facies - 1, replace=False))
    bp_full = [0, *bp_true, n]
    facies_labels = np.zeros(n, dtype=int)
    out = np.zeros((n, 5))
    for k, (a, b) in enumerate(zip(bp_full[:-1], bp_full[1:], strict=True)):
        fac = k % n_facies
        facies_labels[a:b] = fac
        for j in range(5):
            mu = LOG_SUITE_FACIES_PROPS[fac, j]
            sigma = 0.05 * abs(mu) + 1e-3
            out[a:b, j] = gen.normal(mu, sigma, b - a)
    logs = {name: out[:, j] for j, name in enumerate(LOG_SUITE_CURVES)}
    return logs, np.array(bp_true, dtype=int), facies_labels


def shifted_pair(
    n: int,
    shift: int,
    *,
    noise: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[_Float, _Float]:
    """Reference curve and a copy lagged by ``shift`` samples.

    Both come from one multi-harmonic base with ``other[i] = ref[i + shift]``
    (noise-free) — aligning ``other`` to ``ref`` (e.g. with
    ``petrolib.depth_matching.xcorr_shift``) recovers ``shift``.  ``noise``
    adds independent Gaussian noise to each curve.  Pattern:
    src2024_10/rddtw_depth_matching (integer-shift core/log pair),
    src2022_02/article2 (rigid-lag windows).
    """
    s = int(shift)
    m = n + abs(s)
    z = np.linspace(0.0, 4.0 * np.pi, m)
    base = np.sin(z) + 0.5 * np.sin(3.0 * z) + 0.3 * np.cos(11.0 * z)
    ref = base[max(-s, 0) : max(-s, 0) + n]
    other = base[max(s, 0) : max(s, 0) + n]
    if noise > 0.0:
        gen = _rng(rng)
        ref = ref + noise * gen.standard_normal(n)
        other = other + noise * gen.standard_normal(n)
    return np.asarray(ref), np.asarray(other)


def gaussian_mixture_spectrum(
    axis: ArrayLike,
    centres: ArrayLike,
    amps: ArrayLike,
    widths: ArrayLike,
    *,
    noise: float = 0.0,
    log_axis: bool = True,
    rng: np.random.Generator | int | None = None,
) -> _Float:
    """Sum-of-Gaussians spectrum (synthetic NMR T2 / pore-size distributions).

    Each component contributes ``amp * exp(-0.5*((u - c)/width)**2)`` where
    ``u`` is ``log10(axis)`` and ``c`` is ``log10(centre)`` when
    ``log_axis=True`` (widths in decades), or the raw values otherwise —
    the standard-deviation form of src2026_06/a03 (gaussian_t2_model) and
    src2022_06/article1.  ``noise>0`` adds Gaussian noise and clips at zero
    (the src2022_06/article9 convention).
    """
    ax = _arr(axis)
    u = np.log10(ax) if log_axis else ax
    out = np.zeros_like(u)
    comps = zip(_arr(centres).ravel(), _arr(amps).ravel(), _arr(widths).ravel(), strict=True)
    for c, a, w in comps:
        cc = np.log10(c) if log_axis else c
        out += a * np.exp(-0.5 * ((u - cc) / w) ** 2)
    if noise > 0.0:
        out = out + noise * _rng(rng).standard_normal(u.size)
        out = np.clip(out, 0.0, None)
    return out


def sphere_pack_volume(
    shape: tuple[int, int, int],
    r_range: tuple[int, int],
    target_phi: float,
    *,
    rng: np.random.Generator | int | None = None,
    max_grains: int = 100_000,
) -> NDArray[np.uint8]:
    """Random sphere pack on a voxel grid (0 = pore, 1 = grain).

    Spheres with integer voxel radii uniform on ``r_range`` (inclusive) are
    stamped at uniform random centers until the pore fraction drops to
    ``target_phi`` (or ``max_grains`` is hit).  Grain fill uses the
    squared-distance broadcast of src2022_12/article7 (make_sand_pack).
    """
    gen = _rng(rng)
    nx, ny, nz = shape
    cube = np.zeros(shape, dtype=np.uint8)
    x = np.arange(nx)[:, None, None]
    y = np.arange(ny)[None, :, None]
    z = np.arange(nz)[None, None, :]
    r_lo, r_hi = r_range
    for _ in range(max_grains):
        if 1.0 - cube.mean() <= target_phi:
            break
        cx = int(gen.integers(0, nx))
        cy = int(gen.integers(0, ny))
        cz = int(gen.integers(0, nz))
        r = int(gen.integers(r_lo, r_hi + 1))
        d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        cube[d2 <= r * r] = 1
    return cube


def disk_image(
    size: int,
    n_features: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> tuple[_Float, NDArray[np.bool_]]:
    """Grey image with darker random disks; returns ``(image, mask)``.

    Background ``0.6 + 0.05*N(0,1)``, ``n_features`` disks of radius 2-7 px
    at uniform centers, disk pixels ``0.15 + 0.05*N(0,1)``, clipped to
    [0, 1] — a synthetic SEM/pore image.  Source: src2024_04/
    chen_sem_pore_segmentation (synthetic_sem_image, verbatim draw
    sequence; that article seeds ``default_rng(4)``).
    """
    gen = _rng(rng)
    img = 0.6 + 0.05 * gen.standard_normal((size, size))
    mask = np.zeros((size, size), dtype=bool)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(n_features):
        cy, cx = gen.integers(0, size, size=2)
        r = gen.integers(2, 8)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        mask |= m
    img[mask] = 0.15 + 0.05 * gen.standard_normal(mask.sum())
    return np.clip(img, 0, 1), mask
