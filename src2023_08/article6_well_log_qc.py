"""
article6_well_log_qc.py
========================
Implements the well-log data validation, visualisation-helper, and
repeatability checks described in:

    Jin, Y., Xu, C., Lin, T., Li, W., Zeghlache, M.L. (2023).
    "Python Dash for Well Data Validation, Visualization, and Processing",
    Petrophysics, Vol. 64, No. 4, pp. 568-573.
    DOI: 10.30632/PJV64N4-2023a6

Implemented features
--------------------
1. validate_log_set()        -- the 4-rule data-integrity check from
                                "Data Validation":
                                   * required channels present?
                                   * redundant channels?
                                   * units correct?
                                   * value-validity (NaN / out-of-range)
2. summary_table()           -- summary dict / list-of-records suitable for
                                rendering in a Plotly-Dash DataTable.
3. log_difference()          -- Eq. 1: difference between repeat and main.
4. pearson_correlation()     -- Eq. 2: Pearson r between repeat and main.
5. repeatability_metrics()   -- bulk histogram shift + cross-correlation
                                vs. depth-shift, returning the optimal
                                depth shift and peak correlation
                                (the "0.5 ft" example in Fig. 5).

This module deliberately does NOT depend on Dash itself -- the article's
contribution is the *workflow*, which is the testable part.  The Dash
front-end is just a viewer of these results.

Run as a script for the synthetic test suite:

    python article6_well_log_qc.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Validation rules
# ---------------------------------------------------------------------------
@dataclass
class ValidationConfig:
    """Per-tool validation rules loaded from a config file in the paper."""
    required_channels: Sequence[str]
    expected_units:    Dict[str, str]            = field(default_factory=dict)
    valid_ranges:      Dict[str, Tuple[float, float]] = field(default_factory=dict)


def validate_log_set(channels: Dict[str, np.ndarray],
                     units:    Dict[str, str],
                     config:   ValidationConfig):
    """Run the four validation rules listed in the paper.

    `channels` maps channel name -> numpy array of values.
    `units`    maps channel name -> unit string.

    Returns a dict with keys 'missing', 'redundant', 'wrong_units',
    'bad_intervals' (list of (channel, idx_array, reason)).
    """
    present = set(channels.keys())
    required = set(config.required_channels)
    missing = sorted(required - present)
    redundant = sorted(present - required)
    wrong_units = []
    for ch, expected in config.expected_units.items():
        if ch in units and units[ch] != expected:
            wrong_units.append((ch, units[ch], expected))
    bad_intervals = []
    for ch, arr in channels.items():
        arr = np.asarray(arr, dtype=float)
        bad = np.where(np.isnan(arr))[0]
        if bad.size:
            bad_intervals.append((ch, bad.tolist(), "NaN"))
        if ch in config.valid_ranges:
            lo, hi = config.valid_ranges[ch]
            oor = np.where((arr < lo) | (arr > hi))[0]
            if oor.size:
                bad_intervals.append((ch, oor.tolist(), f"out [{lo},{hi}]"))
    return dict(missing=missing,
                redundant=redundant,
                wrong_units=wrong_units,
                bad_intervals=bad_intervals)


def summary_table(validation_result, channels):
    """Produce a list-of-dicts summary suitable for a Dash DataTable."""
    summary = []
    flagged = {ch for ch, *_ in validation_result["bad_intervals"]}
    for ch, arr in channels.items():
        arr = np.asarray(arr, dtype=float)
        n_finite = int(np.sum(np.isfinite(arr)))
        summary.append(dict(
            channel=ch,
            n_samples=len(arr),
            n_valid=n_finite,
            n_nan=int(np.sum(np.isnan(arr))),
            min=float(np.nanmin(arr)) if n_finite else float("nan"),
            max=float(np.nanmax(arr)) if n_finite else float("nan"),
            mean=float(np.nanmean(arr)) if n_finite else float("nan"),
            flagged=ch in flagged,
        ))
    return summary


# ---------------------------------------------------------------------------
# 2. Eq. 1 -- log difference
# ---------------------------------------------------------------------------
def log_difference(main, repeat):
    """diff[i] = repeat[i] - main[i]  (Eq. 1)."""
    return np.asarray(repeat, dtype=float) - np.asarray(main, dtype=float)


# ---------------------------------------------------------------------------
# 3. Eq. 2 -- Pearson correlation
# ---------------------------------------------------------------------------
def pearson_correlation(x, y):
    """Pearson r between repeat (x) and main (y).  NaNs are pair-wise dropped."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float("nan")
    xbar, ybar = x.mean(), y.mean()
    num = np.sum((x - xbar) * (y - ybar))
    den = np.sqrt(np.sum((x - xbar) ** 2) * np.sum((y - ybar) ** 2))
    return float(num / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# 4. Repeatability metrics -- depth-shift cross-correlation + histogram
# ---------------------------------------------------------------------------
def cross_correlation_vs_shift(main, repeat, max_shift=20):
    """Pearson r(main, shifted repeat) for shifts in [-max_shift, +max_shift].

    Returns (shifts, correlations).  A positive shift means the repeat curve
    is shifted *down* (deeper) relative to the main.
    """
    main = np.asarray(main, dtype=float)
    repeat = np.asarray(repeat, dtype=float)
    n = len(main)
    shifts, corrs = [], []
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            a = main[s:]
            b = repeat[: n - s]
        else:
            a = main[: n + s]
            b = repeat[-s:]
        if len(a) > 5:
            shifts.append(s)
            corrs.append(pearson_correlation(a, b))
    return np.asarray(shifts), np.asarray(corrs)


def repeatability_metrics(main, repeat, depth_step=0.5, max_shift=20,
                          n_hist_bins=30):
    """Compute the repeatability summary for a main vs. repeat pass.

    Returns dict with:
        diff             -- per-sample difference array (Eq. 1)
        pearson          -- Pearson correlation at zero shift (Eq. 2)
        bulk_shift       -- difference between repeat and main *means*
        hist_shift       -- shift of the histogram peak (units of `main`)
        best_depth_shift -- optimal depth shift (in metres / feet)
        peak_corr        -- maximum cross-correlation value
        shifts           -- depth shifts probed (in physical units)
        corrs            -- per-shift correlations
        passed           -- bool, True if peak_corr >= 0.95 (typical QC cut)
    """
    main = np.asarray(main, dtype=float)
    repeat = np.asarray(repeat, dtype=float)
    diff = log_difference(main, repeat)
    pearson = pearson_correlation(main, repeat)
    # Bulk histogram shift: difference in means
    bulk_shift = float(np.nanmean(repeat) - np.nanmean(main))
    # Histogram-mode shift
    lo = np.nanmin(np.concatenate([main, repeat]))
    hi = np.nanmax(np.concatenate([main, repeat]))
    bins = np.linspace(lo, hi, n_hist_bins + 1)
    h_main, _   = np.histogram(main[np.isfinite(main)],     bins=bins)
    h_repeat, _ = np.histogram(repeat[np.isfinite(repeat)], bins=bins)
    centres = 0.5 * (bins[:-1] + bins[1:])
    hist_shift = float(centres[np.argmax(h_repeat)] - centres[np.argmax(h_main)])
    # Depth-shift cross-correlation
    shifts_idx, corrs = cross_correlation_vs_shift(main, repeat, max_shift)
    if corrs.size and np.any(np.isfinite(corrs)):
        i_peak = int(np.nanargmax(corrs))
        best_shift = float(shifts_idx[i_peak]) * depth_step
        peak_corr  = float(corrs[i_peak])
    else:
        best_shift, peak_corr = 0.0, float("nan")
    return dict(
        diff=diff,
        pearson=pearson,
        bulk_shift=bulk_shift,
        hist_shift=hist_shift,
        best_depth_shift=best_shift,
        peak_corr=peak_corr,
        shifts=shifts_idx * depth_step,
        corrs=corrs,
        passed=peak_corr >= 0.95,
    )


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
def test_all(verbose=True):
    rng = np.random.default_rng(2024)

    # --- Synthetic well-log dataset ----------------------------------------
    n = 500
    depth = np.linspace(2000.0, 2100.0, n)         # m
    GR    = 50.0 + 30.0 * np.sin(depth / 4.0) + rng.normal(scale=2.0, size=n)
    RHOB  = 2.55 + 0.10 * np.cos(depth / 3.0) + rng.normal(scale=0.01, size=n)
    NPHI  = 0.20 - 0.05 * np.sin(depth / 5.0) + rng.normal(scale=0.005, size=n)
    EXTRA = rng.uniform(0, 1, n)                   # redundant
    GR[10:15] = np.nan                             # bad samples
    NPHI[100] = -1.0                               # out of range

    channels = dict(GR=GR, RHOB=RHOB, NPHI=NPHI, EXTRA=EXTRA)
    units    = dict(GR="GAPI", RHOB="g/cc", NPHI="v/v", EXTRA="-")

    cfg = ValidationConfig(
        required_channels=["GR", "RHOB", "NPHI", "DT"],
        expected_units={"GR": "GAPI", "RHOB": "g/cc", "NPHI": "v/v"},
        valid_ranges={"GR": (0.0, 300.0),
                      "RHOB": (1.5, 3.2),
                      "NPHI": (-0.05, 0.60)},
    )

    # --- 1. Validation -----------------------------------------------------
    res = validate_log_set(channels, units, cfg)
    assert "DT" in res["missing"], "DT should be flagged as missing"
    assert "EXTRA" in res["redundant"]
    assert any(ch == "GR" and reason == "NaN" for ch, _, reason in res["bad_intervals"])
    assert any(ch == "NPHI" for ch, _, _ in res["bad_intervals"])
    if verbose:
        print(f"[1] Validation OK             "
              f"(missing={res['missing']}, redundant={res['redundant']}, "
              f"flagged={[ch for ch,_,_ in res['bad_intervals']]})")

    # --- 2. Summary table --------------------------------------------------
    tbl = summary_table(res, channels)
    flags = {row["channel"]: row["flagged"] for row in tbl}
    assert flags["GR"] and flags["NPHI"] and not flags["RHOB"]
    if verbose:
        print(f"[2] Summary table OK          ({len(tbl)} rows)")

    # --- 3. Log difference + Pearson --------------------------------------
    main_pass    = GR.copy()
    repeat_pass  = GR + rng.normal(scale=0.5, size=n)        # noisy clone
    repeat_pass[np.isnan(main_pass)] = np.nan
    diff = log_difference(main_pass, repeat_pass)
    r = pearson_correlation(main_pass, repeat_pass)
    assert r > 0.99, f"Repeat should correlate strongly with main, r={r}"
    if verbose:
        print(f"[3] Diff + Pearson OK         (r = {r:.4f}, "
              f"mean |diff| = {np.nanmean(np.abs(diff)):.3f})")

    # --- 4. Cross-correlation finds the inserted depth shift ---------------
    # Insert a 4-sample depth shift (== 4 * 0.5 ft = 2 ft if depth_step=0.5)
    shift = 4
    repeat_shifted = np.roll(GR, shift)
    repeat_shifted[:shift] = np.nan
    rep_metrics = repeatability_metrics(main_pass, repeat_shifted,
                                        depth_step=0.5, max_shift=15)
    found_shift = rep_metrics["best_depth_shift"]
    assert abs(found_shift - (-shift * 0.5)) < 0.5 \
        or abs(found_shift - (shift * 0.5)) < 0.5, \
        f"Expected ~+/-{shift*0.5} ft, found {found_shift}"
    if verbose:
        print(f"[4] Repeatability metrics OK  "
              f"(best shift = {found_shift:+.2f} ft, peak r = "
              f"{rep_metrics['peak_corr']:.3f})")

    if verbose:
        print("\nAll article-6 tests passed.")
    return True


if __name__ == "__main__":
    test_all()
