"""Tests for the golden-comparison tooling (numeric tolerance + normalization)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


golden_capture = _load("golden_capture")
golden_diff = _load("golden_diff")

RTOL, ATOL = golden_diff.DEFAULT_RTOL, golden_diff.DEFAULT_ATOL


def eq(a: str, b: str) -> bool:
    return golden_diff.numerically_equal(a, b, RTOL, ATOL)


def test_exact_match() -> None:
    assert eq("phi = 0.25\nPASS\n", "phi = 0.25\nPASS\n")


def test_last_digit_repr_drift_passes() -> None:
    # observed on GitHub runners vs the capture machine (different BLAS)
    assert eq(
        "boundaries = [40.627691142242355, 58.095441613188974]",
        "boundaries = [40.62769114224234, 58.09544161318896]",
    )


def test_near_zero_residual_noise_passes() -> None:
    assert eq("RSS = 1.32e-27", "RSS = 3.06e-27")
    assert eq("RBF train RMSE = 1.45e-15", "RBF train RMSE = 7.63e-16")


def test_real_numeric_drift_fails() -> None:
    assert not eq("Sw = 0.4520", "Sw = 0.4530")  # 2e-3 relative: real drift
    assert not eq("k = 41.0 mD", "k = 41.2 mD")


def test_text_changes_fail() -> None:
    assert not eq("PASS", "FAIL")
    assert not eq("phi = 0.25", "porosity = 0.25")
    assert not eq("a = 1.0 b = 2.0", "a = 1.0")


def test_scientific_notation_and_integers() -> None:
    assert eq(
        "N = 13 passed, k = 1.0138432109866983e-19", "N = 13 passed, k = 1.0138432109866695e-19"
    )
    assert not eq("N = 13 passed", "N = 12 passed")


def test_normalize_masks_timings_and_collapses_padding() -> None:
    line_a = "|  9 passed, 0 failed  (92.2s)     |"
    line_b = "|  9 passed, 0 failed  (142.5s)    |"
    assert golden_capture.normalize(line_a) == golden_capture.normalize(line_b)
    # non-timing alignment is preserved untouched
    table = "  PASS   article1    ok\n"
    assert golden_capture.normalize(table) == table
