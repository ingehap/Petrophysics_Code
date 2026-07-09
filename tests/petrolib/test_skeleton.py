"""Tests for the petrolib skeleton: import policy, constants, units,
compat shims, and the shadow-equivalence helper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import petrolib  # noqa: E402
from petrolib import constants, testing, units  # noqa: E402
from petrolib._compat import deprecated, trapezoid  # noqa: E402

# --- import policy -----------------------------------------------------------


def test_version() -> None:
    assert petrolib.__version__


def test_lazy_submodules_resolve_and_dir_lists_them() -> None:
    for name in ("constants", "units", "testing"):
        assert getattr(petrolib, name) is not None
        assert name in dir(petrolib)
    with pytest.raises(AttributeError):
        _ = petrolib.no_such_submodule


def test_import_petrolib_needs_numpy_only() -> None:
    """`import petrolib` must not drag in scipy or other heavy deps."""
    code = (
        "import sys; import petrolib; petrolib.units, petrolib.constants; "
        "heavy = [m for m in ('scipy', 'sklearn', 'torch', 'xgboost', 'skimage') "
        "if m in sys.modules]; "
        "assert not heavy, f'petrolib imported heavy deps: {heavy}'"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
    )


# --- constants ---------------------------------------------------------------


def test_constants_consistency() -> None:
    assert constants.R_GAS == pytest.approx(constants.KB * constants.NA, rel=1e-12)
    assert constants.GAMMA_H == pytest.approx(2.0 * np.pi * constants.GAMMA_H_HZ, rel=1e-9)
    assert constants.PA_PER_PSI == pytest.approx(6894.757, abs=1e-3)


# --- units -------------------------------------------------------------------


def test_convert_pressure_round_trip_and_known_values() -> None:
    # The value reconciled in src2014_02/article2: 7 bar SI maximum ~ 101.5 psi
    assert units.convert(7.0, "bar", "psi") == pytest.approx(101.526, abs=1e-3)
    values = np.array([1.0, 10.0, 100.0])
    round_trip = units.convert(units.convert(values, "psi", "MPa"), "MPa", "psi")
    np.testing.assert_allclose(round_trip, values, rtol=1e-14)


def test_convert_temperature() -> None:
    np.testing.assert_allclose(units.convert([0.0, 100.0], "degC", "degF"), [32.0, 212.0])
    assert units.convert(75.0, "degF", "degC") == pytest.approx(23.888888, abs=1e-5)
    assert units.convert(273.15, "K", "degC") == pytest.approx(0.0, abs=1e-12)


def test_convert_permeability_and_length() -> None:
    assert units.convert(1.0, "D", "mD") == pytest.approx(1000.0)
    assert units.convert(1.0, "mD", "m2") == pytest.approx(9.869233e-16)
    assert units.convert(1.0, "ft", "m") == pytest.approx(0.3048)


def test_convert_rejects_unknown_and_cross_family() -> None:
    with pytest.raises(ValueError, match="unknown unit"):
        units.convert(1.0, "furlong", "m")
    with pytest.raises(ValueError, match="cannot convert"):
        units.convert(1.0, "psi", "m")


def test_slowness_velocity_adapters() -> None:
    # 55.5 us/ft is a classic sandstone matrix slowness ~ 5490 m/s
    v = units.slowness_to_velocity(55.5, "us/ft")
    assert v == pytest.approx(0.3048e6 / 55.5)
    np.testing.assert_allclose(units.velocity_to_slowness(v, "us/ft"), 55.5, rtol=1e-14)
    assert units.slowness_to_velocity(500.0, "us/m") == pytest.approx(2000.0)
    with pytest.raises(ValueError):
        units.slowness_to_velocity(100.0, "s/ft")


# --- _compat -----------------------------------------------------------------


def test_trapezoid_shim() -> None:
    x = np.linspace(0.0, 1.0, 101)
    assert trapezoid(x, x) == pytest.approx(0.5, abs=1e-4)


def test_deprecated_is_silent_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    @deprecated("use petrolib.units.convert")
    def old(x: float) -> float:
        return 2.0 * x

    monkeypatch.delenv("PETROLIB_WARN_DEPRECATED", raising=False)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert old(2.0) == 4.0  # no DeprecationWarning raised

    monkeypatch.setenv("PETROLIB_WARN_DEPRECATED", "1")
    with pytest.warns(DeprecationWarning, match="use petrolib.units.convert"):
        assert old(3.0) == 6.0


# --- testing.assert_matches_original -----------------------------------------


def test_assert_matches_original_passes_and_counts() -> None:
    def original(rt: float, rw: float, phi: float, m: float = 2.0) -> float:
        return float((rw / (phi**m * rt)) ** 0.5)

    def replacement(rt: float, rw: float, phi: float, m: float = 2.0) -> float:
        return float((rw / rt) ** 0.5 * phi ** (-m / 2.0))

    cases = [
        (40.0, 0.05, 0.25),
        ((40.0, 0.05, 0.25), {"m": 1.8}),
        (5.0, 0.02, 0.12),
    ]
    assert testing.assert_matches_original(original, replacement, cases) == 3


def test_assert_matches_original_catches_disagreement() -> None:
    with pytest.raises(AssertionError):
        testing.assert_matches_original(lambda x: x * 2.0, lambda x: x * 2.0 + 1e-6, [(1.0,)])


def test_assert_matches_original_rejects_empty_cases() -> None:
    with pytest.raises(AssertionError, match="no cases"):
        testing.assert_matches_original(lambda: 0, lambda: 0, [])


def test_assert_matches_original_structured_results() -> None:
    def original(x: float) -> dict[str, object]:
        return {"value": np.array([x, 2 * x]), "label": "ok", "pair": (x, x + 1)}

    testing.assert_matches_original(original, original, [(1.5,), (2.5,)])

    def broken(x: float) -> dict[str, object]:
        return {"value": np.array([x, 2 * x]), "label": "different", "pair": (x, x + 1)}

    with pytest.raises(AssertionError, match="label"):
        testing.assert_matches_original(original, broken, [(1.5,)])
