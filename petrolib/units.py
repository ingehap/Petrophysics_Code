"""Unit conversions for the quantities the article code mixes most.

The duplication analysis found pressure and depth conversions re-declared in
~19 files and unit mismatches behind several live bugs (psi/ft gradients
multiplied by metres; bar fed to a psi correlation).  This module gives one
vectorized, spelling-tolerant ``convert`` plus the sonic slowness adapters.

Conventions (see CONVENTIONS.md):
- SI-first: library physics is SI or unit-neutral; conversions happen at the
  edges, explicitly.
- ``convert`` only converts within one quantity family and raises
  ``ValueError`` across families or for unknown units — never guesses.

Sources: src2014_02/article2 (bar_to_psi/psi_to_bar), src2025_04/gip_porosity
(psi_to_mpa), and inline constants in ~16 further modules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import (
    KGM3_PER_GCC,
    M2_PER_DARCY,
    M_PER_FT,
    M_PER_IN,
    PA_PER_ATM,
    PA_PER_BAR,
    PA_PER_PSI,
)

# Linear families: unit -> factor to the family's SI base unit.
_FAMILIES: dict[str, dict[str, float]] = {
    "pressure [Pa]": {
        "pa": 1.0,
        "kpa": 1.0e3,
        "mpa": 1.0e6,
        "gpa": 1.0e9,
        "bar": PA_PER_BAR,
        "psi": PA_PER_PSI,
        "atm": PA_PER_ATM,
    },
    "length [m]": {
        "m": 1.0,
        "km": 1.0e3,
        "cm": 1.0e-2,
        "mm": 1.0e-3,
        "um": 1.0e-6,
        "nm": 1.0e-9,
        "ft": M_PER_FT,
        "in": M_PER_IN,
    },
    "time [s]": {
        "s": 1.0,
        "ms": 1.0e-3,
        "us": 1.0e-6,
        "min": 60.0,
        "hr": 3600.0,
        "day": 86400.0,
    },
    "permeability [m2]": {
        "m2": 1.0,
        "d": M2_PER_DARCY,
        "darcy": M2_PER_DARCY,
        "md": M2_PER_DARCY * 1.0e-3,
        "ud": M2_PER_DARCY * 1.0e-6,
        "nd": M2_PER_DARCY * 1.0e-9,
    },
    "density [kg/m3]": {
        "kg/m3": 1.0,
        "g/cc": KGM3_PER_GCC,
        "g/cm3": KGM3_PER_GCC,
    },
    "velocity [m/s]": {
        "m/s": 1.0,
        "km/s": 1.0e3,
        "ft/s": M_PER_FT,
    },
}

# Temperature is affine, handled separately: unit -> (scale, offset) with
# kelvin = scale * value + offset.
_TEMPERATURE_TO_K: dict[str, tuple[float, float]] = {
    "k": (1.0, 0.0),
    "degc": (1.0, 273.15),
    "degf": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
}
_TEMPERATURE_ALIASES = {"c": "degc", "f": "degf", "°c": "degc", "°f": "degf"}


def _normalize(unit: str) -> str:
    key = unit.strip().lower().replace("µ", "u")
    return _TEMPERATURE_ALIASES.get(key, key)


def _family_of(unit: str) -> str | None:
    for family, table in _FAMILIES.items():
        if unit in table:
            return family
    if unit in _TEMPERATURE_TO_K:
        return "temperature [K]"
    return None


def convert(values: ArrayLike, from_unit: str, to_unit: str) -> NDArray[np.float64]:
    """Convert ``values`` between two units of the same quantity family.

    >>> convert(7.0, "bar", "psi")
    array(101.52641996)
    >>> convert([100.0, 212.0], "degF", "degC")
    array([ 37.77777778, 100.        ])

    Raises ``ValueError`` for an unknown unit or a cross-family conversion
    (e.g. psi -> metres) — the caller's units are wrong and no answer is
    better than a plausible one.
    """
    src, dst = _normalize(from_unit), _normalize(to_unit)
    family_src, family_dst = _family_of(src), _family_of(dst)
    if family_src is None:
        raise ValueError(f"unknown unit {from_unit!r}")
    if family_dst is None:
        raise ValueError(f"unknown unit {to_unit!r}")
    if family_src != family_dst:
        raise ValueError(
            f"cannot convert {from_unit!r} ({family_src}) to {to_unit!r} ({family_dst})"
        )
    array = np.asarray(values, dtype=np.float64)
    if family_src == "temperature [K]":
        scale_src, offset_src = _TEMPERATURE_TO_K[src]
        scale_dst, offset_dst = _TEMPERATURE_TO_K[dst]
        kelvin = scale_src * array + offset_src
        return np.asarray((kelvin - offset_dst) / scale_dst)
    table = _FAMILIES[family_src]
    return np.asarray(array * (table[src] / table[dst]))


def slowness_to_velocity(slowness: ArrayLike, unit: str = "us/ft") -> NDArray[np.float64]:
    """Sonic slowness (``us/ft`` or ``us/m``) to velocity in m/s."""
    dt = np.asarray(slowness, dtype=np.float64)
    if unit == "us/ft":
        return np.asarray(M_PER_FT * 1.0e6 / dt)
    if unit == "us/m":
        return np.asarray(1.0e6 / dt)
    raise ValueError(f"unknown slowness unit {unit!r} (use 'us/ft' or 'us/m')")


def velocity_to_slowness(velocity: ArrayLike, unit: str = "us/ft") -> NDArray[np.float64]:
    """Velocity in m/s to sonic slowness (``us/ft`` or ``us/m``)."""
    v = np.asarray(velocity, dtype=np.float64)
    if unit == "us/ft":
        return np.asarray(M_PER_FT * 1.0e6 / v)
    if unit == "us/m":
        return np.asarray(1.0e6 / v)
    raise ValueError(f"unknown slowness unit {unit!r} (use 'us/ft' or 'us/m')")
