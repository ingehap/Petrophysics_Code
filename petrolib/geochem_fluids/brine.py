"""Brine resistivity, salinity, capture cross-section, and density.

Water resistivity from NaCl salinity and temperature (Bateman-Konen R75 plus the
Arps temperature correction), the exact inverse, NaCl equivalents, the thermal-
neutron capture cross-section, and a simplified Batzle-Wang brine density.

Unit policy: salinity in ppm (NaCl-equivalent), resistivity in ohm-m, capture
cross-section in capture units (c.u.).  Temperature is passed with an explicit
``unit=`` ("F" or "C"); the Arps constant is 6.77 for Fahrenheit and 21.5 for
Celsius (the corpus uses a 7.0 variant in one file -- pass it via ``c=``).
Sources: src2015_06/article4, src2015_08/article5, src2020_12/article2,
src2023_08/article1, src2023_10/article_01, src2026_06/a03.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]

# Arps temperature-correction constant by temperature unit (ohm-m form).
ARPS_C = {"F": 6.77, "C": 21.5}
# NaCl molar mass (g/mol), for meq/L conversion.
_M_NACL = 58.44


def _arps_c(unit: str) -> float:
    try:
        return ARPS_C[unit]
    except KeyError:
        raise ValueError(f"unit must be 'F' or 'C', got {unit!r}") from None


def rw75_from_salinity(nacl_ppm: ArrayLike) -> _Float:
    """Bateman-Konen water resistivity at 75 degF ``R75 = 0.0123 + 3647.5/C**0.955``."""
    c = np.asarray(nacl_ppm, np.float64)
    return np.asarray(0.0123 + 3647.5 / c**0.955)


def salinity_from_rw75(rw75_ohmm: ArrayLike) -> _Float:
    """NaCl ppm from the 75 degF resistivity -- exact inverse of ``rw75_from_salinity``."""
    r = np.asarray(rw75_ohmm, np.float64)
    return np.asarray((3647.5 / (r - 0.0123)) ** (1.0 / 0.955))


def arps_correct(r1: ArrayLike, t1: ArrayLike, t2: ArrayLike, *, unit: str = "F") -> _Float:
    """Arps temperature correction ``R2 = R1*(T1+c)/(T2+c)``.

    ``c`` is 6.77 for ``unit="F"`` or 21.5 for ``unit="C"``.  Applies equally to
    conductivity with the roles of R inverted by the caller.
    """
    c = _arps_c(unit)
    r1a = np.asarray(r1, np.float64)
    return np.asarray(r1a * (np.asarray(t1, np.float64) + c) / (np.asarray(t2, np.float64) + c))


def rw_from_salinity(nacl_ppm: ArrayLike, temp: ArrayLike, *, unit: str = "F") -> _Float:
    """Water resistivity at ``temp`` from NaCl ppm (75 degF reference).

    Computes R75 (Bateman-Konen) then Arps-corrects from 75 degF to ``temp``.
    ``unit="C"`` converts ``temp`` to Fahrenheit first (75 degF reference kept).
    """
    rw75 = rw75_from_salinity(nacl_ppm)
    t = np.asarray(temp, np.float64)
    t_f = t if unit == "F" else t * 9.0 / 5.0 + 32.0
    return arps_correct(rw75, 75.0, t_f, unit="F")


def salinity_from_rw(rw: ArrayLike, temp: ArrayLike, *, unit: str = "F") -> _Float:
    """NaCl ppm from water resistivity at ``temp`` -- inverse of ``rw_from_salinity``."""
    t = np.asarray(temp, np.float64)
    t_f = t if unit == "F" else t * 9.0 / 5.0 + 32.0
    rw75 = arps_correct(rw, t_f, 75.0, unit="F")
    return salinity_from_rw75(rw75)


def sigma_w_from_salinity(nacl_ppm: ArrayLike, temp_c: ArrayLike = 24.0) -> _Float:
    """Thermal-neutron capture cross-section of brine (c.u.).

    ``Sigma_w = (22 + 750*w)*(1 - 8e-4*(T - 75))`` with ``w`` the NaCl weight
    fraction; 22 c.u. is the fresh-water baseline.  Sources: src2023_08/article1.
    """
    w = np.asarray(nacl_ppm, np.float64) / 1.0e6
    base = 22.0 + 750.0 * w
    return np.asarray(base * (1.0 - 0.0008 * (np.asarray(temp_c, np.float64) - 75.0)))


def nacl_meq_per_liter(nacl_ppm: ArrayLike) -> _Float:
    """NaCl concentration in meq/L ``= ppm/58.44`` (monovalent, brine density ~1)."""
    return np.asarray(np.asarray(nacl_ppm, np.float64) / _M_NACL)


def brine_density_bw92(
    nacl_ppm: ArrayLike, temp_c: ArrayLike, press_mpa: ArrayLike = 0.1
) -> _Float:
    """Simplified Batzle-Wang (1992) brine density (kg/m3).

    ``rho_w = 1000*(1 - (T-4)**2/178000)`` (pure water) plus the NaCl term
    ``S*(300 - 2*T - 60*S)`` with ``S`` the weight fraction; a small linear
    pressure term ``+0.045*(P-0.1)`` (P in MPa).  Sources: src2023_10/article_01.
    """
    s = np.asarray(nacl_ppm, np.float64) * 1.0e-6
    t = np.asarray(temp_c, np.float64)
    rho_w = 1000.0 * (1.0 - (t - 4.0) ** 2 / 178000.0)
    rho_b = rho_w + s * (300.0 - 2.0 * t - 60.0 * s)
    return np.asarray(rho_b + 0.045 * (np.asarray(press_mpa, np.float64) - 0.1) * 1000.0)
