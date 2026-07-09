"""OBM/filtrate contamination mixing and cleanup.

Linear property mixing between virgin and filtrate fluid, the contamination
fraction (its inverse), power-law cleanup vs pumped volume, and the pumped volume
needed to reach a contamination target.

Unit-neutral; ``eta`` is the filtrate (contamination) fraction.  Sources:
src2015_06/article2, src2017_08/article5, src2018_10/article6, src2021_02/article5,
src2024_02/article2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]


def mix_linear(p_v: ArrayLike, p_f: ArrayLike, eta: ArrayLike) -> _Float:
    """Linear mixing ``P = (1 - eta)*P_virgin + eta*P_filtrate``."""
    e = np.asarray(eta, np.float64)
    return np.asarray((1.0 - e) * np.asarray(p_v, np.float64) + e * np.asarray(p_f, np.float64))


def contamination_fraction(p: ArrayLike, p_v: ArrayLike, p_f: ArrayLike) -> _Float:
    """Contamination fraction ``eta = (P - P_v)/(P_f - P_v)`` (inverse of the mixing rule)."""
    pv = np.asarray(p_v, np.float64)
    return np.asarray((np.asarray(p, np.float64) - pv) / (np.asarray(p_f, np.float64) - pv))


def cleanup_powerlaw(
    v: ArrayLike, eta0: ArrayLike, v_star: ArrayLike, *, exponent: float = 5.0 / 12.0
) -> _Float:
    """Power-law cleanup ``eta(V) = eta0*(1 + V/V_star)**(-exponent)``.

    The default exponent 5/12 is the wireline-cleanup value (2/3 for radial).
    """
    ratio = np.asarray(v, np.float64) / np.asarray(v_star, np.float64)
    return np.asarray(np.asarray(eta0, np.float64) * (1.0 + ratio) ** (-exponent))


def volume_to_target(
    eta0: ArrayLike, v_star: ArrayLike, eta_t: ArrayLike, *, exponent: float = 5.0 / 12.0
) -> _Float:
    """Pumped volume to reach contamination ``eta_t`` -- inverse of :func:`cleanup_powerlaw`.

    ``V = V_star*((eta0/eta_t)**(1/exponent) - 1)``.
    """
    ratio = np.asarray(eta0, np.float64) / np.asarray(eta_t, np.float64)
    return np.asarray(np.asarray(v_star, np.float64) * (ratio ** (1.0 / exponent) - 1.0))
