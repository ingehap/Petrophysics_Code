"""Internal compatibility helpers (not part of the public API).

Centralizes the NumPy 1.x/2.x shims that the article modules copy-paste
today, and the opt-in deprecation warning used by permanent facades.
"""

from __future__ import annotations

import functools
import os
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np

_F = TypeVar("_F", bound=Callable[..., Any])

#: np.trapz was renamed np.trapezoid in NumPy 2.0; support both.
trapezoid: Callable[..., Any] = cast(
    "Callable[..., Any]", getattr(np, "trapezoid", getattr(np, "trapz", None))
)


def deprecated(reason: str) -> Callable[[_F], _F]:
    """Mark a function deprecated — warning only when explicitly enabled.

    Facades in the article directories are permanent (LIBRARY_MERGE_PLAN.md
    section 6) and must stay silent for readers running the scripts; the
    warning fires only when ``PETROLIB_WARN_DEPRECATED=1`` is set, so the
    maintainer can audit remaining call sites on demand.
    """

    def decorate(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if os.environ.get("PETROLIB_WARN_DEPRECATED") == "1":
                warnings.warn(
                    f"{func.__qualname__} is deprecated: {reason}",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate
