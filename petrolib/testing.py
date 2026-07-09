"""Regression-safety helpers for the library migration.

``assert_matches_original`` is the shadow-equivalence gate of
LIBRARY_MERGE_PLAN.md section 8: before an article module's local function
body is replaced by a petrolib delegation, the PR must show that the old
body and the new call agree to ``rtol=1e-12`` on the article's own inputs
(plus draws across the physically meaningful ranges).  The documented
fallback ``rtol=1e-9`` is reserved for float reassociation introduced by
vectorizing a scalar loop — and must be called out in the PR.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

#: A call case: positional args, or (positional args, keyword args).
Case = Sequence[Any] | tuple[Sequence[Any], Mapping[str, Any]]


def _split(case: Case) -> tuple[Sequence[Any], Mapping[str, Any]]:
    if (
        isinstance(case, tuple)
        and len(case) == 2
        and isinstance(case[1], Mapping)
        and isinstance(case[0], (tuple, list))
    ):
        return case[0], case[1]
    return case, {}


def _assert_equal(old: Any, new: Any, rtol: float, atol: float, where: str) -> None:
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        assert old.keys() == new.keys(), f"{where}: keys differ: {old.keys()} != {new.keys()}"
        for key in old:
            _assert_equal(old[key], new[key], rtol, atol, f"{where}[{key!r}]")
        return
    if isinstance(old, (tuple, list)) and isinstance(new, (tuple, list)):
        assert len(old) == len(new), f"{where}: lengths differ: {len(old)} != {len(new)}"
        for index, (o, n) in enumerate(zip(old, new, strict=True)):
            _assert_equal(o, n, rtol, atol, f"{where}[{index}]")
        return
    if isinstance(old, (str, bytes, bool)) or old is None:
        assert old == new, f"{where}: {old!r} != {new!r}"
        return
    np.testing.assert_allclose(
        np.asarray(new),
        np.asarray(old),
        rtol=rtol,
        atol=atol,
        err_msg=f"{where}: replacement disagrees with original",
    )


def assert_matches_original(
    original: Callable[..., Any],
    replacement: Callable[..., Any],
    cases: Iterable[Case],
    *,
    rtol: float = 1e-12,
    atol: float = 0.0,
) -> int:
    """Assert ``replacement(*case)`` matches ``original(*case)`` for every case.

    Results are compared with ``np.testing.assert_allclose`` at the given
    tolerances, recursing through tuples/lists/dicts; strings, bools and
    ``None`` must be exactly equal.  Returns the number of cases checked
    (and asserts it is nonzero — an empty case list would vacuously pass).
    """
    count = 0
    for case in cases:
        args, kwargs = _split(case)
        old = original(*args, **kwargs)
        new = replacement(*args, **kwargs)
        _assert_equal(old, new, rtol, atol, where=f"case {count} {tuple(args)!r}")
        count += 1
    assert count > 0, "assert_matches_original received no cases"
    return count
