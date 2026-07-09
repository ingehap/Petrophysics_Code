#!/usr/bin/env python3
"""Capture normalized golden stdout for every issue directory's test runner.

Phase 0 safety net of the library migration (see LIBRARY_MERGE_PLAN.md,
section 8): the ``test_all()`` assertions only pin values they assert, at
tolerances of 1e-6..1e-3.  Everything the runners *print* — intermediate
values, fitted coefficients, classifications — is behavior worth freezing
too, because a migration bug can shift a printed value without tripping an
assert.  This tool runs each directory's runner (via the same subprocess
harness as tools/run_all_issues.py, with hash seed and BLAS threads pinned)
and stores its stdout, normalized for the only legitimately nondeterministic
content (wall-clock timings), under ``tools/golden/<dir>.txt``.

``tools/golden_diff.py`` re-runs and compares against these files; a diff
means printed behavior changed and must be explained in the PR that causes
it (regenerating the golden in the same PR, with a CHANGELOG note, when the
change is deliberate).

Usage:
    python tools/golden_capture.py                 # all directories
    python tools/golden_capture.py src2019_06 ...  # a subset
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all_issues as harness  # noqa: E402

GOLDEN_DIR = harness.REPO_ROOT / "tools" / "golden"

# Wall-clock timings are the only accepted nondeterminism in runner output.
# Each pattern replaces a timing token with a fixed-width placeholder; the
# surrounding text is preserved so the goldens stay readable.
_NORMALIZATIONS = [
    # "-> 0.32s", "( 0.32s)", "(0.32s)", "PASS  0.05s", "0.1s total/elapsed"
    (re.compile(r"\b\d+\.\d+s\b"), "#.##s"),
    # "in 12.3 seconds", "( 0.4 s total)"
    (re.compile(r"\b\d+\.\d+\s+s(econds)?\b"), "#.## s"),
    # src2023_06's runner prints a per-module wall-clock column:
    # "article1_hdt          OK          0.062"
    (re.compile(r"^(\S+\s+(?:OK|FAIL)\s+)\d+\.\d+\s*$", re.M), r"\1#.###"),
    # src2023_12's GAN demo prints stochastic torch training losses
    # (unseeded, thread-order dependent); everything else it prints is pinned
    (re.compile(r"losses g=-?\d+\.\d+ d=-?\d+\.\d+"), "losses g=#.### d=#.###"),
    # defensive: object reprs leaking addresses
    (re.compile(r"0x[0-9a-fA-F]{6,}"), "0x######"),
]


def normalize(text: str) -> str:
    for pattern, replacement in _NORMALIZATIONS:
        text = pattern.sub(replacement, text)
    if not text.endswith("\n"):
        text += "\n"
    return text


def golden_path(dirname: str) -> Path:
    return GOLDEN_DIR / f"{dirname}.txt"


def capture(dirnames: list[str] | None = None, jobs: int = 0, timeout: float = 900.0) -> int:
    results = harness.run_all(dirnames, jobs, timeout)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    n_written = n_failed = n_skipped = 0
    for result in results:
        if result.status == "SKIP":
            n_skipped += 1
            print(f"  not captured (skipped): {result.dirname} — {result.detail}")
            continue
        if result.status == "FAIL":
            n_failed += 1
            print(f"  NOT captured (failing): {result.dirname} — {result.detail}")
            continue
        golden_path(result.dirname).write_text(normalize(result.stdout), encoding="utf-8")
        n_written += 1
    print(
        f"\n  {n_written} goldens written to {GOLDEN_DIR}, {n_failed} failing, {n_skipped} skipped"
    )
    return 1 if n_failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="*", help="issue directories (default: all)")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    return capture(args.dirs or None, args.jobs, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
