#!/usr/bin/env python3
"""Compare current runner stdout against the stored goldens.

Companion to tools/golden_capture.py: re-runs each issue directory's own
test runner under the deterministic subprocess environment, normalizes the
stdout the same way, and diffs it against ``tools/golden/<dir>.txt``.  Any
difference means printed behavior changed; the PR causing it must either
fix the regression or deliberately regenerate the golden with an
explanation.

Exit status: 0 when every compared directory matches; 1 on any mismatch,
missing golden, or failing runner.  Directories skipped for missing
optional dependencies are reported and ignored unless --require-all.

Usage:
    python tools/golden_diff.py                 # all directories
    python tools/golden_diff.py src2019_06 ...  # a subset
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden_capture  # noqa: E402
import run_all_issues as harness  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="*", help="issue directories (default: all)")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--require-all", action="store_true", help="treat SKIP (missing optional deps) as failure"
    )
    parser.add_argument(
        "--max-diff-lines", type=int, default=40, help="diff lines shown per mismatching directory"
    )
    args = parser.parse_args(argv)

    results = harness.run_all(args.dirs or None, args.jobs, args.timeout)
    n_match = n_mismatch = n_skip = 0
    for result in results:
        if result.status == "SKIP":
            n_skip += 1
            print(f"  skipped: {result.dirname} — {result.detail}")
            continue
        if result.status == "FAIL":
            n_mismatch += 1
            print(f"  RUNNER FAILED: {result.dirname} — {result.detail}")
            continue
        golden_file = golden_capture.golden_path(result.dirname)
        if not golden_file.exists():
            n_mismatch += 1
            print(
                f"  MISSING GOLDEN: {result.dirname} (run tools/golden_capture.py {result.dirname})"
            )
            continue
        expected = golden_file.read_text(encoding="utf-8")
        actual = golden_capture.normalize(result.stdout)
        if actual == expected:
            n_match += 1
            continue
        n_mismatch += 1
        print(f"\n===== STDOUT CHANGED: {result.dirname}")
        diff = difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=f"golden/{result.dirname}.txt",
            tofile=f"current/{result.dirname}",
        )
        for i, line in enumerate(diff):
            if i >= args.max_diff_lines:
                print("  ... (diff truncated)")
                break
            print(line, end="")

    print(
        f"\n  {n_match} matched, {n_mismatch} mismatched/failed, "
        f"{n_skip} skipped (of {len(results)} directories)"
    )
    if n_mismatch or (args.require_all and n_skip):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
