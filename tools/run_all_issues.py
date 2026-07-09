#!/usr/bin/env python3
"""Run every issue directory's own test runner and report a per-directory verdict.

Phase 0 safety net of the library migration (see LIBRARY_MERGE_PLAN.md,
section 7): before any code moves into a shared library, this harness pins
the current behavior of all 75 ``srcYYYY_MM`` directories by running each
directory's own runner (``test_all.py`` or its historical variants) in a
subprocess with ``cwd`` set to that directory — exactly the way a reader
runs them.

The runners come from three generations of conventions and do not agree on
exit codes (most of the 2014-2022 era always exit 0), so a directory's
verdict combines the exit code with a parse of the runner's own summary
line.  An unrecognized summary is a FAILURE by design: every runner format
must be known to this harness, so that a silent format change can never be
mistaken for a pass.

Directories whose modules import optional heavy dependencies (torch,
scikit-learn, xgboost, scikit-image, scipy) are SKIPPED when the package is
not installed; ``--require-all`` turns those skips into failures for the
full CI lane.

Usage:
    python tools/run_all_issues.py                 # all 75 directories
    python tools/run_all_issues.py src2019_06 ...  # a subset
    python tools/run_all_issues.py --jobs 4 --require-all
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories whose master runner is not named test_all.py.
RUNNER_OVERRIDES = {
    "src2023_08": "run_all_tests.py",
    "src2023_10": "run_all_tests.py",
    "src2023_12": "run_all.py",
    "src2024_02": "run_all_tests.py",
    "src2024_12": "test_all_modules.py",
    "src2025_06": "run_all_tests.py",
}

# Third-party packages that only some directories need.  numpy is required
# everywhere and is deliberately not listed: without numpy nothing works and
# every directory failing loudly is the correct outcome.
OPTIONAL_PACKAGES = ("scipy", "sklearn", "skimage", "xgboost", "torch")

_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(" + "|".join(OPTIONAL_PACKAGES) + r")\b", re.M
)
_TEST_ALL_RE = re.compile(r"^def test_all\b", re.M)

# Expected repository invariants, asserted by tests/test_articles.py.
# Update these deliberately (with a note in the PR) when issues are added.
EXPECTED_DIR_COUNT = 75
EXPECTED_TEST_ALL_MODULES = 501

# One (pattern, decide) rule per known summary format.  ``decide`` maps the
# match to True (pass) or False (fail).  Rules are tried in order; the LAST
# occurrence of the first matching pattern decides, so a runner's final
# summary always wins over any module output that happens to look similar.
_SUMMARY_RULES = [
    # 2014-2022 era test_all.py, plus src2023_12, src2024_02, src2025_06,
    # src2025_12: "  5/5 modules passed", "== 8 / 8 modules passed ==",
    # "0/13 modules passed,  13 FAILED"
    (re.compile(r"(\d+)\s*/\s*(\d+) modules passed"),
     lambda m: m.group(1) == m.group(2)),
    # src2023_08
    (re.compile(r"ALL ARTICLE MODULES PASSED"), lambda m: True),
    (re.compile(r"^FAILED: \[", re.M), lambda m: False),
    # src2024_10: "  ALL 123 CHECKS PASSED" / "  4 ISSUES DETECTED"
    (re.compile(r"ALL (\d+) CHECKS PASSED"), lambda m: True),
    (re.compile(r"(\d+) ISSUES DETECTED"), lambda m: False),
    # src2024_04: "All 6 article modules passed." (exceptions propagate:
    # any module failure exits nonzero before this line prints)
    (re.compile(r"All (\d+) article modules passed\."), lambda m: True),
    # src2024_06: "All 8 modules passed." / "2 of 8 modules FAILED."
    (re.compile(r"All (\d+) modules passed\."), lambda m: True),
    (re.compile(r"(\d+) of (\d+) modules FAILED"), lambda m: False),
    # src2024_08: "| Passed: 14/14 | Failed: 0/14 |"
    (re.compile(r"Passed:\s*(\d+)/(\d+)\s*\|+\s*Failed:\s*(\d+)/(\d+)"),
     lambda m: m.group(3) == "0" and m.group(1) == m.group(2)),
    # src2026_04 / src2026_06: "  RESULTS:  71/71 passed  |  0 failed"
    (re.compile(r"RESULTS:\s+(\d+)/(\d+) passed\s+\|\s+(\d+) failed"),
     lambda m: m.group(3) == "0" and m.group(1) == m.group(2)),
    # src2025_08: "All 11 modules passed successfully!"
    (re.compile(r"All (\d+) modules passed successfully!"), lambda m: True),
    # Generic "<n> passed, <m> failed" summary, shared by src2023_10
    # ("10 passed, 0 failed (of 10 modules)"), src2024_12 ("Total: 13
    # passed, 0 failed, 1.2s elapsed"), src2025_02/src2025_10 ("RESULTS:
    # 103 passed, 0 failed"), src2025_04 ("9 passed, 0 failed  (92.2s)")
    # and src2026_02 ("RESULTS: 89 passed, 0 failed out of 89 tests").
    (re.compile(r"(\d+) passed,\s*(\d+) failed"),
     lambda m: m.group(2) == "0"),
]


@dataclass
class Result:
    dirname: str
    status: str          # "PASS" | "FAIL" | "SKIP"
    detail: str
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0


def issue_dir_names() -> list[str]:
    return sorted(p.name for p in REPO_ROOT.glob("src20*") if p.is_dir())


def runner_for(dirname: str) -> str:
    return RUNNER_OVERRIDES.get(dirname, "test_all.py")


def required_optional_packages(dirname: str) -> set[str]:
    """Optional third-party packages imported anywhere in the directory."""
    needed: set[str] = set()
    for py in (REPO_ROOT / dirname).glob("*.py"):
        needed.update(_IMPORT_RE.findall(py.read_text(encoding="utf-8")))
    return needed


def missing_packages(dirname: str) -> set[str]:
    return {
        pkg for pkg in required_optional_packages(dirname)
        if importlib.util.find_spec(pkg) is None
    }


def count_test_all_modules() -> int:
    """Article modules (runners and __init__.py excluded) defining test_all()."""
    total = 0
    for dirname in issue_dir_names():
        runner = runner_for(dirname)
        for py in (REPO_ROOT / dirname).glob("*.py"):
            if py.name in (runner, "__init__.py"):
                continue
            if _TEST_ALL_RE.search(py.read_text(encoding="utf-8")):
                total += 1
    return total


def subprocess_env() -> dict[str, str]:
    """Deterministic environment for runner subprocesses.

    Hash randomization changes set-iteration order (and therefore printed
    output); BLAS thread counts can change float reduction order.  Pinning
    both keeps runs reproducible so stdout can serve as a golden baseline.
    """
    env = os.environ.copy()
    env.update(
        PYTHONHASHSEED="0",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    return env


def _verdict(returncode: int, stdout: str) -> tuple[bool, str]:
    if returncode != 0:
        return False, f"runner exited with code {returncode}"
    for pattern, decide in _SUMMARY_RULES:
        match = None
        for match in pattern.finditer(stdout):
            pass
        if match:
            ok = decide(match)
            return ok, f"summary: {match.group(0).strip()!r}"
    return False, "unrecognized summary format (harness must be taught this runner)"


def run_directory(dirname: str, timeout: float = 900.0) -> Result:
    """Run one directory's own runner; never raises."""
    directory = REPO_ROOT / dirname
    runner = runner_for(dirname)
    if not (directory / runner).exists():
        return Result(dirname, "FAIL", f"runner {runner} not found")
    missing = missing_packages(dirname)
    if missing:
        return Result(dirname, "SKIP",
                      f"optional dependencies not installed: {sorted(missing)}")
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, runner],
            cwd=directory,
            env=subprocess_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"")
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        return Result(dirname, "FAIL", f"timed out after {timeout:.0f}s",
                      None, stdout, "", time.time() - start)
    ok, detail = _verdict(proc.returncode, proc.stdout)
    return Result(dirname, "PASS" if ok else "FAIL", detail,
                  proc.returncode, proc.stdout, proc.stderr,
                  time.time() - start)


def run_all(dirnames: list[str] | None = None, jobs: int = 0,
            timeout: float = 900.0) -> list[Result]:
    dirnames = dirnames or issue_dir_names()
    jobs = jobs or min(8, os.cpu_count() or 1)
    results: dict[str, Result] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run_directory, d, timeout): d for d in dirnames}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results[result.dirname] = result
            marker = {"PASS": " ", "FAIL": "!", "SKIP": "-"}[result.status]
            print(f"  [{marker}] {result.dirname:12s} {result.status:4s} "
                  f"({result.duration:5.1f}s)  {result.detail}", flush=True)
    return [results[d] for d in dirnames]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="*", help="issue directories (default: all)")
    parser.add_argument("--jobs", type=int, default=0,
                        help="parallel runners (default: min(8, cpus))")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="per-directory timeout in seconds")
    parser.add_argument("--require-all", action="store_true",
                        help="treat SKIP (missing optional deps) as failure")
    args = parser.parse_args(argv)

    for dirname in args.dirs:
        if dirname not in issue_dir_names():
            parser.error(f"unknown issue directory: {dirname}")

    results = run_all(args.dirs or None, args.jobs, args.timeout)

    n_pass = sum(r.status == "PASS" for r in results)
    n_fail = sum(r.status == "FAIL" for r in results)
    n_skip = sum(r.status == "SKIP" for r in results)
    print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped "
          f"(of {len(results)} directories)")
    for r in results:
        if r.status == "FAIL":
            print(f"\n===== {r.dirname}: {r.detail}")
            tail = "\n".join(r.stdout.splitlines()[-15:])
            if tail:
                print(tail)
            err_tail = "\n".join(r.stderr.splitlines()[-15:])
            if err_tail:
                print("--- stderr ---")
                print(err_tail)
    if n_fail or (args.require_all and n_skip):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
