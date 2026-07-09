"""Pytest wrapper around the 75-directory issue harness.

Phase 0 safety net of the library migration (see LIBRARY_MERGE_PLAN.md,
sections 7-8).  Each issue directory is one parametrized test that runs the
directory's own runner (``test_all.py`` or its historical variants) in a
subprocess with ``cwd`` set to that directory — via the same code as
``python tools/run_all_issues.py``.  Directories needing optional heavy
dependencies (torch, scikit-learn, xgboost, scikit-image, scipy) are
skipped when the package is not installed.

Two invariants are pinned alongside the per-directory runs, so that a
migration step can never silently shrink the regression oracle:

* the repository contains exactly the expected number of issue directories,
  each with a runner the harness recognizes;
* exactly the expected number of article modules define a module-level
  ``test_all()`` (the 2014-2023 era convention; the seven package-style
  directories test centrally in their runner instead).

Run:
    pytest tests/test_articles.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "run_all_issues", REPO_ROOT / "tools" / "run_all_issues.py")
harness = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("run_all_issues", harness)
_spec.loader.exec_module(harness)

DIRS = harness.issue_dir_names()


@pytest.mark.parametrize("dirname", DIRS)
def test_issue_directory(dirname):
    missing = harness.missing_packages(dirname)
    if missing:
        pytest.skip(f"optional dependencies not installed: {sorted(missing)}")
    result = harness.run_directory(dirname)
    assert result.status == "PASS", (
        f"{dirname}: {result.detail}\n"
        f"--- stdout tail ---\n{result.stdout[-2000:]}\n"
        f"--- stderr tail ---\n{result.stderr[-2000:]}"
    )


def test_issue_directory_count():
    assert len(DIRS) == harness.EXPECTED_DIR_COUNT, (
        "issue directory added or removed: update EXPECTED_DIR_COUNT in "
        "tools/run_all_issues.py (deliberately, with a note in the PR) and "
        "capture a golden for any new directory"
    )


def test_every_directory_has_a_recognized_runner():
    missing = [d for d in DIRS
               if not (REPO_ROOT / d / harness.runner_for(d)).exists()]
    assert not missing, (
        f"directories without their expected runner: {missing} — teach "
        "RUNNER_OVERRIDES in tools/run_all_issues.py about them"
    )


def test_test_all_discovery_count():
    count = harness.count_test_all_modules()
    assert count == harness.EXPECTED_TEST_ALL_MODULES, (
        f"module-level test_all() count changed "
        f"({count} != {harness.EXPECTED_TEST_ALL_MODULES}): the regression "
        "oracle grew or shrank — update EXPECTED_TEST_ALL_MODULES in "
        "tools/run_all_issues.py deliberately, never as a side effect"
    )
