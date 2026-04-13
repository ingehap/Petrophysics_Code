"""
run_all_tests.py
================
Executes ``test_all()`` in every article_*.py module in this directory
and prints a pass/fail summary.

Usage:  python run_all_tests.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    sys.path.insert(0, str(HERE))
    modules = sorted(p.stem for p in HERE.glob("article_*.py"))
    passed, failed = 0, 0
    print("=" * 78)
    for name in modules:
        try:
            mod = importlib.import_module(name)
            mod.test_all()
            passed += 1
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print("=" * 78)
    print(f"{passed} passed, {failed} failed (of {len(modules)} modules)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
