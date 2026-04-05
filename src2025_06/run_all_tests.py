#!/usr/bin/env python3
"""
Petrophysics Vol. 66, No. 3 (June 2025) – Module Test Suite
=============================================================

Master test runner for all seven article implementations from:
  SPWLA Petrophysics Journal, Vol. 66, No. 3, June 2025

Articles and corresponding modules:
  1. core_scanner.py              – Mirza et al.   (pp. 352–363)
  2. thomas_stieber_tyurin.py     – Tyurin & Davenport (pp. 365–391)
  3. thomas_stieber_welllog.py    – Eghbali & Torres-Verdín (pp. 392–423)
  4. toc_prediction.py            – Dong et al.    (pp. 425–448)
  5. cross_calibrated_permeability.py – Sifontes et al. (pp. 449–466)
  6. shale_microparams.py         – Jiang et al.   (pp. 468–488)
  7. fracturing_fluid_damage.py   – Li et al.      (pp. 489–520)
  8. injection_fluid_optimization.py – Xiao et al. (pp. 521–535)

Usage:
  python run_all_tests.py
"""

import sys
import traceback
import time


def run_module_test(module_name: str, label: str) -> bool:
    """Import a module and run its test_all() function."""
    print("\n" + "█" * 70)
    print(f"  {label}")
    print("█" * 70)
    try:
        mod = __import__(module_name)
        mod.test_all()
        return True
    except Exception as e:
        print(f"\n✗ FAILED: {module_name}")
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("  PETROPHYSICS Vol. 66, No. 3 (June 2025)")
    print("  Complete Module Test Suite")
    print("=" * 70)

    modules = [
        ("core_scanner",
         "Article 1: Core Scanner – Mirza et al."),
        ("thomas_stieber_tyurin",
         "Article 2: Thomas-Stieber-Tyurin Model – Tyurin & Davenport"),
        ("thomas_stieber_welllog",
         "Article 3: T-S Diagram in Well-Log Domain – Eghbali & Torres-Verdín"),
        ("toc_prediction",
         "Article 4: TOC Prediction with ML – Dong et al."),
        ("cross_calibrated_permeability",
         "Article 5: Cross-Calibrated Permeabilities – Sifontes et al."),
        ("shale_microparams",
         "Article 6: Shale Micro-Parameter Calibration – Jiang et al."),
        ("fracturing_fluid_damage",
         "Article 7: Fracturing Fluid Damage – Li et al."),
        ("injection_fluid_optimization",
         "Article 8: Injection Fluid Optimization – Xiao et al."),
    ]

    t0 = time.time()
    results = []
    for mod_name, label in modules:
        ok = run_module_test(mod_name, label)
        results.append((label, ok))

    elapsed = time.time() - t0

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    n_pass = 0
    for label, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {label}")
        if ok:
            n_pass += 1

    print(f"\n  {n_pass}/{len(results)} modules passed  "
          f"({elapsed:.1f}s total)")
    print("=" * 70)

    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
