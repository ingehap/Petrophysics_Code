#!/usr/bin/env python3
"""
Petrophysics Vol. 66 No. 2 (April 2025) — Complete Test Suite
================================================================
Runs test_all() for every module implementing ideas from each article
in the SPWLA Petrophysics journal, April 2025 issue.

Usage:
    python test_all.py

Articles implemented:
  1. Cuadros et al.    — UDAR Look-Ahead-While-Drilling
  2. Sviridov et al.   — RJMCMC Stochastic Inversion for UDAR
  3. Jiang et al.      — Improved GIP for Shale Porosity
  4. Cheng et al.      — CRA / Retort / NMR Porosity Comparison
  5. Chen et al.       — Ultrasonic Microscopy Pore Characterization
  6. Hu et al.         — Overpressure Genetic Analysis (Isotope Logging)
  7. Varignier et al.  — Neutron Porosity Sensitivity Functions
  8. Yang et al.       — Filter Cake Effect on Cement Zonal Isolation
  9. Machicote et al.  — Microannuli Leak Rate Quantification
"""

import sys
import time
import traceback


def run_all_tests():
    """Import and run test_all() for every module."""
    modules = [
        ("udar_look_ahead",                "Cuadros et al., pp. 190–211"),
        ("stochastic_inversion",            "Sviridov et al., pp. 212–236"),
        ("gip_porosity",                    "Jiang et al., pp. 237–249"),
        ("unconventional_porosity",         "Cheng et al., pp. 250–266"),
        ("ultrasonic_pore_characterization", "Chen et al., pp. 267–282"),
        ("overpressure_isotope",            "Hu et al., pp. 283–293"),
        ("neutron_porosity_sensitivity",    "Varignier et al., pp. 294–317"),
        ("filter_cake_isolation",           "Yang et al., pp. 318–330"),
        ("microannuli_leak_rate",           "Machicote et al., pp. 331–347"),
    ]

    results = []
    t_start = time.time()

    print("╔" + "═" * 70 + "╗")
    print("║  Petrophysics Vol. 66, No. 2 (April 2025) — Test Suite".ljust(71) + "║")
    print("║  SPWLA Journal of Petrophysicists and Well Log Analysts".ljust(71) + "║")
    print("╚" + "═" * 70 + "╝")
    print()

    for i, (module_name, reference) in enumerate(modules, 1):
        print(f"[{i}/9] {module_name}  ({reference})")
        try:
            mod = __import__(module_name)
            mod.test_all()
            results.append((module_name, "PASS", None))
        except Exception as e:
            tb = traceback.format_exc()
            results.append((module_name, "FAIL", tb))
            print(f"  *** FAILED: {e}")
            print(tb)
            print()

    elapsed = time.time() - t_start

    # Summary
    print()
    print("╔" + "═" * 70 + "╗")
    print("║  TEST SUMMARY".ljust(71) + "║")
    print("╠" + "═" * 70 + "╣")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    for name, status, _ in results:
        icon = "✓" if status == "PASS" else "✗"
        line = f"║  {icon}  {name}".ljust(71) + "║"
        print(line)
    print("╠" + "═" * 70 + "╣")
    summary = f"║  {passed} passed, {failed} failed  ({elapsed:.1f}s)"
    print(summary.ljust(71) + "║")
    print("╚" + "═" * 70 + "╝")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
