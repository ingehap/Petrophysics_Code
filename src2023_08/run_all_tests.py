"""
run_all_tests.py
=================
Run the test_all() suites of every Petrophysics-2023-08 article module.

Modules are imported individually so that a failure in one does not stop
the others.  Exits with a non-zero status if any module fails its tests.
"""

from __future__ import annotations

import importlib
import sys
import traceback


MODULES = [
    ("article1_nuclear_logging",        "Fitz: Casedhole Nuclear Logging"),
    ("article2_invasion_simulation",    "Merletti et al. (1): Mud-Filtrate Invasion"),
    ("article3_mineralogical_inversion","Jacomo et al.: Mineralogical Modelling"),
    ("article4_obm_imager_inversion",   "Chen et al.: HD OBM Imager Inversion"),
    ("article5_iterative_resistivity",  "Merletti et al. (2): Iterative Rt Inversion"),
    ("article6_well_log_qc",            "Jin et al.: Python Dash Well-Log QC"),
]


def main():
    failed = []
    for mod_name, title in MODULES:
        print("=" * 72)
        print(f"  {title}   [{mod_name}.py]")
        print("=" * 72)
        try:
            mod = importlib.import_module(mod_name)
            mod.test_all(verbose=True)
        except Exception:
            traceback.print_exc()
            failed.append(mod_name)
        print()
    print("=" * 72)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("ALL ARTICLE MODULES PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
