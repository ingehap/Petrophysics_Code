"""
test_all.py
===========

Top-level runner that imports every article module in this directory and
executes its ``test_all`` function.  Each article module is itself a
standalone script, so running them individually is equivalent.

Usage
-----

    python test_all.py            # run every module's tests, verbose
    python test_all.py --quiet    # run every module's tests, quiet

The modules implemented are:

    article1_nuclear_logging_ccs       Badruzzaman (2024)
    article2_claystone_repository      Strobel (2024)
    article3_hydrogen_storage          Okoroafor et al. (2024)
    article4_facies_classification     Morelli et al. (2024)
    article5_lwd_image_deeplearning    Molossi et al. (2024)
    article6_nmr_t1t2_saturation       Althaus et al. (2024)
    article7_shale_fracture_damage     Jiang et al. (2024)
    article8_r35_fractal_rock_typing   Duan et al. (2024)

All are from Petrophysics Vol. 65 No. 3 (June 2024).
"""

from __future__ import annotations

import importlib
import sys
import traceback

MODULES = [
    "article1_nuclear_logging_ccs",
    "article2_claystone_repository",
    "article3_hydrogen_storage",
    "article4_facies_classification",
    "article5_lwd_image_deeplearning",
    "article6_nmr_t1t2_saturation",
    "article7_shale_fracture_damage",
    "article8_r35_fractal_rock_typing",
]


def test_all(verbose: bool = True) -> int:
    """Run every module's test_all.  Returns number of failures."""
    failures = 0
    for name in MODULES:
        if verbose:
            print("=" * 72)
            print(f"Running {name}.test_all()")
            print("-" * 72)
        try:
            mod = importlib.import_module(name)
            mod.test_all(verbose=verbose)
        except Exception:
            failures += 1
            print(f"*** {name} FAILED ***")
            traceback.print_exc()
    print("=" * 72)
    if failures == 0:
        print(f"All {len(MODULES)} modules passed.")
    else:
        print(f"{failures} of {len(MODULES)} modules FAILED.")
    return failures


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv or "-q" in sys.argv
    n = test_all(verbose=not quiet)
    sys.exit(0 if n == 0 else 1)
