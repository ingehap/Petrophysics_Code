#!/usr/bin/env python3
"""
test_all.py - Master test runner for all Petrophysics Vol. 66, No. 1 (Feb 2025) modules.

Runs the test() function from each of the 12 article-implementation modules
and reports pass/fail status.

Reference: Petrophysics, Vol. 66, No. 1, February 2025
"Best Papers of the 2023 SCA International Symposium"

Usage:
    python test_all.py
"""

import sys
import importlib
import traceback
import time

# List of all 12 modules corresponding to the 12 articles
MODULES = [
    {
        "module": "ebeltoft_scal_model",
        "article": 1,
        "title": "SCAL Model for CO2 Storage (Ebeltoft et al.)",
        "doi": "10.30632/PJV66N1-2025a1",
        "pages": "7-21",
    },
    {
        "module": "mascle_co2_brine_kr",
        "article": 2,
        "title": "CO2/Brine Relative Permeability (Mascle et al.)",
        "doi": "10.30632/PJV66N1-2025a2",
        "pages": "22-38",
    },
    {
        "module": "richardson_scco2_brine_kr",
        "article": 3,
        "title": "Supercritical CO2/Brine Kr (Richardson et al.)",
        "doi": "10.30632/PJV66N1-2025a3",
        "pages": "39-51",
    },
    {
        "module": "jones_egr_co2",
        "article": 4,
        "title": "Enhanced Gas Recovery by CO2 (Jones et al.)",
        "doi": "10.30632/PJV66N1-2025a4",
        "pages": "52-63",
    },
    {
        "module": "mcclure_rev",
        "article": 5,
        "title": "REV for Two-Phase Flow (McClure et al.)",
        "doi": "10.30632/PJV66N1-2025a5",
        "pages": "64-79",
    },
    {
        "module": "regaieg_digital_rock",
        "article": 6,
        "title": "Digital Rock Physics for Kr (Regaieg et al.)",
        "doi": "10.30632/PJV66N1-2025a6",
        "pages": "80-94",
    },
    {
        "module": "fernandes_hybrid_drainage",
        "article": 7,
        "title": "Hybrid Drainage on Bimodal Limestone (Fernandes et al.)",
        "doi": "10.30632/PJV66N1-2025a7",
        "pages": "95-109",
    },
    {
        "module": "nono_primary_drainage",
        "article": 8,
        "title": "Primary Drainage on Non-Water-Wet Rocks (Nono et al.)",
        "doi": "10.30632/PJV66N1-2025a8",
        "pages": "110-122",
    },
    {
        "module": "pairoys_dopants",
        "article": 9,
        "title": "Impact of Dopants on SCAL (Pairoys et al.)",
        "doi": "10.30632/PJV66N1-2025a9",
        "pages": "123-133",
    },
    {
        "module": "wang_dual_porosity",
        "article": 10,
        "title": "Dual Matrix Porosity in Sandstone (Wang & Galley)",
        "doi": "10.30632/PJV66N1-2025a10",
        "pages": "134-154",
    },
    {
        "module": "ansaribaranghar_mr_saturation",
        "article": 11,
        "title": "Bulk Saturation via 13C and 1H MR (Ansaribaranghar et al.)",
        "doi": "10.30632/PJV66N1-2025a11",
        "pages": "155-168",
    },
    {
        "module": "ansaribaranghar_13c_mri",
        "article": 12,
        "title": "13C MRI Hydrocarbon Imaging (Ansaribaranghar et al.)",
        "doi": "10.30632/PJV66N1-2025a12",
        "pages": "169-182",
    },
]


def test_all(verbose=True):
    """
    Run all module tests and report results.

    Parameters
    ----------
    verbose : bool
        If True, print detailed output from each test.

    Returns
    -------
    dict
        Summary with keys: 'passed', 'failed', 'errors', 'total', 'details'
    """
    results = {
        "passed": 0,
        "failed": 0,
        "errors": [],
        "total": len(MODULES),
        "details": [],
    }

    separator = "=" * 78
    print(separator)
    print("  PETROPHYSICS Vol. 66, No. 1 (February 2025)")
    print("  Best Papers of the 2023 SCA International Symposium")
    print("  Master Test Runner - All 12 Modules")
    print(separator)
    print()

    for entry in MODULES:
        mod_name = entry["module"]
        article_num = entry["article"]
        title = entry["title"]
        doi = entry["doi"]
        pages = entry["pages"]

        print(f"--- Article {article_num:2d}: {title}")
        print(f"    Module: {mod_name}.py | Pages: {pages} | DOI: {doi}")

        t0 = time.time()
        try:
            # Import or reload the module
            if mod_name in sys.modules:
                mod = importlib.reload(sys.modules[mod_name])
            else:
                mod = importlib.import_module(mod_name)

            # Call the test function
            if hasattr(mod, "test"):
                if verbose:
                    print()
                mod.test()
                if verbose:
                    print()
            else:
                raise AttributeError(f"Module '{mod_name}' has no test() function")

            elapsed = time.time() - t0
            results["passed"] += 1
            status = "PASS"
            results["details"].append(
                {"module": mod_name, "status": "PASS", "time": elapsed, "error": None}
            )
            print(f"    Result: PASS ({elapsed:.2f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            results["failed"] += 1
            error_msg = f"{type(e).__name__}: {e}"
            results["errors"].append({"module": mod_name, "error": error_msg})
            results["details"].append(
                {
                    "module": mod_name,
                    "status": "FAIL",
                    "time": elapsed,
                    "error": error_msg,
                }
            )
            print(f"    Result: FAIL ({elapsed:.2f}s)")
            print(f"    Error:  {error_msg}")
            if verbose:
                traceback.print_exc()

        print()

    # Summary
    print(separator)
    print("  SUMMARY")
    print(separator)
    print(f"  Total modules:  {results['total']}")
    print(f"  Passed:         {results['passed']}")
    print(f"  Failed:         {results['failed']}")
    print()

    if results["errors"]:
        print("  Failed modules:")
        for err in results["errors"]:
            print(f"    - {err['module']}: {err['error']}")
        print()

    total_time = sum(d["time"] for d in results["details"])
    print(f"  Total test time: {total_time:.2f}s")

    if results["failed"] == 0:
        print()
        print("  ALL 12 MODULES PASSED SUCCESSFULLY!")
    else:
        print()
        print(f"  WARNING: {results['failed']} module(s) failed.")

    print(separator)

    return results


if __name__ == "__main__":
    # Add the directory containing the modules to the path
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    results = test_all(verbose="--quiet" not in sys.argv)
    sys.exit(0 if results["failed"] == 0 else 1)
