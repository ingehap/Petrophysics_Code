"""
Master test runner for the Petrophysics June 2014 (Vol. 55, No. 3) issue -
a formation-evaluation and rock-physics case study of the Bazhenov shale,
borehole-carbon corrections for accurate TOC from nuclear spectroscopy,
magnetic-resonance core-plug analysis with a three-magnet array unilateral
magnet, a method for predicting permeability of complex carbonate reservoirs
from NMR logging, and a shale-line analysis for shaly-sand porosity computation
and sedimentary interpretation in deepwater sediments.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_bazhenov_rock_physics.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Bazhenov Shale Rock Physics",            "article1_bazhenov_rock_physics"),
    ("Article 2 - Borehole Carbon Corrections for TOC",    "article2_borehole_carbon_toc"),
    ("Article 3 - Three-Magnet Array Unilateral MR",       "article3_three_magnet_mr_coreplug"),
    ("Article 4 - NMR Permeability of Carbonates",         "article4_nmr_carbonate_permeability"),
    ("Article 5 - Shaly-Sand Porosity (Shale Line)",       "article5_shaly_sand_porosity_deepwater"),
]


def run_all():
    results = {}
    for title, mod_name in MODULES:
        print()
        print("#" * 70)
        print(f"# {title}")
        print("#" * 70)
        try:
            t0 = time.time()
            mod = importlib.import_module(mod_name)
            r = mod.test_all()
            dt = time.time() - t0
            results[mod_name] = {"status": "PASS", "time": dt, "result": r}
            print(f"  -> {dt:.2f}s")
        except Exception as e:
            traceback.print_exc()
            results[mod_name] = {"status": "FAIL", "error": str(e)}
    print()
    print("=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        status = r["status"]
        extra = f"  ({r['time']:.2f}s)" if status == "PASS" else f"  ({r.get('error', '')})"
        print(f"  {status:5s}  {name}{extra}")
    n_pass = sum(1 for r in results.values() if r["status"] == "PASS")
    print(f"\n  {n_pass}/{len(MODULES)} modules passed")
    return results


if __name__ == "__main__":
    run_all()
