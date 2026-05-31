"""
Master test runner for the Petrophysics February 2016 (Vol. 57, No. 1) issue -
the Best Papers of the 2015 SCA Symposium plus one regular submission: CO2-brine
multiphase flow in sandstone, estimating saturations in organic shales with 2D
NMR, low-permeability measurement insights, low-salinity waterflooding, and
graphical solutions for laminated and dispersed shaly sands.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_co2_brine_multiphase_flow.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - CO2-Brine Multiphase Flow",              "article1_co2_brine_multiphase_flow"),
    ("Article 2 - 2D NMR Shale Saturations",               "article2_2d_nmr_shale_saturations"),
    ("Article 3 - Low-Permeability Measurements",          "article3_low_permeability_measurements"),
    ("Article 4 - Low-Salinity Waterflooding",             "article4_low_salinity_waterflooding"),
    ("Article 5 - Shaly-Sand Graphical Solutions",         "article5_shaly_sand_graphical_solutions"),
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
