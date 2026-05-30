"""
Master test runner for the Petrophysics October 2020 (Vol. 61, No. 5) issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article4_archie_carbonate_consistent.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Nanoindentation of Shale Cuttings",     "article1_nanoindentation_shale"),
    ("Article 2 - Adsorption Isotherm Classification",    "article2_adsorption_isotherm_classification"),
    ("Article 3 - Wellbore Cave-in Detection",            "article3_cavein_clustering_detection"),
    ("Article 4 - Consistent Archie for Carbonates",      "article4_archie_carbonate_consistent"),
    ("Article 5 - Wettability & Water Blockage",          "article5_wettability_water_blockage"),
    ("Article 6 - Sonic Transit Times (ANN)",             "article6_sonic_transit_drilling_nn"),
    ("Article 7 - Multiphysics Rock Classification",      "article7_multiphysics_rock_classification"),
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
