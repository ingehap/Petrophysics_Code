"""
Master test runner for the Petrophysics June 2015 (Vol. 56, No. 3) issue -
three articles plus a technical note: heavy-oil reservoir evaluation with NMR in
the Long Lake / Kinosis SAGD projects, real-time downhole fluid-sample
contamination prediction, an asphaltenes tutorial (Yen-Mullins / Flory-Huggins-
Zuo), and the Bateman-Konen resistivity-salinity transform.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article3_asphaltenes_explained.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Heavy-Oil NMR (SAGD)",                   "article1_heavyoil_nmr_sagd"),
    ("Article 2 - Fluid Contamination Prediction",         "article2_fluid_contamination_prediction"),
    ("Article 3 - Asphaltenes Explained",                  "article3_asphaltenes_explained"),
    ("Article 4 - Bateman-Konen (Technical Note)",         "article4_bateman_konen_transform"),
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
