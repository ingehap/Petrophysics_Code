"""
Master test runner for the Petrophysics April 2014 (Vol. 55, No. 2) issue - the
Special Issue on Deepwater: deepwater exploration and production in the Gulf of
Mexico, the origin and characteristics of turbidite sediments, the dynamics of
reservoir fluids and their systematic variations, fault-block migrations inferred
from asphaltene gradients, formation-evaluation challenges and opportunities in
deepwater, and quantifying the effect of kerogen on resistivity measurements in
organic-rich mudrocks.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article3_reservoir_fluid_dynamics.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Deepwater GoM Overview",                 "article1_deepwater_gom_overview"),
    ("Article 2 - Turbidite Sediments",                    "article2_turbidite_sediments"),
    ("Article 3 - Dynamics of Reservoir Fluids",           "article3_reservoir_fluid_dynamics"),
    ("Article 4 - Asphaltene Fault-Block Migration",       "article4_asphaltene_fault_block_migration"),
    ("Article 5 - Deepwater Formation Evaluation",         "article5_deepwater_formation_evaluation"),
    ("Article 6 - Kerogen Effect on Resistivity",          "article6_kerogen_resistivity_mudrocks"),
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
