"""
Master test runner for the Petrophysics October 2022 (Vol. 63, No. 5) issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article3_stress_dependent_permeability.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Nanoindentation Guide",                      "article1_nanoindentation"),
    ("Article 2 - Shale Capillary Pressure Model",             "article2_shale_capillary_pressure"),
    ("Article 3 - Stress-Dependent Tight Permeability",        "article3_stress_dependent_permeability"),
    ("Article 4 - Mud-Filtrate Invasion (Micro-CT)",           "article4_mud_filtrate_invasion_ct"),
    ("Article 5 - Cement Corrosion (CO2 + H2S)",               "article5_cement_acid_gas_corrosion"),
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
