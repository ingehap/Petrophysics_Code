"""
Master test runner for the Petrophysics June 2020 (Vol. 61, No. 3) issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article4_relperm_resistivity_fractal.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Casedhole FE Along Horizontal Wells",   "article1_casedhole_horizontal_fe"),
    ("Article 2 - Cement Quality Impact on C/O",          "article2_cement_quality_co_pulsed_neutron"),
    ("Article 3 - Relative Permeability (Tight Gas Sand)", "article3_relperm_tight_gas_sand"),
    ("Article 4 - Relative Permeability From Resistivity", "article4_relperm_resistivity_fractal"),
    ("Article 5 - Porosity From Drilling Parameters (ANN)", "article5_porosity_drilling_ann"),
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
