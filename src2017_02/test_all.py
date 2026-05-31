"""
Master test runner for the Petrophysics February 2017 (Vol. 58, No. 1) issue - a
multiphase-flow / SCAL special issue with five articles: flow regimes during
immiscible displacement, MICP relative-permeability effects, osmosis in
low-salinity waterflooding, micro-CT salinity fluid distribution, and a SCAL
simulator comparison.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article5_scal_simulator_comparison.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Flow Regimes (Immiscible Displacement)", "article1_flow_regimes_immiscible"),
    ("Article 2 - MICP Relative-Permeability Effects",     "article2_micp_relperm_transition"),
    ("Article 3 - Osmosis in Low-Salinity Waterflooding",  "article3_osmosis_low_salinity"),
    ("Article 4 - Micro-CT Salinity Fluid Distribution",   "article4_microct_salinity_distribution"),
    ("Article 5 - SCAL Simulator Comparison",              "article5_scal_simulator_comparison"),
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
