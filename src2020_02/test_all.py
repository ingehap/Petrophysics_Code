"""
Master test runner for the Petrophysics February 2020 (Vol. 61, No. 1) issue -
an invited tutorial on borehole-nuclear Monte Carlo modeling plus five regular
articles.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_montecarlo_nuclear_fsf_tutorial.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Monte Carlo Nuclear & FSF (Tutorial)",  "article1_montecarlo_nuclear_fsf_tutorial"),
    ("Article 2 - Russian vs Western Resistivity Logs",   "article2_russian_western_resistivity"),
    ("Article 3 - HDIL Array Induction (Anisotropic)",    "article3_hdil_array_induction_anisotropic"),
    ("Article 4 - Physics-Driven Deep-Learning Inversion", "article4_physics_deeplearning_inversion"),
    ("Article 5 - Bayesian Geosteering via SMC",          "article5_bayesian_geosteering_smc"),
    ("Article 6 - Boomerang Porosity & Net/Gross",        "article6_boomerang_porosity_netgross"),
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
