"""
Master test runner for the Petrophysics April 2022 (Vol. 63, No. 2) issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article2_kozeny_permeability_chalk.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - DEC Tool Bayesian-GPR Inversion",          "article1_dec_tool_bayesian_gpr"),
    ("Article 2 - Kozeny Permeability in Clay-Rich Chalk",   "article2_kozeny_permeability_chalk"),
    ("Article 3 - Pyrite-Aware Sw with HS Bounds",           "article3_pyrite_saturation_hs_bounds"),
    ("Article 4 - Micro-CT Filter-Cake Deposition (proxy)",  "article4_microct_filtercake"),
    ("Article 5 - Methane Solubility in OBM (proxy)",        "article5_methane_solubility_obm"),
    ("Article 6 - Gas-Hydrate Rock Physics (proxy)",         "article6_gas_hydrate_rock_physics"),
    ("Article 7 - Digital-Core Wellbore Stability (proxy)",  "article7_digital_core_wellbore_stability"),
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
