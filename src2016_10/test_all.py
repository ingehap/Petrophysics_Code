"""
Master test runner for the Petrophysics October 2016 (Vol. 57, No. 5) issue -
five articles: electromagnetic look-ahead-while-drilling resistivity, pore-scale
drainage/imbibition water-saturation models in tight gas, first-order error
propagation, advanced dielectric/CRIM log interpretation, and microfracturing
for in-situ stress.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article3_foep_error_propagation.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - EMLA Look-Ahead Resistivity",            "article1_emla_lookahead_resistivity"),
    ("Article 2 - Tight-Gas Drainage/Imbibition Sw",       "article2_tightgas_saturation_height"),
    ("Article 3 - First-Order Error Propagation (FOEP)",   "article3_foep_error_propagation"),
    ("Article 4 - Dielectric/CRIM Archie (Green River)",   "article4_dielectric_archie_greenriver"),
    ("Article 5 - Microfracturing In-Situ Stress",         "article5_microfracturing_insitu_stress"),
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
