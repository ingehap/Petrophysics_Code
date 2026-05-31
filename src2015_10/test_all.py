"""
Master test runner for the Petrophysics October 2015 (Vol. 56, No. 5) issue -
six articles: untangling acoustic anisotropy, reservoir fluid geodynamics
(differing equilibration times of GOR/asphaltenes/biomarkers), a robust Bakken
petrophysical model, EMAT downhole cement evaluation, integrated rock
classification in the McElroy Field, and a consistent evaluation approach to
thin-bedded sands in a Gulf of Mexico deepwater field.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_acoustic_anisotropy.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Untangling Acoustic Anisotropy",         "article1_acoustic_anisotropy"),
    ("Article 2 - Reservoir Fluid Geodynamics",            "article2_reservoir_fluid_geodynamics"),
    ("Article 3 - Bakken Petrophysical Model",             "article3_bakken_petrophysical_model"),
    ("Article 4 - EMAT Cement Evaluation",                 "article4_emat_cement_evaluation"),
    ("Article 5 - McElroy Rock Classification",            "article5_mcelroy_rock_classification"),
    ("Article 6 - Thin-Bedded Sands (Gulf of Mexico)",     "article6_thinbedded_sands_gom"),
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
