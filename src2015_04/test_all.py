"""
Master test runner for the Petrophysics April 2015 (Vol. 56, No. 2) issue -
five articles: automatically quantifying wireline/LWD pressure-test quality,
steady-state stress-dependent permeability of tight oil rocks, permeability
estimation in the McMurray formation from high-resolution data, microresistivity
curve extraction from borehole-microimager data, and a new method to estimate
porosity from NMR data with short relaxation times.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article5_nmr_short_t2_porosity.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Pressure-Test Quality",                  "article1_pressure_test_quality"),
    ("Article 2 - Stress-Dependent Permeability",          "article2_stress_dependent_permeability"),
    ("Article 3 - McMurray Permeability Upscaling",        "article3_mcmurray_permeability_upscaling"),
    ("Article 4 - Microresistivity Curve Extraction",      "article4_microresistivity_extraction"),
    ("Article 5 - NMR Short-T2 Porosity Correction",       "article5_nmr_short_t2_porosity"),
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
