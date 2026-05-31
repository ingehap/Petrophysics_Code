"""
Master test runner for the Petrophysics April 2016 (Vol. 57, No. 2) issue -
five articles: the Reservoir Producibility Index for tight-oil reservoir
quality, integrated petrofacies characterization of the Bakken shale, a new
approach to measuring organic (kerogen) density, a multilevel iterative method
for pore-confinement phase equilibrium, and acoustic anisotropy interpretation
in shales when the Stoneley-wave velocity is missing.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_reservoir_producibility_index.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Reservoir Producibility Index",          "article1_reservoir_producibility_index"),
    ("Article 2 - Bakken Shale Petrofacies",               "article2_bakken_petrofacies"),
    ("Article 3 - Measuring Organic Density",              "article3_organic_density"),
    ("Article 4 - Pore-Confinement Phase Equilibrium",     "article4_pore_confinement_phase_equilibrium"),
    ("Article 5 - Acoustic Anisotropy (No Stoneley)",      "article5_acoustic_anisotropy_no_stoneley"),
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
