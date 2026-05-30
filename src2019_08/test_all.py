"""
Master test runner for the Petrophysics August 2019 (Vol. 60, No. 4) issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article3_shale_gas_z_factor.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - ML Well-Log Correlation",               "article1_ml_well_log_correlation"),
    ("Article 2 - TOC From XRF (Neural Network)",         "article2_toc_xrf_neural_network"),
    ("Article 3 - Shale-Gas Compressibility Factor Z",    "article3_shale_gas_z_factor"),
    ("Article 4 - NMR Wettability Index",                 "article4_nmr_wettability_index"),
    ("Article 5 - Aging Effect on Relative Permeability", "article5_relperm_aging_highthroughput"),
    ("Article 6 - Smart Water-CO2 Flooding",              "article6_smart_water_co2_flooding"),
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
