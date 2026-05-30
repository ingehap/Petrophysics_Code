"""
Master test runner for the Petrophysics August 2022 (Vol. 63, No. 4) issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article3_electric_dipole_sensitivity.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Gas Condensate PVT from FPG",            "article1_gas_condensate_fpg"),
    ("Article 2 - DL 2.5D LWD Resistivity Inversion",      "article2_lwd_dl_inversion"),
    ("Article 3 - Electric Dipole Bed-Detection",          "article3_electric_dipole_sensitivity"),
    ("Article 4 - Bayesian DB Log Interpretation",         "article4_bayesian_log_db"),
    ("Article 5 - CPA + FCM Log-Facies Analysis",          "article5_cpa_fcm_logfacies"),
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
