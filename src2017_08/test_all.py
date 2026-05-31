"""
Master test runner for the Petrophysics August 2017 (Vol. 58, No. 4) issue - six
articles: Bakken NMR relaxometry, wettability from T1/T2, centrifuge capillary
pressure, pressure-gradient measurement errors, contamination quantification,
and fast-neutron gamma density.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article3_centrifuge_capillary_pressure.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Bakken NMR Relaxometry",                 "article1_bakken_nmr_relaxometry"),
    ("Article 2 - Wettability from NMR T1/T2",             "article2_wettability_nmr_t1t2"),
    ("Article 3 - Centrifuge Capillary Pressure",          "article3_centrifuge_capillary_pressure"),
    ("Article 4 - Pressure-Gradient Measurement Errors",   "article4_pressure_gradient_errors"),
    ("Article 5 - Contamination Quantification",           "article5_contamination_quantification"),
    ("Article 6 - Fast-Neutron Gamma Density (proxy)",     "article6_fast_neutron_gamma_density"),
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
