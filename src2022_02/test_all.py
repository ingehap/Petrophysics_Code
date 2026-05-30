"""
Master test runner for the Petrophysics February 2022 (Vol. 63, No. 1) issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article2_cnn_xcorr_depth_matching.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - In-Situ Raman Composition Logging Tool",     "article1_raman_logging_eor_gas_storage"),
    ("Article 2 - CNN vs Cross-Correlation Depth Matching",    "article2_cnn_xcorr_depth_matching"),
    ("Article 3 - Log Analytics: Cross-Correlation / DTW + QC", "article3_log_analytics_dtw_xcorr"),
    ("Article 4 - Ultrasonic Logging of Creeping Shale",       "article4_ultrasonic_creeping_shale"),
    ("Article 5 - Sand-Injectite Thomas-Stieber Volumetrics",  "article5_sand_injectite_thomas_stieber"),
    ("Article 6 - Closed-Retort Core-Based Quantification",    "article6_closed_retort_core_quant"),
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
