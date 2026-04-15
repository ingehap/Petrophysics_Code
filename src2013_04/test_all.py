"""
Master test runner for the Petrophysics April 2023 (Vol. 64, No. 2) AI/ML
Special Issue.

Runs the test_all() function of every article module and prints a summary.
Run from the package directory:

    python test_all.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1  — DP Electrofacies Clustering",          "article01_electrofacies_dp"),
    ("Article 2  — Image Rock Classification",            "article02_image_rock_classification"),
    ("Article 3  — Symbolic Regression",                  "article03_symbolic_regression"),
    ("Article 4  — ML Methods for Log Prediction",        "article04_log_prediction_ml"),
    ("Article 5  — Outlier Detection / Log Editing",      "article05_outlier_detection"),
    ("Article 6  — Borehole Image Artifact Removal",      "article06_borehole_image_artifacts"),
    ("Article 7  — Sonic Log Imputation w/ Uncertainty",  "article07_sonic_log_imputation"),
    ("Article 8  — Exemplar-Guided Facies Modeling",      "article08_egfm_facies"),
    ("Article 9  — Spatial Data Analytics",               "article09_spatial_analytics"),
    ("Article 10 — ML Induction-Log Deconvolution",       "article10_induction_deconvolution"),
    ("Article 11 — ML Induction-Log Convolution",         "article11_induction_convolution"),
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
