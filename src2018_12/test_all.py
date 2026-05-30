"""
Master test runner for the Petrophysics December 2018 (Vol. 59, No. 6) issue -
the "Special Issue: Data-Driven Analytics in Logging and Petrophysics" (a
capillary-pressure tutorial plus the Petrophysical Data-Driven Analytics
papers).

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article3_poisson_ratio_functional_network.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Capillary Pressure Pt.3 (Tutorial)",     "article1_capillary_pressure_tutorial_part3"),
    ("Article 2 - Geological Feature Image ML",            "article2_geological_feature_image_ml"),
    ("Article 3 - Poisson's Ratio Functional Network",     "article3_poisson_ratio_functional_network"),
    ("Article 4 - Borehole Resistivity ML Surrogate",      "article4_borehole_resistivity_ml"),
    ("Article 5 - Ultradeep Resistivity Transdim Inversion", "article5_ultradeep_resistivity_transdim_inversion"),
    ("Article 6 - Lithology CNN",                          "article6_lithology_cnn"),
    ("Article 7 - Hydraulic Fracture Optimization",        "article7_hydraulic_fracture_optimization"),
    ("Article 8 - Shallow Learning Sonic Logs",            "article8_shallow_learning_sonic_logs"),
    ("Article 9 - Fluid Optical Database Reconstruction",  "article9_fluid_optical_database_reconstruction"),
    ("Article 10 - ML Well-Log Depth Matching",            "article10_ml_depth_matching"),
    ("Article 11 - Data Preconditioning",                  "article11_data_preconditioning"),
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
