#!/usr/bin/env python3
"""
Test Runner for Petrophysics Journal 2024 Vol.65 No.6 Modules
================================================================
Runs test_all() for every module corresponding to each article.
"""
import sys, os, traceback, time

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

MODULES = [
    ("m01_image_rock_properties",     "Image-Based AI Rock Properties (Britton, Cox & Ma)"),
    ("m02_dip_picking",               "AI-Driven Automatic Dip Picking (Perrier et al.)"),
    ("m03_synthetic_borehole_images", "Synthetic Borehole Images from Outcrops (Fornero et al.)"),
    ("m04_well_integrity_ccs",        "Well Integrity for CCS (Valstar et al.)"),
    ("m05_casing_cement_inspection",  "Casing & Cement Inspection (Hawthorn et al.)"),
    ("m06_noise_logging",             "Advanced Noise Logging (Galli & Pirrone)"),
    ("m07_sourceless_density",        "Sourceless Density (Mauborgne et al.)"),
    ("m08_tracer_aquifer_sampling",   "Tracer & Aquifer Sampling for CCS (Taplin et al.)"),
    ("m09_gpc_fluid_properties",      "GPC Fluid Properties from Cuttings (Cely et al.)"),
    ("m10_permeability_prediction",   "Physics-Based Permeability Prediction (Pirrone, Bona & Galli)"),
    ("m11_wettability_adsorption",    "Wettability via Adsorption Isotherms (Silveira de Araujo & Heidari)"),
    ("m12_fracability_evaluation",    "Fracability Evaluation for Tight Sandstones (Qian et al.)"),
    ("m13_perched_water",             "Perched Water Observations (Kostin & Sanchez-Ramirez)"),
]

def test_all():
    """Run test_all() for every module and report results."""
    print("#" * 74)
    print("#  Petrophysics Vol.65 No.6 (December 2024) - Module Test Suite")
    print("#  13 articles, 13 modules, 13 test functions")
    print("#" * 74)
    results = []
    total_start = time.time()
    for i, (mod_name, desc) in enumerate(MODULES, 1):
        t0 = time.time()
        try:
            mod = __import__(mod_name)
            mod.test_all()
            results.append((mod_name, "PASS", time.time()-t0, ""))
        except Exception as e:
            results.append((mod_name, "FAIL", time.time()-t0, str(e)))
            print(f"\n  *** FAIL: {e} ***\n{traceback.format_exc()}")
    total = time.time() - total_start
    print("\n" + "=" * 74)
    print("  SUMMARY")
    print("=" * 74)
    n_pass = sum(1 for _, s, _, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _, _ in results if s == "FAIL")
    for name, status, elapsed, err in results:
        icon = "+" if status == "PASS" else "X"
        suf = f" ({err[:50]})" if err else ""
        print(f"  [{icon}] {name:40s} {status:4s} {elapsed:5.2f}s{suf}")
    print(f"\n  Total: {n_pass} passed, {n_fail} failed, {total:.1f}s elapsed")
    print("=" * 74)
    return n_fail == 0

if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
