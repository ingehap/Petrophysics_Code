"""
Master test runner for the Petrophysics April 2019 (Vol. 60, No. 2) issue -
the "Best Papers of the 2018 SCA International Symposium" (two tutorials, seven
SCA articles, three regular submissions).

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article2_resistivity_principles_tutorial.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Organic Mudstone Storage Pt.2 (Tutorial)", "article1_organic_mudstone_storage_part2_tutorial"),
    ("Article 2 - Resistivity Principles (Tutorial)",        "article2_resistivity_principles_tutorial"),
    ("Article 3 - Trapped Oil Capillary Desaturation",       "article3_trapped_oil_capillary_desaturation"),
    ("Article 4 - Image-Segmentation Uncertainty",           "article4_image_segmentation_uncertainty"),
    ("Article 5 - NMR Wettability Review",                   "article5_nmr_wettability_review"),
    ("Article 6 - Waterflood Init Wettability",              "article6_waterflood_init_wettability"),
    ("Article 7 - In-Situ Saturation Monitoring (ISSM)",     "article7_issm_saturation_monitoring"),
    ("Article 8 - Intercept Method Rel-Perm",                "article8_intercept_method_relperm"),
    ("Article 9 - Temperature-Array Monitoring",             "article9_temperature_array_monitoring"),
    ("Article 10 - Invasion Zone Log Inversion",             "article10_invasion_zone_log_inversion"),
    ("Article 11 - Loading Effects on Gas Rel-Perm",         "article11_loading_gas_relperm"),
    ("Article 12 - Borehole Acoustic 3D STC & Ray Tracing",  "article12_borehole_acoustic_stc_raytracing"),
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
