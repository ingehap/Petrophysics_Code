"""
Master test runner for the Petrophysics February 2019 (Vol. 60, No. 1) issue -
a tutorial, four "Best of the 2018 Symposium, Part 2" papers, a depth-control
section, and three regular submissions.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article9_azimuthal_gr_geosteering.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Organic-Mudstone Storage Pt.1 (Tutorial)", "article1_organic_mudstone_storage_part1_tutorial"),
    ("Article 2 - Carbonate Net-Pay Cutoffs",                "article2_carbonate_netpay_cutoffs"),
    ("Article 3 - 2D NMR T1-T2 Maps of Shale",               "article3_2d_nmr_t1t2_shale"),
    ("Article 4 - In-Situ Saturation Core Comparison",       "article4_insitu_saturation_core_comparison"),
    ("Article 5 - Composite Cement Well Integrity",          "article5_composite_cement_well_integrity"),
    ("Article 6 - Depth: A Love and Hate Story",             "article6_depth_love_hate_essay"),
    ("Article 7 - Groningen Depth Control",                  "article7_groningen_depth_control"),
    ("Article 8 - Driller's Depth Correction",               "article8_drillers_depth_correction"),
    ("Article 9 - Azimuthal GR Geosteering",                 "article9_azimuthal_gr_geosteering"),
    ("Article 10 - Hydraulic-Fracturing Stress Test",        "article10_hydraulic_fracturing_stress_test"),
    ("Article 11 - Neutron Generators vs Am-Be",             "article11_neutron_generator_vs_ambe"),
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
