"""
Master test runner for the Petrophysics December 2017 (Vol. 58, No. 6) issue - a
log-preparation tutorial and five articles (driller's depth, carbonate pore
structure, 3D-printed sandstone, kerogen density, and core cleaning).

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article3_carbonate_pore_structure_sonic.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Digital Log Preparation (Tutorial)",     "article1_digital_log_preparation"),
    ("Article 2 - Driller's Depth Way-Point Methodology",  "article2_drillers_depth_waypoint"),
    ("Article 3 - Carbonate Pore Structure & Sonic",       "article3_carbonate_pore_structure_sonic"),
    ("Article 4 - 3D Printing Berea Sandstone",            "article4_3d_printing_berea"),
    ("Article 5 - Kerogen Density & Thermal Maturity",     "article5_kerogen_density_maturity"),
    ("Article 6 - Cleaning Methods & Porosity",            "article6_cleaning_methods_porosity"),
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
