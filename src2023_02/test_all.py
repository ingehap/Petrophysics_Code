"""
Master test runner for the Petrophysics February 2023 (Vol. 64, No. 1) issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article2_carbonate_phi_k.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - RFG + Petroleum System Modeling",            "article1_rfg_petroleum_system"),
    ("Article 2 - Carbonate Rock-Type Permeability",           "article2_carbonate_phi_k"),
    ("Article 3 - Swin-Style Fracture Extraction",             "article3_swin_fracture"),
    ("Article 4 - Hexa-Combo LWD Case Study",                  "article4_hexa_combo_lwd"),
    ("Article 5 - Digital-Core Poisson's Ratio",               "article5_digital_core_poisson"),
    ("Article 6 - LM-EnRML Geosteering",                       "article6_geosteering_enrml"),
    ("Article 7 - Data-Mining Carbonate Permeability",         "article7_dm_permeability"),
    ("Article 8 - Hot-Water Injection T Optimisation",         "article8_hot_water_injection"),
    ("Article 9 - Depth-Matching DTW / CDTW / COW",            "article9_depth_matching"),
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
