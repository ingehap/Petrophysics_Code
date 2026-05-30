"""
Master test runner for the Petrophysics June 2019 (Vol. 60, No. 3) issue -
an invited tutorial plus eight regular articles.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article4_shale_tortuosity_permeability_fvm.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Organic-Mudstone Storage (Tutorial)",   "article1_organic_mudstone_storage_tutorial"),
    ("Article 2 - Niutitang Shale Pore & Adsorption",     "article2_niutitang_shale_pore_adsorption"),
    ("Article 3 - Wellsite Tomography Bayesian Inversion", "article3_wellsite_tomography_bayesian"),
    ("Article 4 - Shale Tortuosity & Permeability (FVM)",  "article4_shale_tortuosity_permeability_fvm"),
    ("Article 5 - Fast NMR T1 Measurement",               "article5_fast_nmr_t1"),
    ("Article 6 - Reconsidering Klinkenberg",             "article6_reconsidering_klinkenberg"),
    ("Article 7 - Carbonate m (Vugs & Fractures)",        "article7_carbonate_m_vugs_fractures"),
    ("Article 8 - Perched Water Contacts",                "article8_perched_water_contacts"),
    ("Article 9 - Wellbore Positioning While Drilling",   "article9_wellbore_positioning_lwd"),
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
