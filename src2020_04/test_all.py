"""
Master test runner for the Petrophysics April 2020 (Vol. 61, No. 2) issue -
a hybrid issue: a "Best of the SCA 2019 Symposium" special section (articles
1-6) followed by regular submissions (articles 7-9).

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article3_crushed_rock_klinkenberg.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Critical Gas Saturation by Micro-CT",   "article1_critical_gas_saturation_microct"),
    ("Article 2 - Coupled NMR + Ultrasonic",              "article2_coupled_nmr_ultrasonic"),
    ("Article 3 - Crushed-Rock Klinkenberg Permeability", "article3_crushed_rock_klinkenberg"),
    ("Article 4 - CEC From Dielectric Analysis",          "article4_cec_dielectric_proxy"),
    ("Article 5 - Wettability Upscaling",                 "article5_wettability_upscaling"),
    ("Article 6 - Gas-Condensate Rel-Perm (LBM)",         "article6_gas_condensate_lbm_relperm"),
    ("Article 7 - Shale Imbibition Rel-Perm & Pc",        "article7_shale_imbibition_relperm_pc"),
    ("Article 8 - Spontaneous Imbibition (Mixed-Wet)",    "article8_spontaneous_imbibition_mixedwet"),
    ("Article 9 - Chemical Formation Damage in Shale",    "article9_chemical_formation_damage_shale"),
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
