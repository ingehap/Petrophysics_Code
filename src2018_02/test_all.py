"""
Master test runner for the Petrophysics February 2018 (Vol. 59, No. 1) issue -
the "Best Papers of the 2017 SCA International Symposium" (a shaly-sand tutorial,
seven SCA best papers, and two regular submissions).

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article6_densitometer_fluid_volume.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Shaly Sand Tutorial 1 of 3 (Tutorial)",  "article1_shaly_sand_tutorial_part1"),
    ("Article 2 - DRP Blind Study (capillary pressure)",   "article2_drp_blind_study_pc"),
    ("Article 3 - Stress Sensitivity of MICP",             "article3_stress_sensitivity_micp"),
    ("Article 4 - Stress-Dependent Permeability",          "article4_stress_dependent_permeability"),
    ("Article 5 - DRT Relative-Permeability QC",           "article5_drt_relperm_qc"),
    ("Article 6 - Densitometer Fluid Volume",              "article6_densitometer_fluid_volume"),
    ("Article 7 - Salt-Bearing Sediments (digital rock)",  "article7_salt_bearing_digital_rock"),
    ("Article 8 - Core Restoration RSM",                   "article8_core_restoration_rsm"),
    ("Article 9 - Geostress Resistivity Correction",       "article9_geostress_resistivity_correction"),
    ("Article 10 - Shale Gas Adsorption",                  "article10_shale_gas_adsorption"),
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
