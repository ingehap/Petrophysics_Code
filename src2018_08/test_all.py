"""
Master test runner for the Petrophysics August 2018 (Vol. 59, No. 4) issue -
the "Special Issue on Flow Diagnostics" (a capillary-pressure tutorial, six
flow-diagnostics articles, and two regular submissions).

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article2_acoustic_flowrate_model.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Capillary Pressure Pt.1 (Tutorial)",     "article1_capillary_pressure_tutorial_part1"),
    ("Article 2 - Acoustic Flow-Rate Model",               "article2_acoustic_flowrate_model"),
    ("Article 3 - Multiphase PL Holdup Correction",        "article3_multiphase_pl_holdup_correction"),
    ("Article 4 - Ultracompact Flow Array (Doppler)",      "article4_ultracompact_flow_array_doppler"),
    ("Article 5 - Downhole Sand-Production Rate",          "article5_downhole_sand_production_rate"),
    ("Article 6 - Distributed Sensing Flow Monitoring",    "article6_distributed_sensing_flow_monitoring"),
    ("Article 7 - ACG Downhole Surveillance (proxy)",      "article7_acg_downhole_surveillance"),
    ("Article 8 - Crushed-Rock Flow-Regime Permeability",  "article8_crushedrock_flowregime_permeability"),
    ("Article 9 - Chargeability of Porous Rocks",          "article9_chargeability_metallic_particles"),
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
