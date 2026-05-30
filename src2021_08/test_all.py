"""
Master test runner for the Petrophysics August 2021 (Vol. 62, No. 4) issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article4_sonic_dispersion_dpsm.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Tutorial: Thinly Bedded Formations",       "article1_thinly_bedded_petrophysics"),
    ("Article 2 - Depth Matching via Deep Q-Learning",       "article2_depth_matching_deep_q"),
    ("Article 3 - NMR Fluid Substitution of T2",             "article3_nmr_fluid_substitution"),
    ("Article 4 - Sonic Dispersion (Differential-Phase)",    "article4_sonic_dispersion_dpsm"),
    ("Article 5 - Synthetic Sonic Log ML Contest",           "article5_synthetic_sonic_ml_contest"),
    ("Article 6 - Oil-Based-Mud Resistivity Imager",         "article6_obm_resistivity_imager"),
    ("Article 7 - Volcanic Saturation Model (acoustic)",     "article7_volcanic_saturation_gassmann"),
    ("Article 8 - Capillary Pressure vs Resistivity Index",  "article8_capillary_resistivity_index"),
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
