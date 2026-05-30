"""
Master test runner for the Petrophysics October 2019 (Vol. 60, No. 5) issue -
the "Best of 2019 SPWLA Symposium" section plus regular submissions.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_tmali_organic_shales.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Thermal Maturity-Adjusted Log Interp",  "article1_tmali_organic_shales"),
    ("Article 2 - Free & Adsorbed Gas in Shale",          "article2_free_adsorbed_gas_shale"),
    ("Article 3 - ML Well-Log Depth Matching",            "article3_ml_depth_matching"),
    ("Article 4 - Net Sand From Images (NN)",             "article4_netsand_borehole_image_nn"),
    ("Article 5 - Log-Soak-Log Imbibition (Tengiz)",      "article5_log_soak_log_imbibition"),
    ("Article 6 - Micro-CT Invasion & Mudcake",           "article6_microct_invasion_mudcake"),
    ("Article 7 - Through-Casing Acoustic Dual-Source",   "article7_through_casing_acoustic_dualsource"),
    ("Article 8 - Unconventional Rock Typing (Permian)",  "article8_unconventional_rock_typing"),
    ("Article 9 - ANN Bulk Density While Drilling",       "article9_ann_bulk_density_drilling"),
    ("Article 10 - Through-Casing Conductivity (TEM)",    "article10_through_casing_tem_conductivity"),
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
