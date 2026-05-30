"""
Master test runner for the Petrophysics December 2022 (Vol. 63, No. 6) issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article5_dipole_shear_mohr.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - DAS-VSP Full-Waveform Inversion",        "article1_das_vsp_fwi"),
    ("Article 2 - Sourceless LWD Acoustics",               "article2_sourceless_lwd_acoustics"),
    ("Article 3 - UDAR-LWD Geosteering Inversion",         "article3_udar_geosteering"),
    ("Article 4 - Fractured-Carbonate SOM Classifier",     "article4_fractured_carbonate_som"),
    ("Article 5 - Dipole Shear + Mohr-Coulomb Filter",     "article5_dipole_shear_mohr"),
    ("Article 6 - Mineral / Fluid MD Interfaces",          "article6_md_mineral_fluid"),
    ("Article 7 - PSWC Quality via Digital Rock Physics",  "article7_pswc_drp_qc"),
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
