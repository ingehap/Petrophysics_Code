"""
Master test runner for the Petrophysics June 2018 (Vol. 59, No. 3) issue - a
shaly-sand tutorial, five formation-evaluation articles, and three regular
submissions implemented as methodology proxies.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article4_clay_network_resistivity_saturation.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Shaly Sand Tutorial No.3 (Tutorial)",    "article1_shaly_sand_tutorial_part3"),
    ("Article 2 - Matrix-Adjusted Shale Porosity",         "article2_matrix_adjusted_shale_porosity"),
    ("Article 3 - NMR Wettability Index in Shales",        "article3_nmr_wettability_index_shales"),
    ("Article 4 - Clay-Network Resistivity Saturation",    "article4_clay_network_resistivity_saturation"),
    ("Article 5 - Wideband EM DEM Permittivity",           "article5_wideband_em_dem_permittivity"),
    ("Article 6 - Dielectric Response & Continuous CEC",   "article6_dielectric_cec_shaly_sand"),
    ("Article 7 - Carbonate Permeability Heterogeneity",   "article7_carbonate_permeability_heterogeneity"),
    ("Article 8 - Saturation-Height Stress Corrections",   "article8_saturation_height_stress_correction"),
    ("Article 9 - NMR Relaxation of Fe3O4 Nanoparticles",  "article9_nmr_fe3o4_nanoparticle_relaxation"),
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
