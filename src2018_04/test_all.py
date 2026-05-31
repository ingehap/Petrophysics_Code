"""
Master test runner for the Petrophysics April 2018 (Vol. 59, No. 2) issue - a
shaly-sand tutorial, six formation-evaluation articles, and three regular
submissions implemented as methodology proxies.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article5_shale_total_porosity_elemental.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Shaly Sand Tutorial No.2 (Tutorial)",    "article1_shaly_sand_tutorial_part2"),
    ("Article 2 - Silt in LRLC Pay (Thomas-Stieber)",      "article2_silt_lrlc_thomas_stieber"),
    ("Article 3 - NMR Pore Coupling",                      "article3_nmr_pore_coupling"),
    ("Article 4 - Neutron & X-Ray Imaging",                "article4_neutron_xray_imaging"),
    ("Article 5 - Shale Total Porosity (elemental)",       "article5_shale_total_porosity_elemental"),
    ("Article 6 - Dielectric Matrix Calibration (CRIM)",   "article6_dielectric_matrix_crim_cda"),
    ("Article 7 - Bakken Dielectric Dispersion",           "article7_bakken_dielectric_dispersion"),
    ("Article 8 - 2D Directional Resistivity Imaging",     "article8_2d_directional_resistivity_imaging"),
    ("Article 9 - Downhole Relative Permeability",         "article9_downhole_relative_permeability"),
    ("Article 10 - PNN Lithofacies Identification",        "article10_pnn_lithofacies"),
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
