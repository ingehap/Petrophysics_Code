"""
Master test runner for the Petrophysics December 2014 (Vol. 55, No. 6) issue -
a review and new dry-clay-parameter shaly-sand method, an inversion-based
workflow for the new-generation oil-based-mud resistivity imager, an
experimental NMR/dielectric study of wettability and saturation in limestone, a
pore-scale evaluation of dielectric measurements in complex pore/grain
structures, the physical basis for a cased-well quantitative gas-saturation
method, and reminiscences on the first commercial array-induction measurement.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_shaly_sand_dry_clay.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Shaly-Sand Models & Dry-Clay Parameters",  "article1_shaly_sand_dry_clay"),
    ("Article 2 - OBM Resistivity Imager Inversion",         "article2_obm_imager_inversion"),
    ("Article 3 - NMR & Dielectric in Limestone",            "article3_nmr_dielectric_limestone"),
    ("Article 4 - Pore-Scale Dielectric Measurements",       "article4_porescale_dielectric"),
    ("Article 5 - Cased-Well Gas-Saturation Analysis",       "article5_cased_well_gas_saturation"),
    ("Article 6 - Array-Induction Geometric Factors",        "article6_array_induction_geometric_factors"),
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
