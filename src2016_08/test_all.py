"""
Master test runner for the Petrophysics August 2016 (Vol. 57, No. 4) issue -
five articles: NMR relaxometry in shale, predicting carbonate rock properties
with NMR and radial-basis-function interpolation, drainage capillary pressure
and resistivity index from short-wait porous-plate experiments, spectral
gamma-ray measurement while drilling, and pure matrix GR.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_nmr_relaxometry_shale.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - NMR Relaxometry in Shale",               "article1_nmr_relaxometry_shale"),
    ("Article 2 - Carbonate NMR + RBF Interpolation",      "article2_carbonate_nmr_rbf"),
    ("Article 3 - Short-Wait Porous-Plate Pc & RI",        "article3_porous_plate_pc_ri"),
    ("Article 4 - Spectral Gamma-Ray While Drilling",      "article4_spectral_gr_mwd"),
    ("Article 5 - Pure Matrix GR",                         "article5_pure_matrix_gr"),
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
