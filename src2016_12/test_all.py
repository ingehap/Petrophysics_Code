"""
Master test runner for the Petrophysics December 2016 (Vol. 57, No. 6) issue -
five articles plus a technical note: shale fracturing with machine learning,
orthorhombic geomechanics, shale Young's moduli from nanoindentation, 2D NMR
kerogen fluid typing, ultrasonic image permeability, and gamma-ray normalization.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article2_orthorhombic_geomechanics.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Shale Fracturing (ANNIE + ML)",          "article1_shale_fracturing_ml"),
    ("Article 2 - Geomechanics of Orthorhombic Media",     "article2_orthorhombic_geomechanics"),
    ("Article 3 - Shale Young's Moduli (Nanoindentation)", "article3_shale_youngs_nanoindentation"),
    ("Article 4 - 2D NMR Kerogen Fluid Typing",            "article4_2d_nmr_kerogen_fluid_typing"),
    ("Article 5 - Ultrasonic Image Permeability",          "article5_ultrasonic_permeability_carbonate"),
    ("Article 6 - GR Normalization (Technical Note)",      "article6_gr_normalization_haynesville"),
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
