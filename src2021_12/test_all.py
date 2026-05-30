"""
Master test runner for the Petrophysics December 2021 (Vol. 62, No. 6) issue
- the "Best Papers of the 2021 Symposium" issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article03_seat_dip_eigenvectors.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Data Quality for ML Models",               "article01_data_quality_ml"),
    ("Article 2 - VAE Mineral Quantification",               "article02_vae_mineral_spectroscopy"),
    ("Article 3 - SEAT Dip Eigenvector Analysis",            "article03_seat_dip_eigenvectors"),
    ("Article 4 - DL Sedimentary Geometry (borehole image)", "article04_borehole_image_cnn_sedimentary"),
    ("Article 5 - Density Breakout Behind Casing",           "article05_density_breakout_behind_casing"),
    ("Article 6 - NanoTags Cuttings Depth Correlation",      "article06_nanotags_cuttings_depth"),
    ("Article 7 - Multistring Isolation (acoustic)",         "article07_multistring_isolation_acoustic"),
    ("Article 8 - Overbalanced-Drilling Correction",         "article08_overbalanced_drilling_correction"),
    ("Article 9 - Integrated Tight-Gas Char. (proxy)",       "article09_tight_gas_neuquen_integrated"),
    ("Article 10 - Resistivity Rock-Physics Wolfcamp (proxy)", "article10_resistivity_rockphysics_wolfcamp"),
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
