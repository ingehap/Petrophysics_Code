"""
Master test runner for the Petrophysics December 2019 (Vol. 60, No. 6) issue -
the "Best of the 2019 Symposium, Part 2" section plus regular submissions.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article3_lwd_permittivity.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Sonic Slowness Deconvolution",          "article1_sonic_inversion_deconvolution"),
    ("Article 2 - Ultrasonic LWD Caliper & Imaging",      "article2_ultrasonic_lwd_imaging"),
    ("Article 3 - Permittivity From LWD Resistivity",     "article3_lwd_permittivity"),
    ("Article 4 - Crushed-Rock GRI+ Workflow",            "article4_crushed_rock_gri_plus"),
    ("Article 5 - NMR Light-HC / Pore Size / Tortuosity", "article5_nmr_lighthc_chalk"),
    ("Article 6 - Magnetic Susceptibility on NMR",        "article6_nmr_magnetic_susceptibility"),
    ("Article 7 - ANN Formation Tops",                    "article7_ann_formation_tops"),
    ("Article 8 - ML Vuggy-Facies Classifiers",           "article8_ml_vuggy_facies_classifiers"),
    ("Article 9 - Gas-Hydrate Joint Inversion",           "article9_gashydrate_inverse_rockphysics"),
    ("Article 10 - Micro/Nanofluidic Transport",          "article10_micronanofluidic_transport_review"),
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
