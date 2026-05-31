"""
Master test runner for the Petrophysics October 2017 (Vol. 58, No. 5) issue -
the "Best of 2017 SPWLA Symposium" (five articles: mixed-wet saturation, kerogen
2D NMR, gamma-ray tool characterization, near-wellbore joint inversion, and
Permian core analysis).

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article2_kerogen_2d_nmr_bitumen.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Mixed-Wet Saturation (PCM)",             "article1_mixedwet_saturation_pcm"),
    ("Article 2 - Kerogen 2D NMR & Bitumen Extraction",    "article2_kerogen_2d_nmr_bitumen"),
    ("Article 3 - Gamma-Ray Tool API Characterization",    "article3_gamma_ray_api_characterization"),
    ("Article 4 - Joint Inversion Near-Wellbore",          "article4_joint_inversion_nearwellbore"),
    ("Article 5 - Permian Core Analysis (proxy)",          "article5_permian_core_analysis"),
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
