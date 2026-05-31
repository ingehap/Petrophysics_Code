"""
Master test runner for the Petrophysics December 2015 (Vol. 56, No. 6) issue -
a carbonate-characterization special issue of four articles: multiscale
heterogeneity and NMR petrophysics of the presalt Sag carbonates (Campos Basin),
presalt carbonate evaluation for Santos Basin, petrophysical characterization of
the bitumen-saturated karsted Grosmont Formation, and rock typing of the giant
Tengiz carbonate field.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article4_tengiz_rock_typing.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Presalt Sag NMR Petrophysics",           "article1_presalt_sag_nmr_petrophysics"),
    ("Article 2 - Santos Presalt Evaluation",              "article2_santos_presalt_evaluation"),
    ("Article 3 - Grosmont Bitumen Carbonates",            "article3_grosmont_bitumen_carbonates"),
    ("Article 4 - Tengiz Carbonate Rock Typing",           "article4_tengiz_rock_typing"),
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
