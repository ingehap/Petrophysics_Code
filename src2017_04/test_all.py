"""
Master test runner for the Petrophysics April 2017 (Vol. 58, No. 2) issue - five
articles: NMF of NMR T1-T2 maps, mudstone hydrocarbon storage and kinetics, NMR
relaxation and pore size via the shape factor, T1/T2 rock-fluid affinity with
Gassmann substitution, and tar-mat asphaltene phase transition.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article3_nmr_pore_size_shape_factor.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - NMF of NMR T1-T2 Maps",                  "article1_nmf_t1t2_fluid_signatures"),
    ("Article 2 - Mudstone HC Storage & Kinetics",         "article2_mudstone_hc_storage_kinetics"),
    ("Article 3 - NMR Relaxation & Pore Size",             "article3_nmr_pore_size_shape_factor"),
    ("Article 4 - T1/T2 Affinity & Gassmann",              "article4_t1t2_affinity_gassmann"),
    ("Article 5 - Tar-Mat Asphaltene Phase Transition",    "article5_tarmat_asphaltene_phase"),
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
