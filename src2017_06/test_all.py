"""
Master test runner for the Petrophysics June 2017 (Vol. 58, No. 3) issue - six
articles: Tuscaloosa Marine Shale NMR, TGIP from magnetic resonance logs, forward
mineral modeling by SVD/ridge regression, recovering elastic properties from rock
fragments, complex-resistivity dispersion logging, and an integrated carbonate
pore-system case study.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article2_tgip_nmr_gas_shale.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Tuscaloosa Marine Shale NMR",            "article1_tms_nmr_characterization"),
    ("Article 2 - TGIP From Magnetic Resonance Logs",      "article2_tgip_nmr_gas_shale"),
    ("Article 3 - Forward Mineral Modeling (SVD/ridge)",   "article3_forward_mineral_svd"),
    ("Article 4 - Elastic Properties From Fragments",      "article4_elastic_from_fragments"),
    ("Article 5 - Complex-Resistivity Dispersion",         "article5_complex_resistivity_dispersion"),
    ("Article 6 - Carbonate Pore System (proxy)",          "article6_carbonate_pore_system"),
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
