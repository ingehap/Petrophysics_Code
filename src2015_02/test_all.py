"""
Master test runner for the Petrophysics February 2015 (Vol. 56, No. 1) issue -
the Best Papers of the 2014 SCA Symposium plus two regular submissions: onset of
oil mobilization and nonwetting-phase cluster-size distribution, CO2 EOR by
diffusive mixing in fractured reservoirs, coupled multiphase-hydrodynamic / NMR
pore-scale modeling, petrophysical characterization of Permian Wolfcamp pore
space, and recharacterization/validation of through-the-bit-logging tool
measurements.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article3_nmr_pore_scale_modeling.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Oil Mobilization & Clusters",            "article1_oil_mobilization_clusters"),
    ("Article 2 - CO2 EOR by Diffusive Mixing",            "article2_co2_diffusive_mixing"),
    ("Article 3 - NMR Pore-Scale Modeling",                "article3_nmr_pore_scale_modeling"),
    ("Article 4 - Wolfcamp Pore-Space Characterization",   "article4_wolfcamp_pore_space"),
    ("Article 5 - Through-the-Bit-Logging Density",        "article5_through_the_bit_logging"),
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
