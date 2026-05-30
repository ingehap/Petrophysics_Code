"""
Master test runner for the Petrophysics October 2018 (Vol. 59, No. 5) issue -
the "Best of 2018 SPWLA Symposium" (a capillary-pressure tutorial, nine
symposium papers, and one regular submission).

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article2_xray_sourceless_density.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Capillary Pressure Pt.2 (Tutorial)",     "article1_capillary_pressure_tutorial_part2"),
    ("Article 2 - X-Ray Sourceless Density Logging",       "article2_xray_sourceless_density"),
    ("Article 3 - Kerogen Properties + Geomechanics",      "article3_kerogen_log_geomechanics"),
    ("Article 4 - Fast Pressure-Decay Permeability",       "article4_fast_pressure_decay_permeability"),
    ("Article 5 - Unsupervised NMR T1-T2 Fluid Volumes",   "article5_unsupervised_nmr_t1t2_fluid_volumes"),
    ("Article 6 - Proxy Stochastic Fluid Sampling",        "article6_proxy_stochastic_fluid_sampling"),
    ("Article 7 - DFA + Gas Chromatography",               "article7_dfa_gas_chromatography"),
    ("Article 8 - Permeability From Rock Fabric (NMR+Elec)", "article8_permeability_nmr_electric_rockfabric"),
    ("Article 9 - Digital Rock Porosity Upscaling",        "article9_digital_rock_porosity_upscaling"),
    ("Article 10 - Resistivity in Mixed-Wet Rocks",        "article10_resistivity_mixedwet_rocks"),
    ("Article 11 - Hierarchical Rock Classification",      "article11_hierarchical_rock_classification"),
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
