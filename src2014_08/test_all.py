"""
Master test runner for the Petrophysics August 2014 (Vol. 55, No. 4) issue -
the Best of the 2013 SCA Symposium plus two regular contributions: drainage
three-phase relative permeability on oil-wet carbonates, direct hydrodynamic
simulation of multiphase flow in porous rock, multiphase flow imaged under
dynamic conditions with fast X-ray microtomography, the impact of wettability on
residual oil saturation and capillary desaturation curves, dielectric
permittivity as a petrophysical parameter for shales, and a petrophysical
analysis of siliceous-ooze sediments in the More Basin.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_threephase_relperm_carbonate.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Three-Phase Rel-Perm on Oil-Wet Carbonate", "article1_threephase_relperm_carbonate"),
    ("Article 2 - Direct Hydrodynamic Multiphase Simulation", "article2_dhd_multiphase_simulation"),
    ("Article 3 - Micro-CT Haines Jumps & Energy Balance",    "article3_microct_haines_jumps"),
    ("Article 4 - Wettability & Capillary Desaturation",      "article4_wettability_capillary_desaturation"),
    ("Article 5 - Dielectric Permittivity of Shales",         "article5_dielectric_permittivity_shales"),
    ("Article 6 - Siliceous-Ooze (Opal-A) Petrophysics",      "article6_siliceous_ooze_petrophysics"),
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
