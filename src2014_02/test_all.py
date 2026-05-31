"""
Master test runner for the Petrophysics February 2014 (Vol. 55, No. 1) issue -
solving the complex dual-water equation with dielectric-NMR-spectroscopy and
conventional logs, capillary pressure and resistivity-index measurements in a
mixed-wet carbonate, spontaneous imbibition of water into oil-wet carbonate
cores using nanofluid, desorbed canister gas sampling and gas isotopic analysis
of two coalbed-methane wells, and thermal-conductivity estimation from
elastic-wave velocity with a petrographic-coded model.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_dualwater_dielectric_nmr.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Dual-Water Dielectric-NMR-Spectroscopy", "article1_dualwater_dielectric_nmr"),
    ("Article 2 - Pc & Resistivity Index in Carbonate",    "article2_pc_resistivity_index_carbonate"),
    ("Article 3 - Nanofluid Spontaneous Imbibition",       "article3_nanofluid_imbibition"),
    ("Article 4 - Canister Gas Sampling & Isotopes",       "article4_canister_gas_isotopes"),
    ("Article 5 - Thermal Conductivity from Velocity",     "article5_thermal_conductivity_velocity"),
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
