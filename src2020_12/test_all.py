"""
Master test runner for the Petrophysics December 2020 (Vol. 61, No. 6) issue -
the "Pulsed-Neutron Logging in the 2020s" nuclear-spectroscopy special issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article2_formation_chlorine_salinity.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - History of Nuclear Spectroscopy",       "article1_nuclear_spectroscopy_history"),
    ("Article 2 - Formation Chlorine -> Salinity",        "article2_formation_chlorine_salinity"),
    ("Article 3 - Self-Compensated Spectroscopy",         "article3_self_compensated_spectroscopy"),
    ("Article 4 - C/O & Sigma Saturation (Malaysia)",     "article4_co_sigma_saturation_casestudy"),
    ("Article 5 - Through-Casing TOC & Saturation",       "article5_through_casing_toc_saturation"),
    ("Article 6 - Pulsed-Neutron Gas Pressure",           "article6_pulsed_neutron_gas_pressure"),
    ("Article 7 - Sigma Gas Saturation (Low-Porosity)",   "article7_sigma_gas_saturation_lowporosity"),
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
