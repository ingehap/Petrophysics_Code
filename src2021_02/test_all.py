"""
Master test runner for the Petrophysics February 2021 (Vol. 62, No. 1) issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article9_dual_nn_permeability_uncertainty.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Mudlogs Net Pay (Tutorial)",            "article1_mudlog_net_pay_tutorial"),
    ("Article 2 - DFA Lateral Gradients & Mixing",        "article2_dfa_lateral_gradients_mixing"),
    ("Article 3 - Marcellus Weak Bedding Planes",         "article3_marcellus_weak_bedding_planes"),
    ("Article 4 - Fracture Fill From Dielectric Imaging", "article4_obm_dielectric_fracture_fill"),
    ("Article 5 - Formation-Tester CO2 Sampling",         "article5_formation_tester_co2_sampling"),
    ("Article 6 - NMR/Resistivity/Pressure Carbonate",    "article6_nmr_resistivity_pressure_carbonate"),
    ("Article 7 - LWD Dual Ultrasonic Slowness",          "article7_lwd_dual_ultrasonic_slowness"),
    ("Article 8 - Multiwell EM 3D Inversion",             "article8_injectite_em_3d_inversion"),
    ("Article 9 - Dual NN Permeability & Uncertainty",    "article9_dual_nn_permeability_uncertainty"),
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
