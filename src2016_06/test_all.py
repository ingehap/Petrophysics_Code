"""
Master test runner for the Petrophysics June 2016 (Vol. 57, No. 3) issue -
five articles: heterogeneous-carbonate saturation-height models with dynamic
data, combining hydraulic and electrical conductivity for pore-space
characterization, permeability interpretation from wireline formation testing
with effective thickness, multiscale leaky-P removal for shear-wave anisotropy
inversion, and a wireline-depth elastic-stretch correction.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_carbonate_saturation_height.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Carbonate Saturation-Height Models",     "article1_carbonate_saturation_height"),
    ("Article 2 - Hydraulic + Electrical Pore Space",      "article2_hydraulic_electrical_pore_space"),
    ("Article 3 - WFT Permeability & Effective Thickness", "article3_wft_permeability_effective_thickness"),
    ("Article 4 - Shear-Wave Anisotropy + Leaky-P",        "article4_shearwave_anisotropy_leakyP"),
    ("Article 5 - Wireline Depth Elastic-Stretch",         "article5_wireline_depth_elastic_stretch"),
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
