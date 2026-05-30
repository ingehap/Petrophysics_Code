"""
Master test runner for the Petrophysics June 2021 (Vol. 62, No. 3) issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article2_nmr_restricted_diffusion.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Tutorial: A Century of Sidewall Coring",   "article1_sidewall_coring_tutorial"),
    ("Article 2 - Pore Size/Tortuosity/Perm From NMR",       "article2_nmr_restricted_diffusion"),
    ("Article 3 - Real-Time Acoustic Velocity (AI)",         "article3_ai_acoustic_velocity"),
    ("Article 4 - ML-Enabled Sonic Shear Processing",        "article4_ml_sonic_shear"),
    ("Article 5 - LWD Co-Located Antenna Anisotropy",        "article5_lwd_colocated_antenna"),
    ("Article 6 - Geosteering & 2D Structural Analysis",     "article6_geosteering_2d_structural"),
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
