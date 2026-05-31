"""
Master test runner for the Petrophysics August 2015 (Vol. 56, No. 4) issue -
four articles plus a technical note: subsurface fluid characterization with NMR
T1-T2 maps and pore-scale imaging, in-situ vapor evaluation via condensed vapor
gamma, gas diffusion into oil with reservoir baffling and tar mats, an
inversion-based interpretation of neutron-induced gamma-ray spectroscopy, and
the Bateman-Konen resistivity-salinity transform.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_nmr_t1t2_fluid_characterization.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - NMR T1-T2 Fluid Characterization",       "article1_nmr_t1t2_fluid_characterization"),
    ("Article 2 - Condensed Vapor Gamma",                  "article2_condensed_vapor_gamma"),
    ("Article 3 - Gas Diffusion into Oil & Tar Mats",      "article3_gas_diffusion_tar_mats"),
    ("Article 4 - Spectroscopy Inversion",                 "article4_spectroscopy_inversion"),
    ("Article 5 - Bateman-Konen (Technical Note)",         "article5_bateman_konen_resistivity_salinity"),
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
